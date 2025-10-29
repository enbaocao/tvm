# Generic Hopper Instruction Set Architecture

## Overview

This document describes the **Generic Instruction Set Architecture (ISA)** for NVIDIA Hopper (H100) and Blackwell (B200) GPUs. Unlike the model-specific instruction sets in `demos/low-latency-llama`, this ISA is designed to work across different transformer architectures without recompilation.

## Motivation

The original megakernel implementation is tightly coupled to specific model architectures:
- **Compile-time dimensions**: Model parameters (hidden_dim, num_heads, etc.) are template parameters
- **Fixed opcodes**: Each demo defines its own opcode scheme (7 opcodes for Llama latency mode, 10 for throughput)
- **Model-specific kernels**: `llama.cu` hardcodes Llama-1B dimensions

This limits reusability. To run a different model size or architecture, you must:
1. Modify C++ template parameters
2. Recompile CUDA kernels (5-10 minutes)
3. Create new Python bindings

**The Generic ISA solves this** by moving configuration to runtime while maintaining peak performance.

## Design Principles

1. **Runtime Configuration**: Model dimensions specified at runtime, not compile-time
2. **Primitive Operations**: ~20 fundamental opcodes that compose into any transformer
3. **Flexible Dispatch**: Single compiled kernel handles multiple model architectures
4. **Performance**: <10% overhead vs. hand-tuned specialized kernels
5. **Portability**: Works on both Hopper (H100) and Blackwell (B200) with architecture-specific optimizations

## Architecture Components

### 1. Opcodes (`include/generic/opcodes.cuh`)

Defines 256 possible opcodes organized into categories:

```cpp
// Memory operations (0x10-0x2F)
OP_LOAD_ACTIVATION, OP_STORE_ACTIVATION, OP_CACHE_APPEND

// Linear algebra (0x30-0x4F)
OP_MATMUL, OP_MATVEC, OP_BATCH_MATMUL

// Normalization (0x50-0x6F)
OP_RMS_NORM, OP_LAYER_NORM

// Attention (0x70-0x8F)
OP_ATTENTION_PARTIAL, OP_ATTENTION_REDUCE, OP_ROPE_EMBED

// Activations (0x90-0x9F)
OP_SILU, OP_GELU, OP_RELU, OP_SWIGLU

// Element-wise (0xA0-0xAF)
OP_ADD, OP_RESIDUAL_ADD

// Fused operations (0xB0-0xCF) - for performance
OP_FUSED_NORM_MATMUL, OP_FUSED_ROPE_APPEND, OP_FUSED_QKV_PROJ
```

**Key Feature**: Fused operations allow specialized kernels for common patterns (e.g., `FUSED_NORM_QKV_ROPE` combines RMSNorm + QKV projection + RoPE in one kernel for Llama).

### 2. Runtime Model Configuration (`include/generic/model_config.cuh`)

Replaces compile-time templates with runtime config:

```cpp
struct RuntimeModelConfig {
    // Architecture
    uint16_t num_layers, hidden_dim, intermediate_dim, vocab_size;

    // Attention
    uint16_t num_q_heads, num_kv_heads, head_dim;  // GQA/MQA support
    uint16_t attn_config;  // Encodes MHA/GQA/MQA/MLA + RoPE/ALiBi + masking

    // MLP
    ActivationType mlp_activation;  // SiLU, GELU, etc.
    bool mlp_gated;  // SwiGLU vs standard MLP

    // Normalization
    NormType norm_type;  // RMSNorm, LayerNorm
    float norm_eps;

    // Tiling parameters (tuned per GPU)
    uint16_t matmul_block_m, matmul_block_n, matmul_block_k;

    // Hardware
    uint16_t sm_count;
    bool is_hopper, is_blackwell;
};
```

**Presets provided**:
- `make_llama_3_1b_config()` - Llama 3.2 1B
- `make_gpt2_config()` - GPT-2 124M
- `make_mistral_7b_config()` - Mistral 7B with sliding window attention

### 3. Generic Instruction Format (`include/generic/instruction.cuh`)

64-byte instruction structure (fits in 2 cache lines):

```cpp
struct GenericInstruction {
    uint8_t opcode;
    uint8_t flags;           // Transpose, accumulate, etc.
    uint16_t layer_idx;

    // Runtime dimensions
    uint16_t m_dim, n_dim, k_dim;
    uint16_t block_idx_m, block_idx_n, block_idx_k;

    // Memory addressing (element offsets)
    uint32_t input_offset_0, input_offset_1, input_offset_2;
    uint32_t output_offset, weight_offset, scratch_offset;

    // Configuration
    uint16_t head_config;    // Attention pattern
    uint16_t reduction_factor;
    uint16_t seq_pos, batch_idx;

    // Synchronization
    uint16_t dependency_mask;
    uint8_t sync_slot, sync_count;

    // Metadata
    uint32_t instruction_id;
    float scale_factor;      // For attention scaling, etc.
};
```

**Builder helpers** provided for common patterns:
- `make_matmul_instruction()`
- `make_norm_instruction()`
- `make_attention_partial_instruction()`
- `make_fused_norm_matmul_instruction()`

### 4. Runtime Globals (`include/generic/globals.cuh`)

Unified memory layout supporting dynamic dimensions:

```cpp
template <typename config>
struct RuntimeGlobals {
    RuntimeModelConfig model_cfg;

    // Unified weight buffers (all layers)
    gl<bf16, -1, -1, -1, -1> unified_weights;
    gl<bf16, -1, -1, -1> norm_weights;

    // KV cache (shape: num_layers, 2, num_kv_heads, max_seq_len, head_dim)
    gl<bf16, -1, -1, -1, -1, -1> kv_cache;

    // Position embeddings
    gl<float, -1, -1> rope_cos, rope_sin;
    gl<bf16, -1, -1> pos_embed_table;  // For GPT-2
    gl<float, -1> alibi_slopes;        // For ALiBi

    // Activation buffers
    gl<bf16, -1, -1> hidden_states, residual_states;
    gl<bf16, -1, -1, -1> q_proj, k_proj, v_proj, attn_output;
    gl<bf16, -1, -1> gate_output, up_output, mlp_output;
    gl<bf16, -1, -1> logits;
};
```

### 5. Python Scheduler (`megakernels/generic_scheduler.py`)

Model-agnostic instruction generation:

```python
class UniversalScheduler:
    def __init__(self, model_cfg: ModelConfig):
        self.cfg = model_cfg

    def build_transformer_layer(self, layer_idx: int) -> List[GenericInstruction]:
        """Generate instructions for one transformer layer"""
        # Automatically adapts to model architecture

    def build_full_model(self) -> List[GenericInstruction]:
        """Generate complete instruction sequence"""
```

**Supports multiple architectures out of the box**:
- **Llama**: RMSNorm + GQA + RoPE + SwiGLU
- **GPT-2**: LayerNorm + MHA + learned position embeddings + GELU
- **Mistral**: RMSNorm + GQA + RoPE + sliding window attention + SwiGLU

## Example Usage

### Python Side: Generate Instructions

```python
from megakernels.generic_scheduler import ModelConfig, UniversalScheduler

# Create config for your model
config = ModelConfig.from_llama_3_1b(sm_count=132)
# Or: config = ModelConfig.from_gpt2()
# Or: config = ModelConfig.from_mistral_7b()

# Generate instructions
scheduler = UniversalScheduler(config)
instructions = scheduler.build_full_model(use_fused_ops=True)

print(f"Generated {len(instructions)} instructions")
print(f"First instruction: {Opcode(instructions[0].opcode).name}")

# Serialize for GPU
instruction_buffer = np.stack([inst.serialize() for inst in instructions])
```

### C++ Side: Dispatch Instructions

```cpp
#include "generic/generic.cuh"

using namespace megakernel::generic;

// Create runtime configuration
RuntimeModelConfig cfg = make_llama_3_1b_config();
PerformanceHints::detect_hardware(cfg);  // Auto-detect H100 vs B200

// Allocate buffers
RuntimeGlobals<default_config> globals;
allocate_buffers(globals, cfg, "cuda:0");

// Decode and execute instruction
GenericInstruction inst = decode_instruction(instruction_buffer);

switch (inst.opcode) {
case OP_MATMUL:
    dispatch_matmul_op(globals, mks, inst);
    break;
case OP_ATTENTION_PARTIAL:
    dispatch_attention_partial_op(globals, mks, inst);
    break;
// ... other opcodes
}
```

## Attention Pattern Support

The `head_config` field encodes different attention patterns:

```cpp
// Create attention config
uint16_t config = make_attn_config(
    ATTN_TYPE_GQA,           // Grouped Query Attention
    POS_EMB_ROPE,            // RoPE position embeddings
    MASK_CAUSAL,             // Causal masking
    ATTN_FEAT_FLASH          // Flash attention style
);

// Query at runtime
if (get_attn_type(config) == ATTN_TYPE_GQA) {
    // Use GQA kernel
    int num_groups = num_q_heads / num_kv_heads;
}
```

**Supported patterns**:
- **MHA** (Multi-Head Attention): All models have separate K/V per head
- **GQA** (Grouped Query Attention): Multiple Q heads share K/V heads (Llama 3, Mistral)
- **MQA** (Multi-Query Attention): All Q heads share one K/V head (some efficient models)
- **MLA** (Multi-Latent Attention): Compressed latent attention (DeepSeek-V2)

**Position embeddings**:
- **RoPE**: Rotary position embeddings (Llama, Mistral, etc.)
- **RoPE Partial**: Only rotate first N dimensions (Qwen)
- **ALiBi**: Attention with Linear Biases
- **Learned**: Absolute position embeddings (GPT-2)

**Masking**:
- **Causal**: Standard autoregressive masking
- **Sliding Window**: Local attention with window (Mistral)
- **Block Sparse**: Block-structured sparsity

## Performance Characteristics

### Expected Performance vs. Specialized Kernels

| Model | Specialized Kernel | Generic ISA | Overhead |
|-------|-------------------|-------------|----------|
| Llama 3.2 1B | 100% | 95-98% | 2-5% |
| GPT-2 124M | 100% | 93-97% | 3-7% |
| Mistral 7B | 100% | 94-97% | 3-6% |

**Overhead sources**:
1. **Runtime dispatch** (1-2%): Opcode switch vs. compile-time dispatch
2. **Less aggressive inlining** (1-2%): Generic kernels can't inline as aggressively
3. **Dimension checks** (0-1%): Runtime validation of dimensions

**Mitigation strategies**:
1. **Fused operations**: `FUSED_NORM_QKV_ROPE` combines multiple ops
2. **Template specialization**: Specialize for common dimension combinations
3. **Constexpr dispatch**: Use C++20 `constexpr if` to eliminate branches
4. **Profile-guided optimization**: JIT compile hot paths

### Hopper vs. Blackwell

The ISA is designed for portability:

```cpp
if (cfg.is_blackwell) {
    // B200 optimizations
    cfg.matmul_block_k *= 2;       // More shared memory
    cfg.attn_block_kv = 128;       // vs 64 on H100
    use_5th_gen_tensor_cores();    // New instructions
}
```

**Expected Blackwell speedups**:
- **1.5-2x** from 5th gen tensor cores
- **1.2-1.3x** from larger shared memory (232KB vs 228KB)
- **1.1-1.2x** from higher SM count (148 vs 132)
- **Combined**: ~2-2.5x total

## Current Status

### ✅ Completed (Phase 1-2)

- [x] Core opcode definitions (~20 opcodes)
- [x] Runtime model configuration structure
- [x] Generic instruction format (64 bytes)
- [x] Runtime globals with dynamic dimensions
- [x] Python scheduler for multi-model support
- [x] Model configuration presets (Llama, GPT-2, Mistral)
- [x] Attention pattern encoding (MHA/GQA/MQA)
- [x] Builder helpers for common instructions

### 🚧 In Progress (Phase 3-4)

- [ ] ThunderKittens operation wrappers
- [ ] Generic MatMul kernel
- [ ] Generic RMSNorm / LayerNorm kernel
- [ ] Flexible attention kernel (MHA/GQA/MQA)
- [ ] RoPE embedding kernel
- [ ] Fused operation kernels
- [ ] Consumer/Loader/Storer dispatch

### 📋 Planned (Phase 5-6)

- [ ] Blackwell-specific optimizations
- [ ] Auto-tuning for block sizes
- [ ] Correctness validation (Llama, GPT-2, Mistral)
- [ ] Performance benchmarking
- [ ] Support for Mamba, RWKV (non-attention models)
- [ ] FP8 support for inference
- [ ] Multi-GPU tensor parallelism

## Next Steps

1. **Implement core operation kernels** using ThunderKittens primitives
2. **Create dispatch infrastructure** to route opcodes to kernels
3. **Integrate with existing megakernel framework** (consumer/loader/storer warps)
4. **Validate correctness** against PyTorch reference implementations
5. **Benchmark performance** and optimize hot paths
6. **Add Blackwell support** and test on B200

## File Structure

```
include/generic/
├── opcodes.cuh          # Opcode definitions and flags
├── model_config.cuh     # Runtime model configuration
├── instruction.cuh      # Generic instruction format
├── globals.cuh          # Runtime globals structure
└── generic.cuh          # Main header (includes all)

megakernels/
└── generic_scheduler.py # Python scheduler for instruction generation

demos/
└── generic-hopper/      # (To be created) Demo using generic ISA
    ├── Makefile
    ├── generic_kernel.cu
    └── test_multi_model.py
```

## References

- **ThunderKittens**: `ThunderKittens/README.md`
- **Megakernel Architecture**: `include/megakernel.cuh`
- **Existing Llama Demo**: `demos/low-latency-llama/`
- **Flash Attention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention"
- **GQA**: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models"

## Contributing

To add support for a new model architecture:

1. Define `RuntimeModelConfig` preset in `model_config.cuh`
2. If needed, add new opcodes to `opcodes.cuh` (use 0xF0-0xFF range)
3. Implement operation kernel if unique (most models use existing ops)
4. Add Python factory method in `generic_scheduler.py`
5. Test correctness against PyTorch reference

Expected time to add new architecture: **<48 hours** (vs. weeks for specialized implementation).
