# Megakernels Quick Start Guide

**Tested and Working on H100 SXM5**

## Prerequisites

- NVIDIA GPU (H100, A100, or RTX 4090)
- CUDA 12.x
- Python 3.10+
- Ubuntu/Linux

## Complete Setup (15 minutes)

### 1. Clone and Setup Repo

```bash
git clone <your-repo-url>
cd tvm
git submodule update --init --recursive
```

### 2. Create Python Environment

```bash
# Use system Python 3.10 (or whatever version you have)
python3 -m venv venv310
source venv310/bin/activate
```

### 3. Install Dependencies

```bash
pip install -e .
```

### 4. HuggingFace Authentication

```bash
# Install HuggingFace CLI
pip install -U "huggingface_hub[cli]"

# Login with your token
huggingface-cli login
```

**Get your token:** https://huggingface.co/settings/tokens

**Accept Llama license:** https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

### 5. Compile CUDA Kernel

```bash
# Set environment variables (adjust PYTHON_VERSION to match your python)
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)
export PYTHON_VERSION=3.10  # Must match: python --version
export GPU=H100             # Options: H100, A100, 4090, B200

# Compile
cd demos/low-latency-llama
make
cd ../..
```

**Compilation time:** ~2-5 minutes

**Expected output:**
```
ptxas info    : Used 96 registers, used 16 barriers...
nvlink info    : 0 bytes gmem
```

You should see: `mk_llama.cpython-310-x86_64-linux-gnu.so`

### 6. Test It Works!

```bash
python megakernels/scripts/generate.py mode=mk prompt="Hello world" ntok=10
```

**Expected output:**
```
Average time: 8.70ms
Tokens per second: 1034.04
Output text: ["!\n\nI'm excited to share my first post on"]
```

🎉 **Success!** You're generating 1000+ tokens/second on H100!

## Common Commands

```bash
# Generate text with megakernel
python megakernels/scripts/generate.py mode=mk prompt="Tell me a story" ntok=50

# Compare megakernel vs PyTorch
python megakernels/scripts/diff_test.py

# Interactive REPL
python megakernels/scripts/llama_repl.py

# Benchmark performance
python megakernels/scripts/generate.py mode=mk ntok=100
```

## Troubleshooting

### Error: "No module named 'mk_llama'"

**Cause:** Python version mismatch between compilation and runtime

**Fix:**
```bash
# Check your Python version
python --version

# Set PYTHON_VERSION to match
export PYTHON_VERSION=3.10  # or 3.11, 3.12, etc.

# Recompile
cd demos/low-latency-llama
make clean
make
cd ../..
```

### Error: "cannot find -lpython3.XX"

**Cause:** Python dev libraries not installed

**Fix:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev

# Or use the system Python version that has dev libs
python3.10 -m venv venv310
```

### Error: "pybind11/pybind11.h: No such file"

**Fix:**
```bash
pip install pybind11
```

### Error: "401 Client Error: Unauthorized"

**Cause:** Need HuggingFace authentication for Llama model

**Fix:**
```bash
huggingface-cli login
# Then accept license at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
```

## GPU-Specific Settings

### H100 (Default)
```bash
export GPU=H100
```

### A100
```bash
export GPU=A100
```

### RTX 4090
```bash
export GPU=4090
```

### B200
```bash
# Don't set GPU variable - defaults to B200
unset GPU
```

## Recompiling After Code Changes

```bash
# Always recompile after modifying .cu files
cd demos/low-latency-llama
make clean
make
cd ../..
```

## Performance Benchmarks (H100)

| Model | Tokens/Second | Latency |
|-------|---------------|---------|
| Llama 3.2 1B | ~1000 | ~8.7ms |
| Llama 3.2 3B | ~TBD | ~TBD |

## Next Steps

- See `MEGAKERNELS_MVP_PLAN.md` for the new instruction set design
- See `GENERIC_ISA.md` for multi-model architecture
- See `COMPILE_INSTRUCTIONS.md` for detailed troubleshooting

## Quick Reference Card

**Initial setup:**
```bash
git submodule update --init --recursive
python3 -m venv venv310 && source venv310/bin/activate
pip install -e .
huggingface-cli login
```

**Compile:**
```bash
export PYTHON_VERSION=3.10 THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens MEGAKERNELS_ROOT=$(pwd) GPU=H100
cd demos/low-latency-llama && make && cd ../..
```

**Run:**
```bash
python megakernels/scripts/generate.py mode=mk prompt="test" ntok=10
```

That's it! 🚀
