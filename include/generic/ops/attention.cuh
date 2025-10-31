#pragma once

#include "../instruction.cuh"
#include "../opcodes.cuh"
#include "../globals.cuh"
#include <math.h>

namespace megakernel {
namespace generic {

// Minimal attention partial op for smoke testing (single head, decode-style)
template <typename config>
struct OpAttentionPartial {
    static constexpr int opcode = OP_ATTENTION_PARTIAL;

    struct consumer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            GenericInstruction inst{};
            inst.deserialize_from(mks.instruction());

            if (::kittens::laneid() == 0) {
                const int head_dim = inst.k_dim;
                const int kv_len = (inst.reduction_factor > 0) ? inst.reduction_factor : 1;
                const float scale = inst.scale_factor == 0.0f ? 1.0f : inst.scale_factor;

                const float *q = g.template ptr_input0<const float>(inst.input_offset_0);
                const float *k = g.template ptr_input1<const float>(inst.input_offset_1);
                const float *v = g.template ptr_input2<const float>(inst.input_offset_2);
                float *o = g.template ptr_output<float>(inst.output_offset);

                // Compute logits = (q · K_j) * scale
                // For kv_len == 1, softmax weight is 1.
                extern __shared__ float smem[]; // not actually used; placeholder
                float max_logit = -INFINITY;
                // First pass: compute logits and track max
                for (int j = 0; j < kv_len; ++j) {
                    float dot = 0.f;
                    const float *kj = k + j * head_dim;
                    #pragma unroll 1
                    for (int d = 0; d < head_dim; ++d) dot += q[d] * kj[d];
                    float logit = dot * scale;
                    if (logit > max_logit) max_logit = logit;
                }
                // Second pass: softmax denom
                float denom = 0.f;
                for (int j = 0; j < kv_len; ++j) {
                    float dot = 0.f;
                    const float *kj = k + j * head_dim;
                    #pragma unroll 1
                    for (int d = 0; d < head_dim; ++d) dot += q[d] * kj[d];
                    float logit = dot * scale;
                    denom += expf(logit - max_logit);
                }
                // Output = sum_j softmax_j * V_j
                for (int d = 0; d < head_dim; ++d) o[d] = 0.f;
                for (int j = 0; j < kv_len; ++j) {
                    float dot = 0.f;
                    const float *kj = k + j * head_dim;
                    #pragma unroll 1
                    for (int d = 0; d < head_dim; ++d) dot += q[d] * kj[d];
                    float logit = dot * scale;
                    float w = expf(logit - max_logit) / denom;
                    const float *vj = v + j * head_dim;
                    #pragma unroll 1
                    for (int d = 0; d < head_dim; ++d) o[d] += w * vj[d];
                }
            }
        }
    };
};

} // namespace generic
} // namespace megakernel

