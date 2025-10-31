#pragma once

#include "../instruction.cuh"
#include "../opcodes.cuh"
#include "../globals.cuh"

namespace megakernel {
namespace generic {

// Generic RMSNorm operation (scaffold)
template <typename config>
struct OpRmsNorm {
    static constexpr int opcode = OP_RMS_NORM;

    struct consumer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            GenericInstruction inst{};
            inst.deserialize_from(mks.instruction());
            // TODO: Implement RMSNorm over n_dim with epsilon = inst.scale_factor
            (void)g; (void)inst;
        }
    };

    struct loader {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            (void)g; (void)mks;
        }
    };

    struct storer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            (void)g; (void)mks;
        }
    };

    struct launcher {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            (void)g; (void)mks;
        }
    };
};

// Generic LayerNorm operation (scaffold)
template <typename config>
struct OpLayerNorm {
    static constexpr int opcode = OP_LAYER_NORM;

    struct consumer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            GenericInstruction inst{};
            inst.deserialize_from(mks.instruction());
            // TODO: Implement LayerNorm over n_dim with epsilon = inst.scale_factor
            (void)g; (void)inst;
        }
    };

    struct loader {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            (void)g; (void)mks;
        }
    };

    struct storer {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            (void)g; (void)mks;
        }
    };

    struct launcher {
        template <typename globals>
        __device__ static inline void run(const globals &g,
                                           ::megakernel::state<config> &mks) {
            (void)g; (void)mks;
        }
    };
};

} // namespace generic
} // namespace megakernel

