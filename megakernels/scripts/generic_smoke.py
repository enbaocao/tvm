#!/usr/bin/env python3
import argparse
import math
import sys


def require_torch():
    try:
        import torch  # noqa: F401
    except Exception:
        print("ERROR: This smoke test requires PyTorch. Please install torch and try again.")
        sys.exit(2)


def require_extension():
    try:
        import importlib
        importlib.import_module("mk_generic")
    except Exception as e:
        print("ERROR: mk_generic extension not found. Build it with:\n"
              "  export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens\n"
              "  export MEGAKERNELS_ROOT=$(pwd)\n"
              "  export PYTHON_VERSION=$(python3 -c 'import sys;print(f\"{sys.version_info[0]}.{sys.version_info[1]}\")')\n"
              "  export GPU=H100  # or 4090/A100; unset for B200\n"
              "  cd demos/generic-hopper && make && cd ../..\n"
              f"Details: {e}")
        sys.exit(3)


def run_matmul():
    import torch
    import importlib
    mk_generic = importlib.import_module("mk_generic")
    device = torch.device("cuda")

    # opcode=0x30: MATMUL, m=1, n=4, k=3
    inst = torch.zeros(32, dtype=torch.int32)
    inst[0] = (0x30)
    inst[1] = (1) | (4 << 16)
    inst[2] = (3)
    inst[4] = 0  # a offset
    inst[8] = 0  # b (weight) offset
    inst[7] = 0  # c offset

    instructions = inst.view(1, 1, 32).contiguous().to(device)
    timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    b = torch.tensor([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 1.0, 1.0,
    ], dtype=torch.float32, device=device)
    c = torch.zeros(4, dtype=torch.float32, device=device)

    mk_generic.mk_generic_matmul(instructions, timings, a, b, c)
    expected = torch.tensor([1.0, 2.0, 3.0, 6.0], device=device)
    ok = torch.allclose(c, expected, atol=1e-6)
    print(f"MATMUL smoke: {'PASS' if ok else 'FAIL'}; got {c.cpu().tolist()}")
    return 0 if ok else 1


def run_rmsnorm():
    import torch
    import importlib
    mk_generic = importlib.import_module("mk_generic")
    device = torch.device("cuda")

    n = 4
    inst = torch.zeros(32, dtype=torch.int32)
    inst[0] = (0x50)  # RMS_NORM
    inst[1] = (1) | (n << 16)
    inst[4] = 0  # x offset in a
    inst[8] = n  # gamma offset in b (we pack after zeros)
    inst[7] = 0  # y offset in c
    inst[15] = torch.tensor(1e-5, dtype=torch.float32).view(torch.int32)

    instructions = inst.view(1, 1, 32).contiguous().to(device)
    timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    gamma = torch.ones(n, dtype=torch.float32, device=device)
    a = x.clone()
    b = torch.cat([torch.zeros(n, device=device), gamma])
    c = torch.zeros(n, device=device)

    mk_generic.mk_generic_matmul(instructions, timings, a, b, c)
    rms = torch.sqrt((x.pow(2).mean()) + 1e-5)
    expected = x / rms
    ok = torch.allclose(c, expected, atol=1e-5)
    print(f"RMS_NORM smoke: {'PASS' if ok else 'FAIL'}; got {c.cpu().tolist()}")
    return 0 if ok else 1


def run_attention():
    import torch
    import importlib
    mk_generic = importlib.import_module("mk_generic")
    device = torch.device("cuda")

    head_dim = 3
    kv_len = 2
    inst = torch.zeros(32, dtype=torch.int32)
    inst[0] = (0x70)  # ATTENTION_PARTIAL
    inst[2] = (head_dim)
    inst[10] = (0) | (kv_len << 16)
    inst[4] = 0  # q in a
    inst[5] = 0  # k in b
    inst[6] = head_dim * kv_len  # v in b after k
    inst[7] = 0
    inst[15] = torch.tensor(1.0, dtype=torch.float32).view(torch.int32)

    instructions = inst.view(1, 1, 32).contiguous().to(device)
    timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

    q = torch.tensor([1.0, 0.0, 0.0], device=device)
    K = torch.tensor([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    ], device=device)
    V = torch.tensor([
        10.0, 0.0, 0.0,
        0.0, 20.0, 0.0,
    ], device=device)
    a = q.clone()
    b = torch.cat([K, V])
    c = torch.zeros(head_dim, device=device)

    mk_generic.mk_generic_matmul(instructions, timings, a, b, c)
    logits = torch.tensor([1.0, 0.0], device=device)
    w = torch.softmax(logits, dim=0)
    expected = w[0] * V[:head_dim] + w[1] * V[head_dim:]
    ok = torch.allclose(c, expected, atol=1e-5)
    print(f"ATTENTION_PARTIAL smoke: {'PASS' if ok else 'FAIL'}; got {c.cpu().tolist()}")
    return 0 if ok else 1


def run_rope():
    import torch
    import importlib
    mk_generic = importlib.import_module("mk_generic")
    device = torch.device("cuda")

    inst = torch.zeros(32, dtype=torch.int32)
    inst[0] = (0x72)  # ROPE_EMBED
    inst[2] = (2)     # head_dim=2
    inst[4] = 0
    inst[7] = 0
    inst[15] = torch.tensor(math.pi/2, dtype=torch.float32).view(torch.int32)

    instructions = inst.view(1, 1, 32).contiguous().to(device)
    timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

    x = torch.tensor([2.0, 3.0], device=device)
    a = x.clone()
    b = torch.zeros(1, device=device)
    c = torch.zeros(2, device=device)

    mk_generic.mk_generic_matmul(instructions, timings, a, b, c)
    expected = torch.tensor([-3.0, 2.0], device=device)
    ok = torch.allclose(c, expected, atol=1e-5)
    print(f"ROPE_EMBED smoke: {'PASS' if ok else 'FAIL'}; got {c.cpu().tolist()}")
    return 0 if ok else 1


def run_fused_norm_matmul():
    import torch
    import importlib
    mk_generic = importlib.import_module("mk_generic")
    device = torch.device("cuda")

    # Dimensions: K=3, N=2
    K, N = 3, 2

    # Instruction: OP_FUSED_NORM_MATMUL (0xB0)
    inst = torch.zeros(32, dtype=torch.int32)
    inst[0] = (0xB0)
    inst[1] = (1) | (N << 16)  # m=1, n=N
    inst[2] = (K)              # k_dim=K
    inst[4] = 0  # x in a
    inst[5] = 0  # gamma in b
    inst[8] = K  # W in b after gamma
    inst[7] = 0  # y in c
    inst[15] = torch.tensor(0.0, dtype=torch.float32).view(torch.int32)  # eps

    instructions = inst.view(1, 1, 32).contiguous().to(device)
    timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    # Choose gamma so that norm(x)*gamma == x (gamma = rms(x))
    rms = torch.sqrt((x.pow(2).mean()))
    gamma = torch.tensor([rms.item(), rms.item(), rms.item()], device=device)
    W = torch.tensor([
        1.0, 0.0, 0.0,
        1.0, 1.0, 1.0,
    ], device=device)

    a = x.clone()
    b = torch.cat([gamma, W])
    c = torch.zeros(N, device=device)

    mk_generic.mk_generic_matmul(instructions, timings, a, b, c)

    expected = torch.tensor([
        x[0].item(),
        (x[0] + x[1] + x[2]).item(),
    ], device=device)
    ok = torch.allclose(c, expected, atol=1e-6)
    print(f"FUSED_NORM_MATMUL smoke: {'PASS' if ok else 'FAIL'}; got {c.cpu().tolist()}")
    return 0 if ok else 1


def run_fused_qkv_rope():
    import torch
    import importlib
    mk_generic = importlib.import_module("mk_generic")
    device = torch.device("cuda")

    # For now, our OP_FUSED_NORM_QKV_ROPE behaves like fused norm+matmul in smoke
    K, N = 3, 5
    inst = torch.zeros(32, dtype=torch.int32)
    inst[0] = (0xB5)
    inst[1] = (1) | (N << 16)
    inst[2] = (K)
    inst[4] = 0
    inst[5] = 0
    inst[8] = K
    inst[7] = 0
    inst[15] = torch.tensor(0.0, dtype=torch.float32).view(torch.int32)

    instructions = inst.view(1, 1, 32).contiguous().to(device)
    timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

    x = torch.tensor([1.0, 2.0, -1.0], device=device)
    gamma = torch.tensor([1.0, 0.5, 2.0], device=device)
    W = torch.tensor([
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 1.0, 1.0,
        2.0, -1.0, 0.0,
    ], device=device)

    a = x.clone()
    b = torch.cat([gamma, W])
    c = torch.zeros(N, device=device)

    mk_generic.mk_generic_matmul(instructions, timings, a, b, c)

    rms = torch.sqrt((x.pow(2).mean()))
    xn = x / rms * gamma
    Wm = W.view(N, K)
    expected = (Wm @ xn).contiguous()
    ok = torch.allclose(c, expected, atol=1e-6)
    print(f"FUSED_NORM_QKV_ROPE smoke: {'PASS' if ok else 'FAIL'}; got {c.cpu().tolist()}")
    return 0 if ok else 1

def main():
    parser = argparse.ArgumentParser(description="Generic ISA CUDA smoke tests")
    parser.add_argument(
        "op",
        choices=[
            "matmul",
            "rmsnorm",
            "attention",
            "rope",
            "fused-norm-matmul",
            "fused-qkv-rope",
            "all",
        ],
        help="Which smoke test to run",
    )
    args = parser.parse_args()

    require_torch()
    require_extension()

    rc = 0
    if args.op in ("matmul", "all"):
        rc |= run_matmul()
    if args.op in ("rmsnorm", "all"):
        rc |= run_rmsnorm()
    if args.op in ("attention", "all"):
        rc |= run_attention()
    if args.op in ("rope", "all"):
        rc |= run_rope()
    if args.op in ("fused-norm-matmul", "all"):
        rc |= run_fused_norm_matmul()
    if args.op in ("fused-qkv-rope", "all"):
        rc |= run_fused_qkv_rope()
    sys.exit(rc)


if __name__ == "__main__":
    main()
