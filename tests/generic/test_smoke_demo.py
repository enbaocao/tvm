import os
import unittest
import importlib


class TestGenericSmokeDemo(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("RUN_GENERIC_SMOKE", "0") == "1", "Set RUN_GENERIC_SMOKE=1 to enable CUDA smoke test")
    def test_matmul_smoke(self):
        # Optional CUDA smoke test; requires compiled extension and torch CUDA
        try:
            import torch
        except Exception:
            self.skipTest("PyTorch not installed")
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        try:
            mk_generic = importlib.import_module("mk_generic")
        except Exception as e:
            self.skipTest(f"mk_generic not importable: {e}")

        device = torch.device("cuda")

        # Build a single MATMUL instruction (packed 32-int format)
        # opcode=0x30 (MATMUL), flags=0, layer_idx=0
        # m=1, n=4, k=3; offsets: a@0, b@0, c@0
        inst = torch.zeros(32, dtype=torch.int32)
        inst[0] = (0x30) | (0 << 8) | (0 << 16)
        inst[1] = (1) | (4 << 16)
        inst[2] = (3) | (0 << 16)
        inst[3] = (0) | (0 << 16)
        inst[4] = 0  # input_offset_0
        inst[8] = 0  # weight_offset
        inst[7] = 0  # output_offset

        instructions = inst.view(1, 1, 32).contiguous().to(device)
        timings = torch.zeros((1, 1, 128), dtype=torch.int32, device=device)

        # Data: a = [1,2,3], b rows = [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]
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
        self.assertTrue(torch.allclose(c, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

