import math
import unittest


def rms_norm(x, gamma=None, eps=1e-5):
    n = len(x)
    sum_sq = sum(v * v for v in x)
    rms = math.sqrt(sum_sq / n + eps)
    inv = 1.0 / rms
    if gamma is None:
        gamma = [1.0] * n
    return [(x[i] * inv) * gamma[i] for i in range(n)]


def matmul_vec(W_rows, x):
    # W_rows: list of rows (N x K), x: length K
    out = []
    for row in W_rows:
        out.append(sum(w * xi for w, xi in zip(row, x)))
    return out


def fused_norm_matmul(x, gamma, W_rows, eps=1e-5):
    xh = rms_norm(x, gamma, eps)
    return matmul_vec(W_rows, xh)


def rope_rotate(x, angle):
    # Pairwise rotation across even-odd dims
    out = list(x)
    c, s = math.cos(angle), math.sin(angle)
    for d in range(0, len(x), 2):
        x0 = x[d]
        x1 = x[d + 1] if d + 1 < len(x) else 0.0
        out[d] = x0 * c - x1 * s
        if d + 1 < len(x):
            out[d + 1] = x0 * s + x1 * c
    return out


def attention_decode(q, K_rows, V_rows, scale=1.0):
    logits = [sum(qi * kij for qi, kij in zip(q, k)) * scale for k in K_rows]
    max_logit = max(logits) if logits else 0.0
    exps = [math.exp(l - max_logit) for l in logits]
    denom = sum(exps) if exps else 1.0
    weights = [e / denom for e in exps]
    # Weighted sum of V rows
    out = [0.0] * len(V_rows[0])
    for w, v in zip(weights, V_rows):
        for i, vi in enumerate(v):
            out[i] += w * vi
    return out


class TestReferenceOps(unittest.TestCase):
    def test_fused_norm_matmul_reference(self):
        x = [1.0, 2.0, -1.0]
        gamma = [1.0, 0.5, 2.0]
        W = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        y = fused_norm_matmul(x, gamma, W, eps=0.0)
        # Manual reference
        n = len(x)
        rms = math.sqrt(sum(v * v for v in x) / n)
        xn = [(x[i] / rms) * gamma[i] for i in range(n)]
        expected = matmul_vec(W, xn)
        self.assertEqual(len(y), len(expected))
        for a, b in zip(y, expected):
            self.assertAlmostEqual(a, b, places=7)

    def test_rope_rotate(self):
        x = [2.0, 3.0]
        y = rope_rotate(x, math.pi / 2.0)
        self.assertAlmostEqual(y[0], -3.0, places=7)
        self.assertAlmostEqual(y[1], 2.0, places=7)

    def test_rope_zero_angle_odd_dim(self):
        x = [2.0, 3.0, -4.0]
        y = rope_rotate(x, 0.0)
        self.assertEqual(y, x)

    def test_attention_decode_small(self):
        q = [1.0, 0.0, 0.0]
        K = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        V = [
            [10.0, 0.0, 0.0],
            [0.0, 20.0, 0.0],
        ]
        y = attention_decode(q, K, V, scale=1.0)
        # Softmax([1,0])
        import math as _m

        w0 = 1.0 / (1.0 + _m.exp(-1.0))
        w1 = 1.0 - w0
        expected = [w0 * 10.0 + w1 * 0.0, w1 * 20.0, 0.0]
        for a, b in zip(y, expected):
            self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()

