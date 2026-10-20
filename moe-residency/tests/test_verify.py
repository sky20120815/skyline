import unittest

import numpy as np

from moe_residency.verify import BitExactVerifier, tensor_digest


class TestDigest(unittest.TestCase):
    def test_stable(self):
        a = np.arange(10, dtype=np.float32)
        self.assertEqual(tensor_digest(a), tensor_digest(a.copy()))

    def test_dtype_normalised(self):
        a32 = np.arange(10, dtype=np.float32)
        a64 = a32.astype(np.float64)
        self.assertEqual(tensor_digest(a32), tensor_digest(a64))

    def test_non_contiguous_normalised(self):
        m = np.arange(12, dtype=np.float32).reshape(3, 4)
        # 转置后是非连续视图，但内容相同 ⇒ 摘要必须相同
        self.assertEqual(tensor_digest(m.T), tensor_digest(np.ascontiguousarray(m.T)))
        self.assertEqual(tensor_digest(m.T), tensor_digest(m.T.copy()))

    def test_different_values_differ(self):
        a = np.zeros(8, dtype=np.float32)
        b = np.zeros(8, dtype=np.float32)
        b[0] = 1e-7                              # 极小但非零的差异也要能发现
        self.assertNotEqual(tensor_digest(a), tensor_digest(b))


class TestVerifier(unittest.TestCase):
    def test_equal_runs(self):
        v = BitExactVerifier()
        for name in ("a", "b"):
            v.start(name)
            for i in range(5):
                v.record(name, np.full(4, i, dtype=np.float32))
        ok, info = v.compare("a", "b")
        self.assertTrue(ok)
        self.assertEqual(info["mismatched_steps"], 0)

    def test_detects_mismatch(self):
        v = BitExactVerifier()
        v.start("a"); v.start("b")
        v.record("a", np.zeros(4, dtype=np.float32))
        v.record("b", np.ones(4, dtype=np.float32))
        ok, info = v.compare("a", "b")
        self.assertFalse(ok)
        self.assertEqual(info["mismatched_steps"], 1)
        self.assertEqual(info["first_mismatch"], 0)

    def test_detects_length_mismatch(self):
        v = BitExactVerifier()
        v.start("a"); v.start("b")
        v.record("a", np.zeros(4, dtype=np.float32))
        v.record("a", np.zeros(4, dtype=np.float32))
        v.record("b", np.zeros(4, dtype=np.float32))
        ok, info = v.compare("a", "b")
        self.assertFalse(ok)
        self.assertEqual(info["reason"], "step_count_mismatch")

    def test_assert_raises(self):
        v = BitExactVerifier()
        v.start("a"); v.start("b")
        v.record("a", np.zeros(4, dtype=np.float32))
        v.record("b", np.ones(4, dtype=np.float32))
        with self.assertRaises(AssertionError):
            v.assert_bitexact("a", "b")

    def test_missing_run_raises(self):
        v = BitExactVerifier()
        v.start("a")
        with self.assertRaises(KeyError):
            v.compare("a", "ghost")


if __name__ == "__main__":
    unittest.main()
