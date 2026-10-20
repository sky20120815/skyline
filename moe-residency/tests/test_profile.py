import os
import tempfile
import unittest

from moe_residency.profile import DomainProfile


class TestDomainProfile(unittest.TestCase):
    def test_hint_unknown_domain_empty(self):
        p = DomainProfile(16)
        self.assertEqual(p.hint("nope", q=4), [])

    def test_note_and_hint_order(self):
        p = DomainProfile(16, ema=0.5)
        for _ in range(10):
            p.note("code", [3, 5])
        p.note("chat", [7])
        self.assertEqual(p.hint("code", q=2), [3, 5])
        self.assertEqual(p.hint("chat", q=2), [7])

    def test_ema_decay_drops_stale(self):
        p = DomainProfile(16, ema=0.1)
        for _ in range(20):
            p.note("code", [1])
        for _ in range(40):
            p.note("code", [2])
        # 1 的计数应被衰减到很低，2 成为热点
        self.assertEqual(p.hint("code", q=1), [2])

    def test_none_domain_ignored(self):
        p = DomainProfile(8)
        p.note(None, [1, 2])
        self.assertEqual(p.domains(), [])

    def test_save_load_roundtrip(self):
        p = DomainProfile(16, ema=0.9)
        for _ in range(5):
            p.note("code", [1, 2, 3])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nested", "profile.json")
            p.save(path)
            q = DomainProfile.load(path)
        self.assertEqual(q.n, 16)
        self.assertEqual(q.hint("code", q=3), p.hint("code", q=3))


if __name__ == "__main__":
    unittest.main()
