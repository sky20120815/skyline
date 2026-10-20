import unittest

from moe_residency.policy import PolicyConfig, ResidencyPolicy, SLRUPolicy, make_policy


class TestPolicyBasics(unittest.TestCase):
    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            PolicyConfig(capacity=4, mode="nope")

    def test_zero_capacity_raises(self):
        with self.assertRaises(ValueError):
            PolicyConfig(capacity=0)

    def test_hit_miss_accounting(self):
        p = ResidencyPolicy(8, PolicyConfig(capacity=3, mode="lru"))
        for _ in range(5):
            p.access(1)
        s = p.stats
        self.assertEqual(s.hits + s.misses, 5)
        self.assertEqual(s.misses, 1)          # 首次访问必然 miss（要搬进来）
        self.assertEqual(s.hits, 4)           # 之后常驻

    def test_lru_respects_capacity(self):
        p = ResidencyPolicy(16, PolicyConfig(capacity=3, mode="lru"))
        for e in range(10):
            p.access(e)
        self.assertLessEqual(len(p.cache), 3)
        self.assertEqual(p.stats.misses, 10)    # 每次都是新的

    def test_lru_evicts_oldest(self):
        p = ResidencyPolicy(16, PolicyConfig(capacity=2, mode="lru"))
        for e in (1, 2, 3):
            p.access(e)
        self.assertIn(3, p.resident())
        self.assertNotIn(1, p.resident())      # 最老的被踢


class TestSLRU(unittest.TestCase):
    def test_promotion_and_probation_eviction(self):
        p = SLRUPolicy(16, PolicyConfig(capacity=4, mode="slru"))
        p.access(1)                            # 进 probation
        p.access(1)                            # 第二次命中 ⇒ 进 protected
        self.assertIn(1, p.protected)
        for e in (2, 3, 4, 5, 6):
            p.access(e)
        self.assertLessEqual(len(p.probation) + len(p.protected), 4)

    def test_protected_survives_churn(self):
        p = SLRUPolicy(32, PolicyConfig(capacity=4, mode="slru"))
        for _ in range(3):
            p.access(7)                        # 变成 protected
        for e in range(20):                    # 大量新条目冲击
            p.access(e)
        self.assertIn(7, p.resident())         # protected 不该被 probation 挤掉


class TestPinned(unittest.TestCase):
    def test_pin_never_evicted(self):
        p = ResidencyPolicy(64, PolicyConfig(capacity=6, mode="pinned", pin_frac=0.5))
        for _ in range(50):
            p.access(3)                        # 让 3 变热
        added = p.refresh_pin()
        self.assertIn(3, added)
        self.assertIn(3, p.pin)
        for e in range(64):                    # 冲击
            p.access(e)
        self.assertIn(3, p.resident())         # pin 住的不会被踢

    def test_pin_only_in_pinned_modes(self):
        p = ResidencyPolicy(16, PolicyConfig(capacity=4, mode="lru"))
        self.assertEqual(p.refresh_pin(), [])


class TestPrefetch(unittest.TestCase):
    def test_prefetch_only_in_prefetch_modes(self):
        p = ResidencyPolicy(16, PolicyConfig(capacity=4, mode="lru"))
        self.assertEqual(p.switch_domain([1, 2, 3]), [])

    def test_prefetch_lands_in_resident(self):
        p = ResidencyPolicy(64, PolicyConfig(capacity=8, mode="prefetch", prefetch_q=5))
        out = p.switch_domain([10, 11, 12, 13, 14])
        self.assertEqual(len(out), 5)
        for e in out:
            self.assertIn(e, p.resident())     # 预取必须进驻留集合，否则会被立刻踢掉
        self.assertEqual(p.stats.prefetched, 5)

    def test_prefetch_used_is_counted(self):
        p = ResidencyPolicy(64, PolicyConfig(capacity=8, mode="prefetch", prefetch_q=3))
        p.switch_domain([20, 21, 22])
        for e in (20, 21, 22):
            p.access(e)
        self.assertEqual(p.stats.prefetch_used, 3)

    def test_prefetch_respects_capacity(self):
        p = ResidencyPolicy(64, PolicyConfig(capacity=4, mode="prefetch", prefetch_q=32))
        p.switch_domain(list(range(100, 132)))
        self.assertLessEqual(len(p.cache), 4)


class TestFactory(unittest.TestCase):
    def test_make_policy_dispatch(self):
        self.assertIsInstance(make_policy(8, PolicyConfig(capacity=2, mode="slru")), SLRUPolicy)
        self.assertIsInstance(make_policy(8, PolicyConfig(capacity=2, mode="lru")), ResidencyPolicy)

    def test_stats_as_dict(self):
        p = make_policy(8, PolicyConfig(capacity=2, mode="lru"))
        p.access(1)
        d = p.stats.as_dict()
        for k in ("hits", "misses", "evictions", "prefetched", "prefetch_used", "hit_rate"):
            self.assertIn(k, d)


if __name__ == "__main__":
    unittest.main()
