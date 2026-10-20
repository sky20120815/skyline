import unittest

import numpy as np

from moe_residency.plugin import ResidencyPlugin, load_plugin


class TestLifecycle(unittest.TestCase):
    def setUp(self):
        self.p = ResidencyPlugin(n_experts=64, capacity=8, mode="prefetch", prefetch_q=4)

    def test_first_visit_no_prefetch(self):
        self.assertEqual(self.p.before_route(domain="code"), [])   # 画像还是空的

    def test_revisit_domain_prefetches(self):
        # 小缓存 + 大量访问 ⇒ 早期的"热点"会被淘汰，切回时才真的需要预取
        p = ResidencyPlugin(n_experts=64, capacity=2, mode="prefetch", prefetch_q=4)
        p.before_route(domain="code")                             # 建立 current_domain
        for _ in range(20):                                       # 积累 code 的画像并冲刷缓存
            p.on_route(list(range(40)))
            p.after_token()
        p.before_route(domain="chat")                             # 切走
        out = p.before_route(domain="code")                        # 切回 ⇒ 应预取该域热点
        self.assertTrue(out, "切回已建画像的域时应触发预取")
        for e in out:
            self.assertIn(e, p.policy.resident())                 # 预取必须进驻留集合

    def test_no_prefetch_for_resident_hotset(self):
        """热点其实还在缓存里时，不该做无谓的预取（避免浪费带宽）。"""
        p = ResidencyPlugin(n_experts=64, capacity=8, mode="prefetch", prefetch_q=4)
        p.before_route(domain="code")
        p.on_route([1, 2, 3])
        p.before_route(domain="chat")
        self.assertEqual(p.before_route(domain="code"), [])       # 都还在缓存 ⇒ 无需预取

    def test_same_domain_no_reprefetch(self):
        for _ in range(5):
            self.p.on_route([1, 2, 3])
        self.p.before_route(domain="code")
        self.assertEqual(self.p.before_route(domain="code"), [])   # 同一域不重复预取

    def test_on_route_reports_misses(self):
        r = self.p.on_route([9, 10, 11])
        self.assertEqual(set(r), {9, 10, 11})
        self.assertTrue(all(r.values()))                           # 首次都是 miss
        r2 = self.p.on_route([9, 10, 11])
        self.assertFalse(any(r2.values()))                         # 之后是 hit

    def test_stats_shape(self):
        self.p.on_route([1, 2])
        self.p.after_token()
        s = self.p.stats()
        for k in ("plugin", "mode", "capacity", "policy", "domains", "resident"):
            self.assertIn(k, s)
        self.assertEqual(s["mode"], "prefetch")

    def test_verify_requires_verifier(self):
        with self.assertRaises(RuntimeError):
            self.p.record(np.zeros(3, dtype=np.float32))

    def test_verify_roundtrip(self):
        from moe_residency.verify import BitExactVerifier
        v = BitExactVerifier()
        a = ResidencyPlugin(8, capacity=2, verifier=v, run_name="a")
        b = ResidencyPlugin(8, capacity=2, verifier=v, run_name="b")
        for _ in range(3):
            a.record(np.ones(4, dtype=np.float32))
            b.record(np.ones(4, dtype=np.float32))
        b.assert_bitexact("a")                                    # 不抛异常即通过

    def test_profile_persisted_on_after_token(self):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "p.json")
            p = ResidencyPlugin(8, capacity=2, profile_path=path)
            for i in range(3):
                p.on_route([i]); p.after_token()
            p.profile.save(path)                                    # 显式保存应可用
            self.assertTrue(os.path.exists(path))


class TestManifest(unittest.TestCase):
    def test_manifest_fields(self):
        m = ResidencyPlugin.manifest()
        for k in ("name", "version", "entry_point", "capabilities", "modes", "license"):
            self.assertIn(k, m)
        self.assertEqual(m["name"], "moe-residency")
        self.assertIn("prefetch", m["modes"])

    def test_load_plugin_callable(self):
        p = load_plugin(n_experts=8, capacity=2)
        self.assertIsInstance(p, ResidencyPlugin)


if __name__ == "__main__":
    unittest.main()
