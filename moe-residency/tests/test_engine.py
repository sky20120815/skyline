"""★ 核心不变量：**分层放置只改速度，不改语义**。

这是整个仓库存在的理由，所以它必须是测试，而不是 demo。
"""
import tempfile
import unittest

import numpy as np

from moe_residency.engine import ReferenceMoE, TieredExpertStore
from moe_residency.policy import PolicyConfig, make_policy
from moe_residency.verify import BitExactVerifier

MODES = ["lru", "slru", "prefetch", "pinned", "pinned_prefetch"]


def build(n=24, per_domain=12, cycles=2, seed=3):
    moe = ReferenceMoE(n_experts=n, d_model=16, hidden=24, top_k=3,
                       seed=1337, router_skew=0.4)
    rng = np.random.default_rng(seed)
    pool = np.arange(n)
    rng.shuffle(pool)
    biases = {}
    for k in range(3):
        b = np.zeros(n, dtype=np.float32)
        b[pool[:4]] += np.float32(1.1)
        b[pool[4 + k * 4: 8 + k * 4]] += np.float32(1.0)
        biases[f"d{k}"] = b
    xs, tags = [], []
    for _ in range(cycles):
        for k in range(3):
            base = rng.normal(0, 1, moe.d).astype(np.float32)
            x = base.copy()
            for _ in range(per_domain):
                x = (np.float32(0.5) * x + np.float32(0.5) * base
                     + np.float32(0.8) * rng.normal(0, 1, moe.d).astype(np.float32))
                xs.append(x.astype(np.float32))
                tags.append(f"d{k}")
    return moe, biases, xs, tags


class TestBitExactAcrossTiers(unittest.TestCase):
    """每种策略、每种层级组合，logits 都必须与「全驻留」逐位相同。"""

    def test_all_modes_bitexact(self):
        moe, biases, xs, tags = build()
        golden = BitExactVerifier()
        golden.start("resident")
        for x, d in zip(xs, tags):
            golden.record("resident", moe.forward(x, domain_bias=biases[d])[0])

        for mode in MODES:
            with self.subTest(mode=mode):
                golden.start(mode)
                store = TieredExpertStore(moe.blocks, warm_cap=8)
                pol = make_policy(moe.E, PolicyConfig(capacity=4, mode=mode,
                                                      pin_frac=0.5, prefetch_q=4))
                from moe_residency.profile import DomainProfile
                prof = DomainProfile(moe.E, ema=0.9)
                prev = None
                for x, d in zip(xs, tags):
                    if d != prev:
                        prev = d
                        if mode in ("prefetch", "pinned_prefetch"):
                            pol.switch_domain(prof.hint(d, q=4))
                            store.load_resident(list(pol.resident()))
                    pol.refresh_pin()
                    out, _ = moe.forward(x, policy=pol, store=store, profile=prof,
                                         domain=d, domain_bias=biases[d])
                    golden.record(mode, out)
                ok, info = golden.compare("resident", mode)
                self.assertTrue(ok, f"{mode} 出现不一致: {info}")


class TestDiskTier(unittest.TestCase):
    def test_disk_roundtrip_is_bitexact(self):
        moe = ReferenceMoE(n_experts=8, d_model=8, hidden=8, top_k=2, seed=1)
        with tempfile.TemporaryDirectory() as d:
            store = TieredExpertStore(moe.blocks, warm_cap=2, disk_dir=d)
            for eid, blk in moe.blocks.items():
                got = store.fetch(eid)
                for k in blk:
                    self.assertTrue(np.array_equal(got[k], blk[k]),
                                    f"专家 {eid} 的 {k} 经磁盘往返后不一致")
            self.assertGreater(store.stats["disk_reads"], 0)

    def test_warm_cap_limits_residency(self):
        moe = ReferenceMoE(n_experts=16, d_model=8, hidden=8, top_k=2, seed=1)
        store = TieredExpertStore(moe.blocks, warm_cap=4)
        self.assertLessEqual(len(store.warm), 4)

    def test_retain_drops_non_resident(self):
        moe = ReferenceMoE(n_experts=8, d_model=8, hidden=8, top_k=2, seed=1)
        store = TieredExpertStore(moe.blocks, warm_cap=8)
        for eid in (0, 1, 2):
            store.put(eid, store.fetch(eid))
        n = store.retain({1})
        self.assertEqual(n, 2)
        self.assertEqual(set(store.fast), {1})


class TestDeterminism(unittest.TestCase):
    def test_two_runs_identical(self):
        moe, biases, xs, tags = build()
        a = [moe.forward(x, domain_bias=biases[d])[0] for x, d in zip(xs, tags)]
        b = [moe.forward(x, domain_bias=biases[d])[0] for x, d in zip(xs, tags)]
        for u, v in zip(a, b):
            self.assertTrue(np.array_equal(u, v))


if __name__ == "__main__":
    unittest.main()
