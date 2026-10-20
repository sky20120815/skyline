# moe-residency

![CI](https://github.com/sky20120815/moe-residency/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-numpy%20only-blue.svg)

**Bit-exact expert residency for MoE offloading — tiering may change speed, never semantics.**

An engine-agnostic residency-policy layer and a bit-exact verifier for Mixture-of-Experts
offloading across VRAM / RAM / SSD.

It gives an existing engine two things:

1. **A reusable bit-exactness check** — prove that moving experts to a slower tier does not
   change a single bit of the output.
2. **A pluggable residency policy** — LRU, SLRU, domain-profile prefetch, frequency pinning.
   A host integrates it in three call sites.

> **This is not an inference engine.** There is no tokenizer, no attention kernel, no KV
> management, no dequantization kernel here, and it cannot load DeepSeek / Qwen / GLM directly.
> To run a real model, use colibri, FreeToken, Strata, KTransformers or llama.cpp — see
> [docs/INTEGRATION.md](docs/INTEGRATION.md).

---

## Table of contents

- [Non-goals](#non-goals)
- [Install](#install)
- [Quickstart](#quickstart)
- [What the bit-exact verifier checks](#what-the-bit-exact-verifier-checks)
- [Residency policies](#residency-policies)
- [Using it as a plugin](#using-it-as-a-plugin)
- [Benchmarks](#benchmarks)
- [Hardware matrix](#hardware-matrix)
- [Limitations](#limitations)
- [Why it is built this way](#why-it-is-built-this-way)
- [Development and tests](#development-and-tests)
- [Acknowledgements](#acknowledgements)
- [Roadmap](#roadmap)
- [License](#license)

---

## Non-goals

Please read this section first — it is short on purpose.

- **It does not load any real model.** No tokenizer, no attention kernel, no KV management, no
  dequantization.
- **It will not run DeepSeek-V4-Flash, Qwen or GLM**, and it publishes **no tokens/second numbers
  for real models**.
- The only "engine" in this repository is a **64-expert × 32-dim toy model** in `engine.py`, used
  to exercise the mechanisms. Its "fast/slow" behaviour is derived from a **simulated I/O count**,
  not from measured decode throughput.
- **It does not make anything faster.** `prefetch` and `pin` change *which misses have to wait*;
  they do not change the underlying bandwidth. DeepSeek-V4-Flash (284B / 13B active) on
  8 GB VRAM + 8 GB RAM runs at roughly **1 tok/s** — the bottleneck is SSD and RAM capacity, and
  nothing in this repository changes that.

If you want real throughput, use one of the engines listed above. This repository answers two
questions only: *does placement change the output?* and *which experts should be cached?*

---

## Install

Requires Python 3.10+ and NumPy. That is the whole dependency list.

```bash
git clone https://github.com/sky20120815/moe-residency.git
cd moe-residency
pip install -e .
```

Optional, for the test suite:

```bash
pip install -e ".[dev]"
```

---

## Quickstart

```bash
python examples/demo_engine.py     # end-to-end: 3-tier vs fully-resident + 5 policies
python examples/plugin_demo.py     # same thing driven through the plugin interface
python benchmarks/policy_bench.py  # policy layer only: A/B on skewed routing
```

Actual output of `examples/demo_engine.py` (reproducible):

```
expert pool 64 | fast tier 16 | RAM 20 | tokens 540 | 6 domains × 3 rounds (30 tokens each)
workload: 6 global hot experts + 8 domain-private experts

policy                   stall misses    stall rate    total I/O    prefetch I/O     used     bit-exact
lru                     123       5.7%      84        0        0         yes
slru                    140       6.5%     100        0        0         yes
prefetch                 93       4.3%      91       26       36         yes
pinned                  135       6.2%      93        0        0         yes
pinned_prefetch          99       4.6%      96       27       37         yes

stall-miss change vs LRU: slru +13.8%  prefetch -24.4%  pinned +9.8%  pinned_prefetch -19.5%
All five policies produce logits bit-identical to the fully-resident path.
```

---

## What the bit-exact verifier checks

Tiering is supposed to be a pure performance decision. `BitExactVerifier` makes that claim
testable: it hashes the float32 logits of each forward pass with SHA-256 and compares the two runs
digit by digit.

```python
from moe_residency.verify import BitExactVerifier

verifier = BitExactVerifier()

verifier.start("fully_resident")
verifier.record(logits_if_everything_fits_in_vram)

verifier.start("tiered")
verifier.record(logits_with_ram_and_disk_tiers)

verifier.assert_bitexact("fully_resident", "tiered")   # raises on any bit of difference
```

Measured result from `examples/demo_engine.py`:

| path                  | logits digest          | verdict            |
| --------------------- | ---------------------- | ------------------ |
| fully resident        | `b2cd912099de821d …`   | —                  |
| 3-tier (real disk I/O) | `b2cd912099de821d …`   | bit-identical      |

The verifier is a **library, not a test script** — any new policy, cache or compression scheme can
be checked with three lines. This is the piece the ecosystem usually lacks: colibri states the same
principle in its README and verifies it with token-exact diffing, but if *you* add a cache, a
prefetcher or a quantizer to *your* engine, this class is how you prove you did not change the
output.

---

## Residency policies

| mode              | mechanism                                    | good fit                        |
| ----------------- | -------------------------------------------- | ------------------------------- |
| `lru`             | least recently used (baseline)               | general purpose                 |
| `slru`            | segmented LRU (probation + protected)        | working set slightly over capacity, bursty |
| `prefetch`        | LRU + **prefetch the domain's historical hot set on task switch** | **agentic / multi-task workloads** |
| `pinned`          | frequency EMA, top-P **never evicted**       | **globally** hot experts        |
| `pinned_prefetch` | both of the above                            | global hot set + task switching |

### Metrics: why "total I/O" is the wrong yardstick

Prefetching **increases** total reads (it moves them earlier) but **reduces stalls** (they overlap
with compute). Scoring prefetch by read count therefore always reaches the wrong conclusion. This
repository reports two metrics separately:

- **stall miss** — a miss that must be waited on. This is what the user feels.
- **prefetch I/O** — bytes moved ahead of time. Not a penalty, but it belongs in the bandwidth budget.

---

## Using it as a plugin

The package publishes a standard Python entry point in the group `moe_residency.plugins`, so a host
discovers it without knowing this package's path.

```python
from importlib.metadata import entry_points

ep = next(e for e in entry_points(group="moe_residency.plugins") if e.name == "moe-residency")
plugin = ep.load()(n_experts=19456, capacity=64, mode="prefetch")

for token in stream:
    plugin.before_route(domain=detect_task(token))    # 1) before routing -> experts to prefetch
    ids = my_router.topk(token)                       #    host's own routing
    for eid, missed in plugin.on_route(ids).items():   # 2) after routing -> True means "fetch it"
        if missed:
            my_engine.load_expert(eid)                #    host moves the bytes (may be async)
    out = my_engine.compute(token)
    plugin.after_token()                              # 3) end of token -> reconcile residency
```

| call                     | when                       | plugin does                                              | host does                    |
| ------------------------ | -------------------------- | -------------------------------------------------------- | ---------------------------- |
| `before_route(domain)`   | on task/domain switch, **before routing** | read the domain profile, mark the hot set resident | prefetch those experts (async) |
| `on_route(ids)`          | after routing              | account hits/misses, update domain profile and frequency EMA | move the missed experts |
| `after_token()`          | end of each token          | periodically refresh pinned hot set, persist the profile | —                           |

Optional but recommended: pass a `BitExactVerifier` into the plugin, call
`plugin.record(logits)` on every forward pass, and run a fully-resident control process so you can
call `plugin.assert_bitexact("run_a", "run_b")`.

### Engine adapters

| adapter                     | works today                                                                                                         | needs an upstream change                            |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `adapters/colibri.py`       | **environment-variable recipes** (`PIN` / `PIN_GB` / `PREFETCH` / `PILOT_REAL` / `PILOT_EVICT_GUARD` / `PIPE`, all documented upstream) + half-life conversion for `COLI_USAGE_DECAY` | our addition is **domain-conditioned** hot sets; the `.coli_usage` format is not published, so we do not guess it |
| `adapters/llama_cpp.py`     | exports a **hot-set manifest** (our own format) + the currently available `--n-cpu-moe` command line                    | "pin by heat into the GPU buffer" requires touching the expert placement path (insertion points are listed) |
| `adapters/freetoken.py`     | explicit mount point: **semantic anchor / task boundary** (KV is already cached there, experts are not prefetched)      | per-layer granularity needs upstream work; do **not** touch q\* (orthogonal) |

---

## Benchmarks

### Result 1 — bit-exactness: stable pass

See [the table above](#what-the-bit-exact-verifier-checks). All five policies are verified
independently against the fully-resident path on every demo run.

### Result 2 — policy gains are strongly workload-dependent

| workload                                  | observation                                                                                     |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------- |
| toy engine, 6 domains, fast tier 16      | **pure domain-profile prefetch: −24.4 % stall misses**; `pin` alone +9.8 % (worse); SLRU +13.8 % (worse) |
| synthetic skewed routing, 256 experts, capacity 12→64 | **SLRU is a stable −17 %**; prefetch ≈ 0 (misses come from the long tail, not from switching); pin −3 % |
| single-domain steady load                 | LRU is already near-optimal, all policies within 4 %                                           |

No single policy wins everywhere. **Measure your workload, then pick a policy** — which is exactly
what `benchmarks/policy_bench.py` is for.

### Result 3 — placement policy does not buy bandwidth

`prefetch` and `pin` can shave the *peak* number of stalls (roughly 20 % in the demo), but they
cannot turn 0.5 tok/s into 5 tok/s. The gap is SSD bandwidth and pool capacity, not policy.

Raw numbers: [benchmarks/RESULTS.md](benchmarks/RESULTS.md).

---

## Hardware matrix

**Q: can this run DeepSeek-V4-Flash (284B / 13B active) at 5–10 tok/s on 8 GB VRAM + 8 GB RAM?**

**No — and it is off by more than an order of magnitude.** This repository does not load real
models at all (see [Non-goals](#non-goals)).

For reference, the arithmetic for V4-Flash. Pool budget = VRAM − 1 GB (display) + RAM − 3 GB (OS):

| configuration           | pool budget | expert pool residency             | disk read per token | speed                                                       |
| ----------------------- | ----------- | --------------------------------- | ------------------- | ----------------------------------------------------------- |
| **8 GB VRAM + 8 GB RAM** | **12 GB**   | pool 56 GB (@1.58-bit) → 21 % cached | ~2.0 GB (79 % miss) | **≈1.3 tok/s** (order of magnitude, not streaming)           |
| 8 GB VRAM + 16 GB RAM    | 20 GB       | 36 % cached                        | ~1.6 GB             | ≈1.6 tok/s                                                  |
| 8 GB VRAM + 64 GB RAM    | 68 GB       | entire pool in RAM                 | 0                   | **5–10 tok/s** (bandwidth ceiling ~30; compute binds first) |
| 32 GB VRAM + 128 GB RAM  | 156 GB      | fully resident                     | 0                   | **22 tok/s** (measured by FreeToken)                        |

**Model ceiling on an 8 + 8 machine** (requires the pool to fit in RAM):

| quantization | total-parameter ceiling | largest model that fits                          |
| ------------ | ----------------------- | ------------------------------------------------ |
| 4-bit        | ≈24B                    | nothing on the list fits (35B-A3B at 4-bit is 17.5 GB) |
| 3-bit        | ≈32B                    | 35B-A3B at 3-bit (theoretical ~27 tok/s, optimistic) |
| 2-bit        | ≈48B                    | **Qwen3.6-35B-A3B** (theoretical ~97 tok/s, clearly optimistic) |
| 1.58-bit     | ≈61B                    | 35B-A3B (theoretical ~123 tok/s)                 |

> All published 35B-A3B throughput numbers (llama.cpp 34.9, FreeToken 39.3) were measured with
> **≥32 GB of RAM**. There is **no public measurement on an 8 GB RAM machine**; the 35B row above is
> an optimistic theoretical figure.

Two conclusions:

1. The 5–10 tok/s threshold requires **"the pool fits in RAM" ≈ 64 GB of RAM**, not a better policy.
2. With 8 GB of RAM even the dense part plus a handful of experts is tight, most tokens stream from
   SSD, and the physical ceiling on consumer NVMe is ~1 tok/s.

Full matrix (every hardware class × every model, theoretical speed, measured cross-checks,
calibration and recommendations): [docs/HARDWARE_MATRIX.md](docs/HARDWARE_MATRIX.md). Generated by
`benchmarks/theoretical_speed.py` — edit the model list or the hardware and re-run.

---

## Limitations

- **Not an engine.** The toy engine is NumPy, 64 experts of 32 dimensions, built to exercise the
  mechanisms. Its performance numbers are not representative.
- **Single run, synthetic workload.** Numbers come from fixed seeds and hand-written workloads; a
  real model's access distribution is different.
- **No quantization.** Ternary/int4 introduces its own error. This repository verifies that
  *placement* does not change semantics — not that *compression* does not.
- **No concurrency.** Single stream; continuous batching and KV reuse are out of scope.
- **`pin` and SLRU can be worse than LRU** on some workloads. That is a real result, see
  [Benchmarks](#benchmarks).

---

## Why it is built this way

1. **The residency set is decided by the policy alone**; the store only materializes and releases
   (`store.retain(policy.resident())`). Two independent LRUs is the most common source of bugs in
   this class of system.
2. **Prefetch must be reflected in the residency set**, otherwise the next `retain()` evicts it and
   the read was wasted.
3. **The policy runs before routing** (`switch_domain()` → `access()`), not as an after-the-fact fix.
4. **Verification is a library, not a script** — see [the verifier section](#what-the-bit-exact-verifier-checks).

Details: [docs/DESIGN.md](docs/DESIGN.md).

### A real bug the tests caught

`switch_domain` originally decided "is this expert resident?" *during* insertion, so it also
reported experts that LRU immediately evicted again. The host would dutifully move them to the fast
tier, and the next `retain()` would drop them — one wasted I/O. It violated this repository's own
design rule ("prefetch must be reflected in the residency set", `docs/DESIGN.md` §4.2). It is now
reconciled once after the loop, returning only experts that are actually resident.

Neither demo exposed it, because both use `prefetch_q ≤ capacity`. This is the value of writing the
claim down as a test.

---

## Development and tests

```bash
python run_tests.py -v      # 53 test cases, ~0.2 s, no pytest required
pytest tests/               # works too
```

Coverage: residency/eviction/prefetch/pin semantics, domain-profile EMA and persistence round-trip,
positive and negative digest comparison, the three engine adapters, the plugin's three-call
lifecycle, and **`tests/test_engine.py::test_all_modes_bitexact`** — all five policies across all
three tiers must match the fully-resident path bit for bit.

CI runs the tests plus both smoke demos on Python 3.10–3.13, and checks that
`moe_residency.plugin.json` stays consistent with the code.

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## Acknowledgements

No code was copied from any of the projects below; the implementation is written from scratch. These
projects provided the mechanisms and the prior art:

| project | license | what was taken conceptually |
|---|---|---|
| [colibri](https://github.com/JustVugg/colibri) | Apache-2.0 | three-tier placement (VRAM/RAM/NVMe), **learned pinned hot store**, one-layer-ahead prefetch, and the hard "placement changes speed, not semantics" promise with token-exact verification |
| [FreeToken](https://github.com/FlashML-org/FreeToken) (UC Berkeley / MIT) | Apache-2.0 | elastic GPU expert cache, q\* bandwidth-adaptive scheduling, full-layer double buffering, semantic anchors |
| [Strata](https://github.com/Niko1221/Strata) | MIT | GPU for dense + hot experts, CPU computes cold experts, whole pool resident in RAM |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) (Tsinghua MADSys, SOSP'25) | Apache-2.0 | CPU/GPU heterogeneous cooperation, CPU-side expert compute |
| [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) | — | hot/cold separation granularity (it goes to neuron level; this repository stops at expert level) |
| [MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity) | — | **activation-aware expert prefetching** — the direct source of `profile.py` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | MIT | two-tier expert cache proposal (issue #20757), SLRU segmentation; the `--n-cpu-moe` offload form |
| [vLLM](https://github.com/vllm-project/vllm) | Apache-2.0 | the GPU-side cache idea in PR #37190 |
| Bonsai 2 / BitNet | — | ternary quantization (not implemented, referenced in design discussion only) |

Full detail: [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md).

---

## Roadmap

- [ ] colibri integration: a `prefetch` hook on its per-layer LRU, with `verify.py` as the regression gate
- [ ] llama.cpp integration: `--hot-expert-pin` driven by the use counters, measured tok/s under `--n-cpu-moe`
- [ ] Prefetch waste metric (prefetched but unused / evicted early)
- [ ] Bit-exactness under compression (error bounds for int8/int4 instead of exact equality)

## License

MIT — see [LICENSE](LICENSE).
