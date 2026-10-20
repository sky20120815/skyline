#!/usr/bin/env python
"""理论速度矩阵 —— **主目标：8GB 显存 + 8GB 内存**，其余档位仅作对照。

⚠️ **这是理论估算，不是实测。** 公式与假设全部写在下面；实测对照见 docs/HARDWARE_MATRIX.md。

## 计算模型

1. **可用快速内存（池预算）**
   - 显存扣 ~1GB 给显示/帧缓冲
   - 内存扣 **3GB 给操作系统与运行时**（8GB 内存的系统，OS 本身就吃掉一大块）
   - `池预算 = 可用显存 + 可用内存`；`池 = 总参 × 每参数字节`

2. **池装得下** ⇒ `tok/s = min(跨层带宽 ÷ 激活字节, CPU 算力上限)`
   跨层带宽按显存/内存的**容量占比做调和平均**（不是算术平均）

3. **池装不下** ⇒ `tok/s = SSD带宽 ÷ (激活字节 × miss率)`，`miss率 = 1 − 池预算/池`
   ⇒ **断崖**

## 参数与来源
- `B_ssd` = **2.6 GB/s**（本机实测：YMTC NVMe，`dd bs=1M count=1024`）
- `B_ram` = DDR5-4800 单通道 38.4 / 双通道 76.8 GB/s（**理论峰值，未实测**）
- `B_vram` = **保守假设 200 GB/s**（8GB 显存笔记本档）；桌面 32GB 卡用 1792 GB/s（公开规格）
- `F_cpu` = **假设 400 GOPS**（int4，i7-13650HX **无 AVX-512**，此值偏乐观）

跑法：python benchmarks/theoretical_speed.py
"""
from __future__ import annotations

# (名称, 总参B, 激活参B, 备注)
MODELS = [
    ("Qwen3.6-35B-A3B",      35,   3, "最小的一档；llama.cpp / FreeToken 均有实测"),
    ("Qwen3.8-Flash-Next",  125,  10, "Strata / FreeToken 有实测"),
    ("DeepSeek-V4-Flash",   284,  13, "FreeToken 实测 22 tok/s（池在 RAM 的工作站）"),
    ("DeepSeek-V4.1-Flash", 552,  26, "colibri v1.11 原生读取"),
    ("GLM-5.2",             744,  40, "colibri / FreeToken 均有实测"),
    ("DeepSeek-V4-Pro",    1600,  49, "当前无消费级实测"),
]

# (名称, 内存GB, 显存GB, B_ram, B_vram, B_ssd, 是否主目标, 备注)
HARDWARE = [
    ("8GB显存 + 8GB内存",      8.0,   8.0, 38.4,  200.0, 2.6, True,
     "目标配置：内存扣 3GB 给 OS，显存扣 1GB 给显示"),
    ("8GB显存 + 16GB内存",    16.0,   8.0, 38.4,  200.0, 2.6, False, "小幅加内存"),
    ("8GB显存 + 32GB内存",    32.0,   8.0, 76.8,  200.0, 2.6, False, "双通道"),
    ("8GB显存 + 64GB内存",    64.0,   8.0, 76.8,  200.0, 2.6, False, "双通道"),
    ("RTX 5090 32GB + 128GB", 128.0, 32.0, 76.8, 1792.0, 2.6, False, "工作站档（对照）"),
    ("6×RTX 5090 (192GB)",     64.0, 192.0, 76.8, 1792.0, 2.6, False, "多卡档（对照）"),
]

BITS = [("4-bit", 0.5), ("3-bit", 0.375), ("2-bit", 0.25), ("1.58-bit", 0.1975)]
F_CPU_GOPS = 400.0
RAM_RESERVE = 3.0     # 操作系统 + 运行时
VRAM_RESERVE = 1.0    # 显示 / 帧缓冲


def budget_of(ram_gb: float, vram_gb: float) -> float:
    """可用来放专家池的快速内存（GB）。"""
    return max(ram_gb - RAM_RESERVE, 0.0) + max(vram_gb - VRAM_RESERVE, 0.0)


def estimate(total_b: float, active_b: float, ram_gb: float, vram_gb: float,
             b_ram: float, b_vram: float, b_ssd: float, bits: float):
    pool_gb = total_b * bits
    active_gb = active_b * bits
    budget = budget_of(ram_gb, vram_gb)
    compute_cap = F_CPU_GOPS * 1e9 / (2 * active_b * 1e9)            # tok/s（纯 CPU）
    if pool_gb <= budget:
        ram_use = max(ram_gb - RAM_RESERVE, 0.0)
        vram_use = max(vram_gb - VRAM_RESERVE, 0.0)
        tot = max(ram_use + vram_use, 1e-9)
        eff_bw = 1.0 / (ram_use / tot / b_ram + vram_use / tot / b_vram)
        bw_cap = eff_bw / active_gb
        gpu_cap = b_vram / active_gb
        speed = max(bw_cap, min(gpu_cap, compute_cap * 3))
        speed = min(speed, max(bw_cap, compute_cap))
        return speed, pool_gb, f"池进内存（{pool_gb:.0f}G ≤ {budget:.0f}G）"
    miss = 1.0 - budget / pool_gb
    return b_ssd / (active_gb * miss), pool_gb, f"磁盘流式（miss≈{miss*100:.0f}%）"


def max_runnable(bits: float):
    rows = []
    for hw, ram, vram, _br, _bv, _bs, primary, _note in HARDWARE:
        budget = budget_of(ram, vram)
        best = ("（无：清单里没有模型装得下）", 0.0, 0.0)
        for name, total, active, _s in MODELS:
            if total * bits <= budget and total > best[1]:
                best = (name, total, active)
        rows.append((hw, budget, best, primary))
    return rows


def main() -> int:
    print("=" * 106)
    print("理论速度矩阵（估算，非实测）    目标配置：8GB 显存 + 8GB 内存")
    print(f"  预算扣减：内存 -{RAM_RESERVE}GB(OS/运行时)  显存 -{VRAM_RESERVE}GB(显示)")
    print("  参数：SSD 2.6GB/s(实测) | RAM 38.4/76.8(理论) | VRAM 200(保守假设)/1792(规格) | CPU 400 GOPS(假设)")
    print("=" * 106)

    # ---------------------------------------------- 主目标详表
    for bits_name, bits in BITS:
        print(f"\n{'-'*106}\n【目标 8GB 显存 + 8GB 内存】量化 {bits_name}（{bits} B/参数）\n{'-'*106}")
        print(f"  {'模型':<22}{'池大小':>9}{'激活参数':>10}{'激活字节':>10}{'理论tok/s':>11}  状态")
        for name, total, active, _src in MODELS:
            sp, pool, how = estimate(total, active, 8.0, 8.0, 38.4, 200.0, 2.6, bits)
            flag = "  ✅ 可对话" if sp >= 10 else ("  ⚠ 偏慢" if sp >= 3 else "  ⚠ 爬行(仅批量)")
            print(f"  {name:<22}{pool:>8.1f}G{active:>9.0f}B{active*bits:>9.2f}G{sp:>11.2f}  {how}{flag}")

    # ---------------------------------------------- 各档位最大可跑
    print(f"\n{'-'*106}\n各档位「最大可跑模型」（要求专家池能进内存，否则只是爬行）\n{'-'*106}")
    for bits_name, bits in BITS:
        print(f"\n量化 {bits_name}：")
        print(f"   {'硬件':<26}{'池预算':>8}  可跑的最大模型（总参 / 激活）")
        for hw, budget, (nm, tot, act), primary in max_runnable(bits):
            mark = "* " if primary else "  "
            tail = f"（{tot:.0f}B / {act:.0f}B）" if tot else ""
            print(f" {mark}{hw:<25}{budget:>7.0f}G  {nm}{tail}")

    # ---------------------------------------------- 结论
    b88 = budget_of(8.0, 8.0)
    print(f"\n{'-'*106}\n结论（目标 8GB 显存 + 8GB 内存，池预算 = {b88:.0f}GB）\n{'-'*106}")
    for bits_name, bits in BITS:
        print(f"   {bits_name:<9} ⇒ 总参上限 ≈ {b88/bits:>5.0f}B")
    print(f"""
   · **8+8 的天花板**：4-bit 约 {b88/0.5:.0f}B ｜ 3-bit 约 {b88/0.375:.0f}B ｜ 2-bit 约 {b88/0.25:.0f}B ｜ 1.58-bit 约 {b88/0.1975:.0f}B
   · **35B-A3B 在 8+8 上需要 2~3-bit**；4-bit 只够 ~{b88/0.5:.0f}B 的模型
   · **V4-Flash(284B) 在 8+8 上不可能"可对话"**：即使 1.58-bit，池 56GB 对 {b88:.0f}GB 预算
     ⇒ 约 {(1-b88/56)*100:.0f}% 的专家要流式读盘 ⇒ 模型估算 ≈1.3 tok/s；即便按 colibri 在
     全盘流场景实测到的 5~10× 修正，也只是"可能"到个位数，且 **1.58-bit 质量已明显退化**
   · **要真跑 284B 且可对话**：内存加到 ~150GB（或 32GB 显存工作站 + 128GB 内存）

   校准：本模型对**纯磁盘流式**偏保守 —— colibri 在 25GB 笔电跑 GLM-5.2 744B（池 372GB、
   几乎全盘流）实测 0.05~0.1 tok/s，本模型给 ~0.015，低估 5~10 倍（大 slab 顺序读 +
   页缓存 + I/O 流水线未建模）。⇒ **磁盘流式档以实测为准，本表价值在断崖位置与量级。**""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
