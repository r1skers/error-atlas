"""Regenerate the coherence-dominance figure from the exact oracle.

Reproducible figure generator (not a research runner; writes only the SVG under
docs/figures/). For each controlled wide-range input, it decomposes E^2 = A + C over
32 candidate trees and plots the tree-to-tree spread of the local-energy term A against
the sign-coherence term C. The point of the figure is that C, not A, drives the spread.

Run from the repository root:

    python tools/make_coherence_figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import pstdev

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "topics/softmax/experiments"))

from predictor_calibration_inputs import wide_range_random  # noqa: E402
from predictor_tree_generator import (  # noqa: E402
    random_contiguous_split_graph,
    random_pair_merge_graph,
)
from reduction_analysis import CoherenceAnalysis, replay  # noqa: E402

INPUT_SEEDS = (1, 2, 3, 22260821)
WIDTH = 256
TREES = 32
OUT = ROOT / "docs/figures/coherence_dominance.svg"


def measure() -> list[dict]:
    rows = []
    for seed in INPUT_SEEDS:
        generated = wide_range_random(WIDTH, seed=seed)
        a_vals, c_vals = [], []
        for k in range(TREES):
            builder = random_contiguous_split_graph if k % 2 else random_pair_merge_graph
            graph = builder(WIDTH, seed=1000 + k)
            ac = CoherenceAnalysis(replay(generated.values, graph)).ac
            a_vals.append(float(ac.a_local))
            c_vals.append(float(ac.c_coherence))
        std_a, std_c = pstdev(a_vals), pstdev(c_vals)
        rows.append({"seed": seed, "std_a": std_a, "std_c": std_c, "ratio": std_c / std_a})
    return rows


def render(rows: list[dict]) -> str:
    W, H = 720, 420
    pad_l, pad_r, pad_t, pad_b = 70, 24, 70, 70
    plot_w = W - pad_l - pad_r
    plot_h = H - pad_t - pad_b
    scale = 1e10  # display units: 1e-10
    ymax = max(max(r["std_a"], r["std_c"]) for r in rows) * scale * 1.18
    n = len(rows)
    group_w = plot_w / n
    bar_w = group_w * 0.30

    def y(v):  # value in 1e-10 units -> pixel
        return pad_t + plot_h - (v / ymax) * plot_h

    A_COLOR, C_COLOR, INK, MUTE = "#7c8aa5", "#e06c5a", "#1b2430", "#6b7684"
    p = []
    p.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'font-family="ui-sans-serif,system-ui,Segoe UI,Roboto,Helvetica,Arial" '
        f'font-size="13">'
    )
    p.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
    p.append(
        f'<text x="{pad_l}" y="30" font-size="17" font-weight="700" fill="{INK}">'
        f"Coherence dominates tree-to-tree error variation</text>"
    )
    p.append(
        f'<text x="{pad_l}" y="50" fill="{MUTE}">'
        f"Std. dev. across {TREES} trees per input, width {WIDTH}. "
        f"E² = A (local energy) + C (sign coherence).</text>"
    )
    # y grid + ticks
    ticks = 4
    for i in range(ticks + 1):
        v = ymax * i / ticks
        yy = y(v)
        p.append(
            f'<line x1="{pad_l}" y1="{yy:.1f}" x2="{W-pad_r}" y2="{yy:.1f}" '
            f'stroke="#e7eaee" stroke-width="1"/>'
        )
        p.append(
            f'<text x="{pad_l-8}" y="{yy+4:.1f}" text-anchor="end" fill="{MUTE}" '
            f'font-size="11">{v:.1f}</text>'
        )
    p.append(
        f'<text x="18" y="{pad_t+plot_h/2:.0f}" fill="{MUTE}" font-size="11" '
        f'transform="rotate(-90 18 {pad_t+plot_h/2:.0f})" text-anchor="middle">'
        f"std. dev.  (×10⁻¹⁰)</text>"
    )
    for gi, r in enumerate(rows):
        cx = pad_l + group_w * (gi + 0.5)
        a = r["std_a"] * scale
        c = r["std_c"] * scale
        ax = cx - bar_w - 3
        cxx = cx + 3
        p.append(
            f'<rect x="{ax:.1f}" y="{y(a):.1f}" width="{bar_w:.1f}" '
            f'height="{pad_t+plot_h-y(a):.1f}" fill="{A_COLOR}" rx="2"/>'
        )
        p.append(
            f'<rect x="{cxx:.1f}" y="{y(c):.1f}" width="{bar_w:.1f}" '
            f'height="{pad_t+plot_h-y(c):.1f}" fill="{C_COLOR}" rx="2"/>'
        )
        p.append(
            f'<text x="{cx:.1f}" y="{y(c)-8:.1f}" text-anchor="middle" '
            f'fill="{INK}" font-size="12" font-weight="700">{r["ratio"]:.1f}×</text>'
        )
        label = "seed " + (str(r["seed"]) if r["seed"] < 1000 else "22260821")
        p.append(
            f'<text x="{cx:.1f}" y="{pad_t+plot_h+20:.1f}" text-anchor="middle" '
            f'fill="{MUTE}" font-size="11">{label}</text>'
        )
    # legend
    lx, ly = W - pad_r - 210, pad_t + 6
    p.append(f'<rect x="{lx}" y="{ly}" width="12" height="12" fill="{A_COLOR}" rx="2"/>')
    p.append(f'<text x="{lx+18}" y="{ly+11}" fill="{INK}">std(A) local energy</text>')
    p.append(f'<rect x="{lx}" y="{ly+20}" width="12" height="12" fill="{C_COLOR}" rx="2"/>')
    p.append(f'<text x="{lx+18}" y="{ly+31}" fill="{INK}">std(C) sign coherence</text>')
    p.append(
        f'<text x="{pad_l}" y="{H-18}" fill="{MUTE}" font-size="11">'
        f"The number above each pair is std(C)/std(A): C varies 2.5–3.6× more than A, "
        f"so it decides which tree is best.</text>"
    )
    p.append("</svg>")
    return "".join(p)


def main() -> int:
    rows = measure()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render(rows), encoding="utf-8")
    for r in rows:
        print(f"seed={r['seed']:>9} std_a={r['std_a']:.3e} std_c={r['std_c']:.3e} ratio={r['ratio']:.2f}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
