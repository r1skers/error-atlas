"""Plot verified attribution cell means into a new directory; no numerical rerun."""

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
    summary_path = args.bundle / "summary.json"
    if manifest["status"] != "complete" or manifest["files"]["summary.json"] != sha(summary_path):
        raise ValueError("A complete, intact attribution summary is required")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = summary["cells"]
    spreads = sorted({cell["spread"] for cell in cells})
    arms = sorted({cell["fused"] for cell in cells})
    fig, axes = plt.subplots(len(spreads), len(arms), figsize=(10, 7), squeeze=False,
                             sharex=True, sharey=True, layout="constrained")
    zeros = 0
    for i, spread in enumerate(spreads):
        for j, fused in enumerate(arms):
            axis = axes[i, j]
            for schedule, style in (("chain", "-"), ("balanced", "--")):
                selected = sorted((cell for cell in cells if cell["spread"] == spread
                                   and cell["fused"] == fused and cell["schedule"] == schedule), key=lambda c: c["count"])
                for component, color, marker in (("R", "#b64c26", "o"), ("W", "#346c9b", "s")):
                    values = []
                    for cell in selected:
                        interval = cell["mean_abs_" + component]
                        value = float((Decimal(interval["low"]) + Decimal(interval["high"])) / 2)
                        zeros += value == 0
                        values.append(value if value > 0 else float("nan"))
                    axis.plot([cell["count"] for cell in selected], values, linestyle=style,
                              marker=marker, markersize=4, color=color,
                              label=f"{schedule}: mean |{component}| / l*")
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            counts = sorted({cell["count"] for cell in cells})
            axis.set_xticks(counts, list(map(str, counts)))
            axis.grid(alpha=0.2, which="major")
            axis.set_title(f"spread = {spread} | {'FMA' if fused else 'separate mul/add'}")
            if i == len(spreads) - 1:
                axis.set_xlabel("Block count (n)")
            if j == 0:
                axis.set_ylabel("Mean normalized component magnitude")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Post-hoc attribution | frozen-weight rounding R vs weight discrepancy W\n8 families per cell; logarithmic y-axis; no statistical confidence intervals", fontsize=11)
    args.output.mkdir(parents=True, exist_ok=False)
    image_path = args.output / "components.png"
    fig.savefig(image_path, dpi=180)
    plt.close(fig)
    source = Path(__file__).read_bytes()
    (args.output / "plot_online_attribution.py.txt").write_bytes(source)
    metadata = {"input_summary_sha256": sha(summary_path), "script_sha256": hashlib.sha256(source).hexdigest(),
                "png_sha256": sha(image_path), "matplotlib": matplotlib.__version__,
                "display": "Midpoints of numerical enclosures; zero means omitted on logarithmic y-axis.",
                "zero_means_omitted": zeros}
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(str(image_path))


if __name__ == "__main__":
    main()
