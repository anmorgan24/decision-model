"""Draw the attention mask for one state and three questions.

    python scripts/plot_attention_mask.py --out attention_mask.png

Rows are the token doing the looking; columns are the tokens it can look at.
The grid comes straight from decision_model.build_attention_mask.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from decision_model import build_attention_mask

SEGMENTS = [("State", 6), ("Question 1", 5), ("Question 2", 5), ("Question 3", 5)]
COLORS = ["#EEF0F3", "#475467", "#2F6FDE", "#E0761A", "#0F8B8D"]  # blocked, state, Q1, Q2, Q3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="attention_mask.png")
    args = parser.parse_args()

    segment_ids = [i for i, (_, n) in enumerate(SEGMENTS) for _ in range(n)]
    allowed = build_attention_mask(segment_ids).numpy()
    # Color each allowed cell by the part of the sequence its row belongs to.
    grid = np.where(allowed, np.array(segment_ids)[:, None] + 1, 0)

    n = len(segment_ids)
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(grid, cmap=ListedColormap(COLORS), vmin=0, vmax=len(COLORS) - 1)

    # Cell borders, drawn directly (tick-based gridlines vanish where they meet a label).
    for edge in np.arange(-0.5, n, 1):
        ax.axhline(edge, color="white", linewidth=1.5)
        ax.axvline(edge, color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)

    starts = np.cumsum([0] + [size for _, size in SEGMENTS])
    centers = [(a + b - 1) / 2 for a, b in zip(starts[:-1], starts[1:])]
    labels = [name for name, _ in SEGMENTS]
    ax.set_xticks(centers, labels, fontsize=10)
    ax.set_yticks(centers, labels, fontsize=10)
    ax.xaxis.tick_top()
    for boundary in starts[1:-1]:
        ax.axhline(boundary - 0.5, color="#98A2B3", linewidth=1)
        ax.axvline(boundary - 0.5, color="#98A2B3", linewidth=1)

    # Each question's answer is read from its last token, the one ending in "Answer:".
    for end in starts[2:]:
        ax.annotate("answer read here", xy=(n - 0.5, end - 1), xytext=(n + 0.4, end - 1),
                    va="center", fontsize=9, color="#475467",
                    arrowprops={"arrowstyle": "-", "color": "#98A2B3", "linewidth": 1})

    ax.set_xlabel("can attend to", fontsize=11, labelpad=10)
    ax.xaxis.set_label_position("top")
    ax.set_ylabel("token", fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    fig.savefig(args.out.rsplit(".", 1)[0] + ".svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
