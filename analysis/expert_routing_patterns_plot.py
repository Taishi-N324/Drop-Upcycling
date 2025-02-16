import os
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_and_process_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    processed_data = defaultdict(lambda: defaultdict(float))
    for layer, expert_data in data.items():
        total = sum(sum(token_counts.values()) for token_counts in expert_data.values())
        for expert, token_counts in expert_data.items():
            processed_data[layer][expert] = sum(token_counts.values()) / total

    return processed_data


def create_plot():
    base_path = "routing_output"
    models = [
        "llm-jp/FS-8x1.5B",
        "llm-jp/NU-8x1.5B",
        "llm-jp/DU-0.5-8x1.5B",
    ]

    model_labels = [
        "8×1.5B From Scratch",
        "8×1.5B naïve Upcycling",
        r"8×1.5B Drop-Upcycling ($r = 0.5$)",
    ]

    datasets = ["c4", "en_wikipedia", "ja_mc4", "ja_wikipedia", "code_stack"]
    dataset_labels = [
        "C4",
        "English Wikipedia",
        "Japanese MC4",
        "Japanese Wikipedia",
        "The Stack",
    ]

    layers_to_plot = [0, 8, 16, 23]

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    sns.set_style("whitegrid")
    plt.rcParams["axes.linewidth"] = 1.5

    FONTSIZE = 32
    colors = sns.color_palette("colorblind", n_colors=len(datasets))

    fig, axes = plt.subplots(
        len(layers_to_plot), len(models), figsize=(28, 14), sharex="none", sharey=False
    )

    bar_width = 0.15
    index = np.arange(8)

    for model_idx, (model, model_label) in enumerate(zip(models, model_labels)):
        for layer_idx, layer in enumerate(layers_to_plot):
            ax = axes[layer_idx, model_idx]
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])

            for dataset_idx, (dataset, color) in enumerate(zip(datasets, colors)):
                file_path = os.path.join(
                    base_path,
                    model,
                    "eid2token",
                    f"{dataset.lower().replace(' ', '_')}.pkl",
                )

                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    continue

                data = load_and_process_data(file_path)

                if layer not in data:
                    print(f"Layer {layer} not found in data for {dataset} in {model}")
                    continue

                expert_proportions = [data[layer][expert] for expert in range(8)]
                ax.bar(
                    index + dataset_idx * bar_width,
                    expert_proportions,
                    bar_width,
                    label=dataset_labels[dataset_idx]
                    if layer_idx == 0 and model_idx == 0
                    else "",
                    color=color,
                    alpha=0.7,
                    edgecolor="white",
                    linewidth=0.5,
                )

            if layer_idx == 0:
                ax.set_title(f"{model_label}", fontsize=FONTSIZE - 2, fontweight="bold")
            if model_idx == 0:
                ax.set_ylabel(f"Layer {layer}", fontsize=FONTSIZE - 4)

            ax.set_ylim(0, 1)
            ax.set_xlim(-0.5, 8.0)

            ax.set_xticks(index + bar_width * (len(datasets) - 1) / 2)
            if (
                layer_idx == len(layers_to_plot) - 1
            ):  # Only show x-axis labels for the bottom row
                ax.set_xticklabels(range(1, 9), fontsize=FONTSIZE - 8)
            else:
                ax.set_xticklabels([])  # Remove x-axis labels for other rows
            ax.tick_params(axis="both", which="major", labelsize=FONTSIZE - 8)

            ax.axhline(y=0.125, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, linestyle="--", alpha=0.3)

            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_yticklabels(
                ["0", "0.25", "0.5", "0.75", "1"] if model_idx == 0 else []
            )
            if (
                layer_idx == len(layers_to_plot) - 1
            ):  # Only add x-axis label for the bottom row
                ax.set_xlabel("Expert ID", fontsize=FONTSIZE - 6)
            else:
                ax.set_xlabel("")  # Remove x-axis label for other rows

    fig.text(
        0.01,
        0.5,
        "Routing Probability",
        va="center",
        rotation="vertical",
        fontsize=FONTSIZE,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        fontsize=FONTSIZE - 4,
        frameon=True,
        edgecolor="lightgray",
        fancybox=True,
        shadow=False,
        borderpad=0.5,
        labelspacing=0.5,
        handletextpad=0.5,
        markerscale=1,
    )

    plt.tight_layout()
    plt.subplots_adjust(
        top=0.95, bottom=0.18, left=0.08, right=0.98, hspace=0.2, wspace=0.2
    )
    plt.savefig(
        "Figure5.pdf", format="pdf", dpi=300, bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


def main():
    create_plot()


if __name__ == "__main__":
    main()
