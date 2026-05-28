"""
Figure Generation Script
========================
Generates report-ready figures from experiment result files.

Outputs are saved under:
- figures/

Generated figures:
1. Model F1 comparison
2. Runtime training time comparison
3. Automata parameter sensitivity, F1
4. Automata parameter sensitivity, state count
5. Automata parameter sensitivity, transition density
6. Transition probability heatmap from explainability JSON
7. Automata state diagram from explainability JSON
"""

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


RESULTS_DIR = "results"
FIGURES_DIR = "figures"

os.makedirs(FIGURES_DIR, exist_ok=True)


def save_current_figure(filename: str) -> None:
    """Saves the active matplotlib figure with a clean layout."""
    path = os.path.join(FIGURES_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def load_csv(filename: str) -> pd.DataFrame:
    """Loads a CSV file from the results directory."""
    path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

    return pd.read_csv(path)


def load_json(filename: str) -> Dict:
    """Loads a JSON file from the results directory."""
    path = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def plot_model_f1_comparison() -> None:
    """
    Plots F1 comparison for DL models and automata models.

    DL values come from dl_experiment_summary.csv.
    Automata values come from runtime_summary.csv.
    """
    dl_df = load_csv("dl_experiment_summary.csv")
    runtime_df = load_csv("runtime_summary.csv")

    dl_plot = dl_df[["dataset", "model", "f1_mean"]].copy()
    automata_plot = runtime_df[
        runtime_df["model"].str.lower() == "automata"
    ][["dataset", "model", "f1_mean"]].copy()

    combined = pd.concat([dl_plot, automata_plot], ignore_index=True)
    combined["label"] = (
        combined["dataset"].astype(str).str.upper()
        + " - "
        + combined["model"].astype(str).str.upper()
    )

    combined = combined.sort_values(["dataset", "model"])

    plt.figure(figsize=(10, 5))
    plt.bar(combined["label"], combined["f1_mean"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("F1-score")
    plt.title("Model F1-score Comparison")
    save_current_figure("model_f1_comparison.png")


def plot_runtime_training_time() -> None:
    """Plots mean training time comparison from runtime_summary.csv."""
    df = load_csv("runtime_summary.csv").copy()
    df["label"] = (
        df["dataset"].astype(str).str.upper()
        + " - "
        + df["model"].astype(str).str.upper()
    )

    df = df.sort_values(["dataset", "model"])

    plt.figure(figsize=(10, 5))
    plt.bar(df["label"], df["training_time_sec_mean"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Mean Training Time, seconds")
    plt.title("Training Time Comparison")
    save_current_figure("runtime_training_time_comparison.png")


def plot_parameter_sensitivity_metric(metric: str, output_name: str, title: str) -> None:
    """
    Plots automata parameter sensitivity for a selected metric.

    X-axis combines window size and alphabet size.
    """
    df = load_csv("automata_param_search.csv").copy()

    if metric not in df.columns:
        raise ValueError(f"Metric column not found in automata_param_search.csv: {metric}")

    df["param_label"] = (
        "w="
        + df["window_size"].astype(str)
        + ", a="
        + df["alphabet_size"].astype(str)
    )

    df = df.sort_values(["dataset", "window_size", "alphabet_size"])

    for dataset_name, dataset_df in df.groupby("dataset"):
        plt.figure(figsize=(12, 5))
        plt.plot(dataset_df["param_label"], dataset_df[metric], marker="o")
        plt.xticks(rotation=60, ha="right")
        plt.ylabel(metric)
        plt.title(f"{title} - {dataset_name.upper()}")
        save_current_figure(f"{dataset_name}_{output_name}")


def extract_transitions_from_explainability(filename: str) -> List[Tuple[str, str, float]]:
    """Extracts transitions from an explainability JSON file."""
    data = load_json(filename)

    steps = data.get("steps", [])

    transitions = []

    for step in steps:
        transition = step.get("transition")

        if not transition:
            continue

        source = transition.get("from")
        target = transition.get("to")
        probability = transition.get("probability")

        if source is None or target is None or probability is None:
            continue

        transitions.append((str(source), str(target), float(probability)))

    return transitions


def plot_transition_probability_heatmap(
    explainability_file: str,
    output_name: str,
    max_states: int = 25
) -> None:
    """
    Plots transition probability heatmap from explainability transitions.

    To keep the figure readable, only the most frequent states are included.
    """
    transitions = extract_transitions_from_explainability(explainability_file)

    if not transitions:
        print(f"No transitions found in {explainability_file}. Skipping heatmap.")
        return

    state_counter = Counter()

    for source, target, _ in transitions:
        state_counter[source] += 1
        state_counter[target] += 1

    selected_states = [
        state for state, _ in state_counter.most_common(max_states)
    ]

    selected_set = set(selected_states)

    matrix = pd.DataFrame(
        0.0,
        index=selected_states,
        columns=selected_states
    )

    probability_accumulator = defaultdict(list)

    for source, target, probability in transitions:
        if source in selected_set and target in selected_set:
            probability_accumulator[(source, target)].append(probability)

    for (source, target), probabilities in probability_accumulator.items():
        matrix.loc[source, target] = sum(probabilities) / len(probabilities)

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix.values, aspect="auto")
    plt.colorbar(label="Transition Probability")
    plt.xticks(range(len(selected_states)), selected_states, rotation=90)
    plt.yticks(range(len(selected_states)), selected_states)
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.title(f"Transition Probability Heatmap - {explainability_file}")
    save_current_figure(output_name)


def plot_automata_state_diagram(
    explainability_file: str,
    output_name: str,
    max_edges: int = 35
) -> None:
    """
    Plots a readable automata state diagram.

    The diagram uses the most frequently observed transitions to avoid a dense graph.
    """
    transitions = extract_transitions_from_explainability(explainability_file)

    if not transitions:
        print(f"No transitions found in {explainability_file}. Skipping state diagram.")
        return

    transition_counter = Counter(
        (source, target)
        for source, target, _ in transitions
    )

    probability_values = defaultdict(list)

    for source, target, probability in transitions:
        probability_values[(source, target)].append(probability)

    selected_edges = transition_counter.most_common(max_edges)

    graph = nx.DiGraph()

    for (source, target), count in selected_edges:
        probabilities = probability_values[(source, target)]
        mean_probability = sum(probabilities) / len(probabilities)

        graph.add_edge(
            source,
            target,
            weight=count,
            probability=mean_probability
        )

    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=900
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowstyle="->",
        width=1.5
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=8
    )

    edge_labels = {
        (source, target): f"{data['probability']:.2f}"
        for source, target, data in graph.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=7
    )

    plt.axis("off")
    plt.title(f"Automata State Diagram - {explainability_file}")
    save_current_figure(output_name)


def main() -> None:
    """Generates all available report figures."""
    print("--- Generating report figures ---")

    plot_model_f1_comparison()
    plot_runtime_training_time()

    plot_parameter_sensitivity_metric(
        metric="f1_score",
        output_name="parameter_sensitivity_f1.png",
        title="Automata Parameter Sensitivity, F1-score"
    )

    plot_parameter_sensitivity_metric(
        metric="state_count",
        output_name="parameter_sensitivity_state_count.png",
        title="Automata Parameter Sensitivity, State Count"
    )

    plot_parameter_sensitivity_metric(
        metric="transition_density",
        output_name="parameter_sensitivity_transition_density.png",
        title="Automata Parameter Sensitivity, Transition Density"
    )

    plot_transition_probability_heatmap(
        explainability_file="skab_fold0_explainability.json",
        output_name="transition_probability_heatmap_skab_fold0.png"
    )

    plot_transition_probability_heatmap(
        explainability_file="batadal_fold0_explainability.json",
        output_name="transition_probability_heatmap_batadal_fold0.png"
    )

    plot_automata_state_diagram(
        explainability_file="skab_fold0_explainability.json",
        output_name="automata_state_diagram_skab_fold0.png"
    )

    plot_automata_state_diagram(
        explainability_file="batadal_fold0_explainability.json",
        output_name="automata_state_diagram_batadal_fold0.png"
    )

    print("--- Figure generation completed ---")


if __name__ == "__main__":
    main()