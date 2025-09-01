import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path


def load_experiment_data(experiment_folder):
    """
    Load data from a single experiment folder.

    Args:
        experiment_folder (str): Path to the experiment folder

    Returns:
        dict: Dictionary containing the three JSON files' data, threshold if available, evaluation data if available, and topic data if available
    """
    experiment_path = Path(experiment_folder)

    # Load the three JSON files
    with open(experiment_path / "ablated_neurons.json", 'r') as f:
        ablated_neurons = json.load(f)

    with open(experiment_path / "cannot_ablate_neurons.json", 'r') as f:
        cannot_ablate_neurons = json.load(f)

    with open(experiment_path / "neurons_not_checked.json", 'r') as f:
        neurons_not_checked = json.load(f)

    result = {
        'ablated_neurons': ablated_neurons,
        'cannot_ablate_neurons': cannot_ablate_neurons,
        'neurons_not_checked': neurons_not_checked
    }

    # Load threshold.json if it exists
    threshold_file = experiment_path / "threshold.json"
    if threshold_file.exists():
        with open(threshold_file, 'r') as f:
            threshold_data = json.load(f)
            result['threshold'] = threshold_data.get('threshold')

    # Load topic.json if it exists
    topic_file = experiment_path / "topic.json"
    if topic_file.exists():
        with open(topic_file, 'r') as f:
            topic_data = json.load(f)
            result['topic_id'] = topic_data.get('topic_id')
            result['topic_name'] = topic_data.get('topic_name')

    # Load evaluation files if they exist
    v_eval_file = experiment_path / "v_eval.json"
    if v_eval_file.exists():
        with open(v_eval_file, 'r') as f:
            v_eval_data = json.load(f)
            result['v_eval'] = v_eval_data.get('hits_at_1')

    t_eval_file = experiment_path / "t_eval.json"
    if t_eval_file.exists():
        with open(t_eval_file, 'r') as f:
            t_eval_data = json.load(f)
            result['t_eval'] = t_eval_data.get('hits_at_1')

    return result


def load_all_experiments(results_folder, show_val=False, load_all=False):
    """
    Load experiment data from the ablation_results folder, filtering by folder name prefix.

    Args:
        results_folder (str): Path to the folder containing experiment folders
        show_val (bool): If True, load experiments starting with 'v_', otherwise load experiments starting with 't_'

    Returns:
        dict: Dictionary mapping experiment names to their data
    """
    experiments = {}
    results_path = Path(results_folder)

    if not results_path.exists():
        raise FileNotFoundError(f"Results folder '{results_folder}' not found")

    # Determine the prefix to filter by
    if load_all:
        print(f"Loading all experiments")
    else:
        prefix = 'v_' if show_val else 't_'
        print(f"Loading experiments with prefix: '{prefix}'")

    # Iterate through all subdirectories (experiment folders)
    for experiment_dir in results_path.iterdir():
        if experiment_dir.is_dir():
            experiment_name = experiment_dir.name
            # Only load experiments that start with the specified prefix
            if load_all or experiment_name.startswith(prefix):
                try:
                    experiments[experiment_name] = load_experiment_data(experiment_dir)
                    print(f"Loaded experiment: {experiment_name}")
                except Exception as e:
                    print(f"Error loading experiment {experiment_name}: {e}")
            else:
                print(f"Skipping experiment {experiment_name} (doesn't start with '{prefix}')")

    return experiments


def validate_neurons_not_checked(experiments):
    """
    Assert that neurons_not_checked.json is always an empty list.

    Args:
        experiments (dict): Dictionary of experiment data

    Raises:
        AssertionError: If any neurons_not_checked.json is not empty
    """
    print("\nValidating neurons_not_checked.json files...")

    for experiment_name, data in experiments.items():
        neurons_not_checked = data['neurons_not_checked']
        assert neurons_not_checked == [], f"Experiment {experiment_name} has non-empty neurons_not_checked: {neurons_not_checked}"
        print(f"✓ {experiment_name}: neurons_not_checked is empty")

    print("✓ All neurons_not_checked.json files are empty!")


def validate_no_intersection(experiments):
    """
    Assert that ablated_neurons and cannot_ablate_neurons do not intersect for each experiment.

    Args:
        experiments (dict): Dictionary of experiment data

    Raises:
        AssertionError: If any experiment has intersecting neurons between ablated and cannot_ablate
    """
    print("\nValidating no intersection between ablated and cannot_ablate neurons...")

    for experiment_name, data in experiments.items():
        ablated_neurons = set(tuple(neuron) for neuron in data['ablated_neurons'])
        cannot_ablate_neurons = set(tuple(neuron) for neuron in data['cannot_ablate_neurons'])

        intersection = ablated_neurons.intersection(cannot_ablate_neurons)
        assert len(intersection) == 0, f"Experiment {experiment_name} has intersecting neurons: {intersection}"
        print(f"✓ {experiment_name}: no intersection between ablated and cannot_ablate neurons")

    print("✓ All experiments have no intersection between ablated and cannot_ablate neurons!")


def validate_total_neurons(experiments):
    """
    Assert that the union of ablated_neurons and cannot_ablate_neurons equals 1024*4*6 for each experiment.

    Args:
        experiments (dict): Dictionary of experiment data

    Raises:
        AssertionError: If any experiment doesn't have exactly 6144 total neurons
    """
    print("\nValidating total neuron count (1024*4*6 = 6144)...")

    expected_total = 1024 * 4 * 6

    for experiment_name, data in experiments.items():
        ablated_neurons = set(tuple(neuron) for neuron in data['ablated_neurons'])
        cannot_ablate_neurons = set(tuple(neuron) for neuron in data['cannot_ablate_neurons'])

        # Union of both sets
        total_neurons = ablated_neurons.union(cannot_ablate_neurons)
        actual_total = len(total_neurons)

        assert actual_total == expected_total, f"Experiment {experiment_name} has {actual_total} total neurons, expected {expected_total}"
        print(
            f"✓ {experiment_name}: total neurons = {actual_total} (ablated: {len(ablated_neurons)}, cannot_ablate: {len(cannot_ablate_neurons)})")

    print(f"✓ All experiments have exactly {expected_total} total neurons!")


def create_cannot_ablate_heatmap(experiments, save_path="cannot_ablate_heatmap.png", show_plot=True):
    """
    Create a heatmap visualization of cannot_ablate_neurons across all experiments.

    Args:
        experiments (dict): Dictionary of experiment data
        save_path (str): Path to save the heatmap image
        show_plot (bool): If True, show the plot, otherwise just save
    """
    print("\nCreating heatmap for cannot_ablate_neurons...")

    # Count occurrences of each (layer, neuron_idx) pair
    neuron_counts = defaultdict(int)
    min_layer = None
    max_layer = 0
    max_neuron_idx = 0

    for experiment_name, data in experiments.items():
        cannot_ablate_neurons = data['cannot_ablate_neurons']

        for layer, neuron_idx in cannot_ablate_neurons:
            neuron_counts[(layer, neuron_idx)] += 1
            if min_layer is None:
                min_layer = layer
            else:
                min_layer = min(min_layer, layer)
            max_layer = max(max_layer, layer)
            max_neuron_idx = max(max_neuron_idx, neuron_idx)

    if min_layer is None:
        print("No neuron data found!")
        return

    print(f"Found neurons across layers {min_layer}-{max_layer} and neuron indices 0-{max_neuron_idx}")
    print(f"Total unique neuron locations: {len(neuron_counts)}")

    # Create the heatmap matrix (only for the actual layer range)
    num_layers = max_layer - min_layer + 1
    heatmap_matrix = np.zeros((num_layers, max_neuron_idx + 1))

    for (layer, neuron_idx), count in neuron_counts.items():
        # Map layer to matrix index (layer - min_layer)
        matrix_layer_idx = layer - min_layer
        heatmap_matrix[matrix_layer_idx, neuron_idx] = count

    # Create the visualization
    plt.figure(figsize=(20, 12))

    # Create the heatmap with actual layer labels
    layer_labels = [str(layer) for layer in range(min_layer, max_layer + 1)]

    sns.heatmap(heatmap_matrix,
                cmap='hot',
                cbar_kws={'label': 'Number of experiments where neuron cannot be ablated'},
                xticklabels=100,  # Show every 100th tick
                yticklabels=layer_labels)  # Show actual layer numbers

    plt.title(f'Retained Neurons Heatmap\n(across {len(experiments)} experiments, layers {min_layer}-{max_layer})',
              fontsize=16, fontweight='bold')
    plt.xlabel('Neuron Index', fontsize=12)
    plt.ylabel('Layer', fontsize=12)

    # Add statistics as text
    max_count = np.max(heatmap_matrix)
    total_neurons = np.sum(heatmap_matrix > 0)
    avg_count = np.mean(heatmap_matrix[heatmap_matrix > 0])

    stats_text = f'Max count: {max_count}\nTotal unique neurons: {total_neurons}\nAvg count (non-zero): {avg_count:.2f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {save_path}")
    if show_plot:
        plt.show()


def create_layer_ablation_distribution(experiments, save_path="layer_ablation_distribution.png", show_plot=True):
    """
    Show how many neurons can/cannot be ablated per layer across experiments.
    Combines both neuron count and percentage in a single graph using dual y-axes.

    Args:
        experiments (dict): Dictionary of experiment data0
        save_path (str): Path to save the visualization
        show_plot (bool): If True, show the plot, otherwise just save
    """
    print("\nCreating layer ablation distribution...")

    # Find the actual layer range
    min_layer = None
    max_layer = 0

    for data in experiments.values():
        for layer, neuron_idx in data['ablated_neurons'] + data['cannot_ablate_neurons']:
            if min_layer is None:
                min_layer = layer
            else:
                min_layer = min(min_layer, layer)
            max_layer = max(max_layer, layer)

    if min_layer is None:
        print("No neuron data found!")
        return

    layer_stats = defaultdict(lambda: {'ablated': 0, 'cannot_ablate': 0})

    for data in experiments.values():
        for layer, neuron_idx in data['ablated_neurons']:
            layer_stats[layer]['ablated'] += 1
        for layer, neuron_idx in data['cannot_ablate_neurons']:
            layer_stats[layer]['cannot_ablate'] += 1

    layers = sorted(layer_stats.keys())
    ablated_counts = [layer_stats[layer]['ablated'] for layer in layers]
    cannot_ablate_counts = [layer_stats[layer]['cannot_ablate'] for layer in layers]

    # Create single figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Calculate percentages
    total_per_layer = [ablated_counts[i] + cannot_ablate_counts[i] for i in range(len(layers))]
    ablated_pct = [ablated_counts[i] / total_per_layer[i] * 100 for i in range(len(layers))]
    cannot_ablate_pct = [cannot_ablate_counts[i] / total_per_layer[i] * 100 for i in range(len(layers))]

    # Create bars for neuron counts (left y-axis)
    bars1 = ax1.bar(layers, ablated_counts, label='Ablated (Count)', alpha=0.7, color='skyblue')
    bars2 = ax1.bar(layers, cannot_ablate_counts, bottom=ablated_counts, label='Retained (Count)', alpha=0.7,
                    color='lightcoral')

    # Annotate retained bars with absolute counts when retained >95%
    for i, bar in enumerate(bars2):
        if total_per_layer[i] > 0 and cannot_ablate_pct[i] > 95:
            count = cannot_ablate_counts[i]
            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = bar.get_y() + bar.get_height() / 2
            ax1.text(x_pos, y_pos, f'{count}', ha='center', va='center', fontsize=8, color='black', fontweight='bold')

    # Set up left y-axis for neuron counts
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Neuron Count', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create right y-axis for percentages
    ax2 = ax1.twinx()

    # Set up right y-axis for percentages (no lines, just for reference)
    ax2.set_ylabel('Percentage (%)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, 100)

    # Set title
    ax1.set_title('Neuron Ablation Status by Layer (Count + Percentage)', pad=20)

    # Legend for bars only
    ax1.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Layer ablation distribution saved to: {save_path}")
    if show_plot:
        plt.show()


def create_best_experiment_layer_distribution(experiments, save_path="best_experiment_layer_distribution.png",
                                              show_plot=True):
    """
    Show neuron ablation status by layer for the experiment with the least cannot_ablate neurons.
    Only works for experiments that have threshold data.

    Args:
        experiments (dict): Dictionary of experiment data
        save_path (str): Path to save the visualization
        show_plot (bool): If True, show the plot, otherwise just save
    """
    print("\nCreating layer ablation distribution for best experiment...")

    # Filter experiments that have threshold data
    threshold_experiments = {name: data for name, data in experiments.items() if 'threshold' in data}

    if not threshold_experiments:
        print("No experiments with threshold data found!")
        return

    # Find the experiment with the least cannot_ablate neurons
    best_experiment_name = None
    min_cannot_ablate = float('inf')

    for experiment_name, data in threshold_experiments.items():
        cannot_ablate_count = len(data['cannot_ablate_neurons'])
        if cannot_ablate_count < min_cannot_ablate:
            min_cannot_ablate = cannot_ablate_count
            best_experiment_name = experiment_name

    if best_experiment_name is None:
        print("Could not find best experiment!")
        return

    best_experiment_data = threshold_experiments[best_experiment_name]
    threshold_value = best_experiment_data['threshold']

    print(f"Best experiment: {best_experiment_name} (threshold: {threshold_value}, cannot_ablate: {min_cannot_ablate})")

    # Find the actual layer range for this experiment
    min_layer = None
    max_layer = 0

    for layer, neuron_idx in best_experiment_data['ablated_neurons'] + best_experiment_data['cannot_ablate_neurons']:
        if min_layer is None:
            min_layer = layer
        else:
            min_layer = min(min_layer, layer)
        max_layer = max(max_layer, layer)

    if min_layer is None:
        print("No neuron data found!")
        return

    # Calculate layer statistics for this experiment
    layer_stats = defaultdict(lambda: {'ablated': 0, 'cannot_ablate': 0})

    for layer, neuron_idx in best_experiment_data['ablated_neurons']:
        layer_stats[layer]['ablated'] += 1
    for layer, neuron_idx in best_experiment_data['cannot_ablate_neurons']:
        layer_stats[layer]['cannot_ablate'] += 1

    layers = sorted(layer_stats.keys())
    ablated_counts = [layer_stats[layer]['ablated'] for layer in layers]
    cannot_ablate_counts = [layer_stats[layer]['cannot_ablate'] for layer in layers]

    # Create single figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Calculate percentages
    total_per_layer = [ablated_counts[i] + cannot_ablate_counts[i] for i in range(len(layers))]
    ablated_pct = [ablated_counts[i] / total_per_layer[i] * 100 for i in range(len(layers))]
    cannot_ablate_pct = [cannot_ablate_counts[i] / total_per_layer[i] * 100 for i in range(len(layers))]

    # Create bars for neuron counts (left y-axis)
    bars1 = ax1.bar(layers, ablated_counts, label='Ablated (Count)', alpha=0.7, color='skyblue')
    bars2 = ax1.bar(layers, cannot_ablate_counts, bottom=ablated_counts, label='Retained (Count)', alpha=0.7,
                    color='lightcoral')

    # Annotate retained bars with absolute counts when retained >95%
    for i, bar in enumerate(bars2):
        if total_per_layer[i] > 0 and cannot_ablate_pct[i] <= 5:
            count = cannot_ablate_counts[i]
            x_pos = bar.get_x() + bar.get_width() / 2
            y_pos = bar.get_y() + bar.get_height() / 2
            ax1.text(x_pos, y_pos, f'{count}', ha='center', va='center', fontsize=8, color='black', fontweight='bold')

    # Set up left y-axis for neuron counts
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Neuron Count', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create right y-axis for percentages
    ax2 = ax1.twinx()

    # Set up right y-axis for percentages (no lines, just for reference)
    ax2.set_ylabel('Percentage (%)', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.set_ylim(0, 100)

    # Set title
    ax1.set_title(
        f'Neuron Ablation Status by Layer (Count + Percentage)\n(Best Experiment - Threshold: {threshold_value})',
        pad=20)

    # Legend for bars only
    ax1.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Best experiment layer distribution saved to: {save_path}")
    if show_plot:
        plt.show()


def create_threshold_vs_cannot_ablate_plot(experiments, save_path="threshold_vs_cannot_ablate.png", show_plot=True):
    """
    Create a scatter plot showing the relationship between threshold values and total cannot_ablate neurons.
    Only works for experiments that have threshold data.

    Args:
        experiments (dict): Dictionary of experiment data
        save_path (str): Path to save the visualization
        show_plot (bool): If True, show the plot, otherwise just save
    """
    print("\nCreating threshold vs retained neurons plot...")

    # Filter experiments that have threshold data
    threshold_experiments = {name: data for name, data in experiments.items() if 'threshold' in data}

    if not threshold_experiments:
        print("No experiments with threshold data found!")
        return

    # Extract threshold values and cannot_ablate counts
    thresholds = []
    cannot_ablate_counts = []
    experiment_names = []

    for experiment_name, data in threshold_experiments.items():
        threshold = data['threshold']
        cannot_ablate_count = len(data['cannot_ablate_neurons'])

        thresholds.append(threshold)
        cannot_ablate_counts.append(cannot_ablate_count)
        experiment_names.append(experiment_name)

    if not thresholds:
        print("No valid threshold data found!")
        return

    # Create the scatter plot
    plt.figure(figsize=(12, 8))

    # Scatter plot
    plt.scatter(thresholds, cannot_ablate_counts, alpha=0.7, s=100)

    # Add trend line if there are multiple points
    if len(thresholds) > 1:
        z = np.polyfit(thresholds, cannot_ablate_counts, 1)
        p = np.poly1d(z)
        plt.plot(thresholds, p(thresholds), "r--", alpha=0.8, label=f'Trend line')
        plt.legend()

    # Add labels for each point
    for i, (threshold, count, name) in enumerate(zip(thresholds, cannot_ablate_counts, experiment_names)):
        plt.annotate(f'{name}\n({threshold}, {count})',
                     (threshold, count),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.xlabel('Threshold Value')
    plt.ylabel('Total retained Neurons')
    plt.title('Threshold vs Retained Neurons\n(All experiments with threshold data)')
    plt.grid(True, alpha=0.3)

    # Add statistics
    min_threshold = min(thresholds)
    max_threshold = max(thresholds)
    min_cannot_ablate = min(cannot_ablate_counts)
    max_cannot_ablate = max(cannot_ablate_counts)

    stats_text = f'Threshold range: {min_threshold} - {max_threshold}\nRetained range: {min_cannot_ablate} - {max_cannot_ablate}\nTotal experiments: {len(thresholds)}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Threshold vs retained plot saved to: {save_path}")
    if show_plot:
        plt.show()


def create_experiment_similarity_matrix(experiments, save_path="experiment_similarity.png", show_plot=True):
    """
    Show how similar experiments are in terms of ablation patterns.

    Args:
        experiments (dict): Dictionary of experiment data
        save_path (str): Path to save the visualization
        show_plot (bool): If True, show the plot, otherwise just save
    """
    print("\nCreating experiment similarity matrix...")

    experiment_names = list(experiments.keys())
    similarity_matrix = np.zeros((len(experiment_names), len(experiment_names)))

    for i, exp1 in enumerate(experiment_names):
        for j, exp2 in enumerate(experiment_names):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                # Convert to sets for comparison
                cannot_ablate_1 = set(tuple(neuron) for neuron in experiments[exp1]['cannot_ablate_neurons'])
                cannot_ablate_2 = set(tuple(neuron) for neuron in experiments[exp2]['cannot_ablate_neurons'])

                # Jaccard similarity
                intersection = len(cannot_ablate_1.intersection(cannot_ablate_2))
                union = len(cannot_ablate_1.union(cannot_ablate_2))
                similarity_matrix[i, j] = intersection / union if union > 0 else 0

    # Check if experiments have threshold data or topic data
    has_threshold_data = any('threshold' in experiments[name] for name in experiment_names)
    has_topic_data = any('topic_name' in experiments[name] for name in experiment_names)

    if has_threshold_data:
        # Sort experiments by threshold (numeric first ascending), then by name
        sortable = []
        for idx, name in enumerate(experiment_names):
            thr = experiments[name].get('threshold')
            key = (0, float(thr)) if isinstance(thr, (int, float)) else (1, name)
            sortable.append((key, idx, name))
        sortable.sort(key=lambda x: x[0])
        sorted_indices = [orig_idx for _, orig_idx, _ in sortable]
        experiment_names = [nm for _, _, nm in sortable]
        # Reorder similarity matrix to match sorted experiments
        similarity_matrix = similarity_matrix[np.ix_(sorted_indices, sorted_indices)]

        # Use threshold values for labels (fallback to name if missing)
        x_labels = []
        y_labels = []
        for name in experiment_names:
            threshold = experiments[name].get('threshold')
            if threshold is not None and isinstance(threshold, (int, float)):
                x_labels.append(f'{threshold:.0f}')
                y_labels.append(f'{threshold:.0f}')
            else:
                x_labels.append(name)
                y_labels.append(name)
        x_axis_label = 'Threshold'
        y_axis_label = 'Threshold'
    elif has_topic_data:
        # Sort experiments by topic name for determinism
        sorted_pairs = sorted([(name, i) for i, name in enumerate(experiment_names)], key=lambda t: experiments[t[0]].get('topic_name', ''))
        sorted_indices = [i for _, i in sorted_pairs]
        experiment_names = [name for name, _ in sorted_pairs]
        similarity_matrix = similarity_matrix[np.ix_(sorted_indices, sorted_indices)]

        # Use topic names for labels (clean up for display)
        x_labels = []
        y_labels = []
        for name in experiment_names:
            topic_name = experiments[name].get('topic_name', name)
            if isinstance(topic_name, str) and '_' in topic_name:
                # Extract the descriptive part after the first underscore
                clean_name = topic_name.split('_', 1)[1]
                x_labels.append(clean_name)
                y_labels.append(clean_name)
            else:
                x_labels.append(str(topic_name))
                y_labels.append(str(topic_name))
        x_axis_label = 'Topic'
        y_axis_label = 'Topic'
    else:
        # Optional: sort by experiment name for determinism
        sorted_pairs = sorted([(name, i) for i, name in enumerate(experiment_names)], key=lambda t: t[0])
        sorted_indices = [i for _, i in sorted_pairs]
        experiment_names = [name for name, _ in sorted_pairs]
        similarity_matrix = similarity_matrix[np.ix_(sorted_indices, sorted_indices)]

        # Create numbered labels for experiments (1, 2, 3, ...)
        x_labels = [str(i + 1) for i in range(len(experiment_names))]
        y_labels = [str(i + 1) for i in range(len(experiment_names))]
        x_axis_label = 'Experiment Number'
        y_axis_label = 'Experiment Number'

    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix,
                xticklabels=x_labels,
                yticklabels=y_labels,
                cmap='coolwarm',
                cbar_kws={'label': 'Jaccard Similarity'})

    plt.title('Experiment Similarity Matrix\n(Based on retained neurons overlap)')
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Experiment similarity matrix saved to: {save_path}")
    if show_plot:
        plt.show()


def create_ablation_ratio_distribution(experiments, save_path="ablation_ratio_distribution.png", show_plot=True):
    """
    Show the distribution of ablation ratios across experiments.

    Args:
        experiments (dict): Dictionary of experiment data
        save_path (str): Path to save the visualization
        show_plot (bool): If True, show the plot, otherwise just save
    """
    print("\nCreating ablation ratio distribution...")

    ablation_ratios = []

    for data in experiments.values():
        total_neurons = len(data['ablated_neurons']) + len(data['cannot_ablate_neurons'])
        ablation_ratio = len(data['ablated_neurons']) / total_neurons
        ablation_ratios.append(ablation_ratio)

    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(ablation_ratios, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Ablation Ratio (Ablated/Total)')
    plt.ylabel('Number of Experiments')
    plt.title('Distribution of Ablation Ratios')
    mean_ratio = float(np.mean(ablation_ratios))
    plt.axvline(mean_ratio, color='red', linestyle='--', label=f'Mean: {mean_ratio:.0f}')
    plt.legend()

    # Box plot
    plt.subplot(1, 2, 2)
    plt.boxplot(ablation_ratios)
    plt.ylabel('Ablation Ratio')
    plt.title('Ablation Ratio Box Plot')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Ablation ratio distribution saved to: {save_path}")
    if show_plot:
        plt.show()


def print_experiment_statistics(experiments):
    """
    Print statistics about the experiments.

    Args:
        experiments (dict): Dictionary of experiment data
    """
    print(f"\n{'=' * 60}")
    print("EXPERIMENT STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total experiments: {len(experiments)}")

    total_ablated = 0
    total_cannot_ablate = 0

    for experiment_name, data in experiments.items():
        ablated_count = len(data['ablated_neurons'])
        cannot_ablate_count = len(data['cannot_ablate_neurons'])

        total_ablated += ablated_count
        total_cannot_ablate += cannot_ablate_count

        print(f"{experiment_name}:")
        print(f"  - Ablated neurons: {ablated_count}")
        print(f"  - Retained neurons: {cannot_ablate_count}")

    print(f"\nSUMMARY:")
    print(f"  - Total ablated neurons across all experiments: {total_ablated}")
    print(f"  - Total retained neurons across all experiments: {total_cannot_ablate}")
    print(f"  - Average ablated per experiment: {total_ablated / len(experiments):.0f}")
    print(f"  - Average retained per experiment: {total_cannot_ablate / len(experiments):.0f}")


def create_topic_vs_retained_plot(experiments, save_path="topic_vs_retained.png", show_plot=True):
    """
    Create a scatter plot showing the relationship between topic names and total retained neurons.
    Only works for experiments that have topic data loaded.

    Args:
        experiments (dict): Dictionary of experiment data
        save_path (str): Path to save the visualization
        show_plot (bool): If True, show the plot, otherwise just save
    """
    print("\nCreating topic vs retained neurons plot...")

    # Filter experiments that have topic data loaded
    topic_experiments = {name: data for name, data in experiments.items() if 'topic_name' in data and 'topic_id' in data}

    if not topic_experiments:
        print("No experiments with topic data found!")
        return

    # Extract topic information and cannot_ablate counts
    topic_names = []
    topic_ids = []
    cannot_ablate_counts = []
    experiment_names = []

    for experiment_name, data in topic_experiments.items():
        topic_name = data['topic_name']
        topic_id = data['topic_id']
        cannot_ablate_count = len(data['cannot_ablate_neurons'])

        topic_names.append(topic_name)
        topic_ids.append(topic_id)
        cannot_ablate_counts.append(cannot_ablate_count)
        experiment_names.append(experiment_name)

    if not topic_names:
        print("No valid topic data found!")
        return

    # Create the scatter plot
    plt.figure(figsize=(16, 10))

    # Create x-axis positions for topics
    x_positions = list(range(len(topic_names)))

    # Scatter plot
    plt.scatter(x_positions, cannot_ablate_counts, alpha=0.7, s=100, color='blue')

    # Add trend line if there are multiple points
    if len(cannot_ablate_counts) > 1:
        z = np.polyfit(x_positions, cannot_ablate_counts, 1)
        p = np.poly1d(z)
        plt.plot(x_positions, p(x_positions), "r--", alpha=0.8, label=f'Trend line')
        plt.legend()

    # Add labels for each point
    for i, (topic_name, count, exp_name) in enumerate(zip(topic_names, cannot_ablate_counts, experiment_names)):
        plt.annotate(f'{topic_name}\n({count})',
                     (i, count),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.xlabel('Topic')
    plt.ylabel('Total Retained Neurons')
    plt.title('Topic vs Retained Neurons\n(All topic-based experiments)')
    plt.grid(True, alpha=0.3)

    # Set x-axis labels to topic names
    plt.xticks(x_positions, [name.split('_', 1)[1] if '_' in name else name for name in topic_names],
               rotation=45, ha='right')

    # Add statistics
    min_cannot_ablate = min(cannot_ablate_counts)
    max_cannot_ablate = max(cannot_ablate_counts)
    avg_cannot_ablate = np.mean(cannot_ablate_counts)

    stats_text = f'Retained range: {min_cannot_ablate} - {max_cannot_ablate}\nAverage: {avg_cannot_ablate:.0f}\nTotal experiments: {len(cannot_ablate_counts)}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Topic vs retained plot saved to: {save_path}")
    if show_plot:
        plt.show()


def create_evaluation_hits_at_1_plot(experiments, save_path="evaluation_hits_at_1.png", show_plot=True):
    """
    Create a bar chart showing hits_at_1 values from v_eval.json and t_eval.json files for each experiment.
    If evaluation data is missing, shows "N/A" and uses experiment numbers on the x-axis.

    Args:
        experiments (dict): Dictionary of experiment data
        save_path (str): Path to save the visualization
        show_plot (bool): If True, show the plot, otherwise just save
    """
    print("\nCreating evaluation hits_at_1 plot...")

    # Include all experiments, not just those with evaluation data
    all_experiments = experiments
    if not all_experiments:
        print("No experiments found!")
        return

    # Check if any experiments have threshold data
    has_threshold_data = any('threshold' in data for data in all_experiments.values())

    # Extract data for plotting
    experiment_names = []
    x_values = []
    v_eval_values = []
    t_eval_values = []
    
    # Track if we need sequential numbering
    need_sequential_numbering = False
    if not has_threshold_data:
        # Check if any experiments have topic data loaded
        has_topic_data = any('topic_name' in data for data in all_experiments.values())
        need_sequential_numbering = not has_topic_data
        print(f"Need sequential numbering: {need_sequential_numbering}")
    
    # Counter for sequential numbering
    exp_counter = 1

    for experiment_name, data in all_experiments.items():
        if has_threshold_data and 'threshold' in data:
            # Use threshold value as x-axis value
            x_value = data['threshold']
            x_values.append(x_value)
        else:
            # Check if topic data is loaded for this experiment
            if 'topic_name' in data:
                topic_name = data['topic_name']
                x_values.append(topic_name)
            else:
                # If no topic data loaded and no threshold data, use sequential numbering
                if need_sequential_numbering:
                    print(f"Using sequential numbering for {experiment_name}: {exp_counter}")
                    x_values.append(exp_counter)
                    exp_counter += 1
                else:
                    # Fallback: extract numeric part from experiment name
                    if '_' in experiment_name:
                        exp_number = experiment_name.split('_', 1)[1]
                    else:
                        exp_number = experiment_name
                    x_values.append(exp_number)

        experiment_names.append(experiment_name)
        
        # Get eval values, defaulting to None if not available
        # Restore original behavior: flip sets for experiments starting with 't_'
        flip_sets = experiment_name.startswith('t_')
        v_eval_val = data.get('t_eval' if flip_sets else 'v_eval', None)
        t_eval_val = data.get('v_eval' if flip_sets else 't_eval', None)
        
        v_eval_values.append(v_eval_val)
        t_eval_values.append(t_eval_val)

    # Debug output to see what x-axis values we're using
    print(f"X-axis values: {x_values}")
    print(f"Experiment names: {experiment_names}")
    print(f"Validation values: {v_eval_values}")
    print(f"Test values: {t_eval_values}")

    # Create the bar chart
    plt.figure(figsize=(16, 10))

    # Set up the bar positions
    x = np.arange(len(experiment_names))
    width = 0.35

    if has_threshold_data:
        # Sort by threshold values for better visualization
        sorted_indices = np.argsort([float(x) if isinstance(x, (int, float)) else 0 for x in x_values])
        x_values = [x_values[i] for i in sorted_indices]
        v_eval_values = [v_eval_values[i] for i in sorted_indices]
        t_eval_values = [t_eval_values[i] for i in sorted_indices]
        experiment_names = [experiment_names[i] for i in sorted_indices]

        # Update x positions after sorting
        x = np.arange(len(experiment_names))

    # Create bars for validation and test data
    # Handle None values by setting them to 0 for visualization
    t_eval_plot_values = [val if val is not None else 0 for val in t_eval_values]
    v_eval_plot_values = [val if val is not None else 0 for val in v_eval_values]
    
    t_bars = plt.bar(x - width / 2, t_eval_plot_values,
                     width, label='Test', color='blue', alpha=0.7)
    v_bars = plt.bar(x + width / 2, v_eval_plot_values,
                     width, label='Validation', color='red', alpha=0.7)

    # Customize the plot
    if has_threshold_data:
        x_axis_label = 'Threshold'
        title_suffix = 'by Threshold'
    else:
        # Check if we have topic data loaded or experiment numbers
        has_topic_data = any('topic_name' in all_experiments[exp] for exp in experiment_names)
        print(f"Has topic data: {has_topic_data}")
        if has_topic_data:
            x_axis_label = 'Topic'
            title_suffix = 'by Topic'
        else:
            x_axis_label = 'Experiment Number'
            title_suffix = 'Experiment # (Sequential)'
    
    plt.xlabel(x_axis_label)
    plt.ylabel('Hits@1 Score')
    plt.title(f'Evaluation Hits@1 Scores {title_suffix}')

    # Set x-axis labels based on data type
    if has_threshold_data:
        # Use threshold values as x-axis labels
        plt.xticks(x, [f'{val:.0f}' if isinstance(val, (int, float)) else str(val) for val in x_values], rotation=0,
                   ha='center')
    else:
        # Check if we have topic data loaded or experiment numbers
        has_topic_data = any('topic_name' in all_experiments[exp] for exp in experiment_names)
        if has_topic_data:
            # Use topic names from loaded data when available
            # Clean up topic names for better display (remove prefix numbers)
            clean_topic_names = []
            for val in x_values:
                if isinstance(val, str) and '_' in val:
                    # Extract the descriptive part after the first underscore
                    clean_name = val.split('_', 1)[1]
                    clean_topic_names.append(clean_name)
                else:
                    clean_topic_names.append(str(val))
            plt.xticks(x, clean_topic_names, rotation=45, ha='right')
        else:
            # Use sequential experiment numbers as x-axis labels
            plt.xticks(x, x_values, rotation=0, ha='center')

    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add value labels on bars using the actual bar heights
    for i in range(len(experiment_names)):
        # Left bar = Test
        test_height = t_eval_plot_values[i]
        if test_height > 0:
            plt.text(i - width / 2, test_height + 0.01, f'{test_height:.3f}',
                     ha='center', va='bottom', fontsize=8)
        # Right bar = Validation
        val_height = v_eval_plot_values[i]
        if val_height > 0:
            plt.text(i + width / 2, val_height + 0.01, f'{val_height:.3f}',
                     ha='center', va='bottom', fontsize=8)

    # Add statistics
    v_vals = [val for val in v_eval_values if val is not None]
    t_vals = [val for val in t_eval_values if val is not None]
    
    # Count experiments with missing data
    missing_v_count = sum(1 for val in v_eval_values if val is None)
    missing_t_count = sum(1 for val in t_eval_values if val is None)

    if v_vals:
        v_avg = np.mean(v_vals)
        v_min = min(v_vals)
        v_max = max(v_vals)
    else:
        v_avg = v_min = v_max = 0

    if t_vals:
        t_avg = np.mean(t_vals)
        t_min = min(t_vals)
        t_max = max(t_vals)
    else:
        t_avg = t_min = t_max = 0

    stats_text = f'Validation - Avg: {v_avg:.3f}, Min: {v_min:.3f}, Max: {v_max:.3f} (Missing: {missing_v_count})\nTest - Avg: {t_avg:.3f}, Min: {t_min:.3f}, Max: {t_max:.3f} (Missing: {missing_t_count})\nTotal experiments: {len(experiment_names)}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation hits_at_1 plot saved to: {save_path}")
    if show_plot:
        plt.show()


def create_per_experiment_layer_distribution(experiments, save_dir, show_plot=True):
    """
    For each experiment, create a stacked bar chart per layer showing:
    - Ablated neuron count (bottom)
    - Retained neuron count (top)
    If retained < 5% of total per layer, place the retained count on top of the bar.

    Args:
        experiments (dict): Mapping experiment_name -> data dict with 'ablated_neurons' and 'cannot_ablate_neurons'
        save_dir (str or Path): Directory to save per-experiment figures
        show_plot (bool): If True, show each plot after saving
    """
    print("\nCreating per-experiment layer distributions...")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for experiment_name, data in experiments.items():
        # Aggregate counts per layer for this experiment
        layer_stats = defaultdict(lambda: {'ablated': 0, 'retained': 0})
        for layer, _ in data['ablated_neurons']:
            layer_stats[layer]['ablated'] += 1
        for layer, _ in data['cannot_ablate_neurons']:
            layer_stats[layer]['retained'] += 1

        if not layer_stats:
            print(f"No neuron data for experiment {experiment_name}; skipping.")
            continue

        layers = sorted(layer_stats.keys())
        ablated_counts = [layer_stats[layer]['ablated'] for layer in layers]
        retained_counts = [layer_stats[layer]['retained'] for layer in layers]
        totals = [ablated_counts[i] + retained_counts[i] for i in range(len(layers))]

        plt.figure(figsize=(12, 6))
        bars_abl = plt.bar(layers, ablated_counts, label='Ablated', color='skyblue', alpha=0.8)
        bars_ret = plt.bar(layers, retained_counts, bottom=ablated_counts, label='Retained', color='lightcoral', alpha=0.8)

        # Annotate retained counts when retained < 5% of total per layer
        for i, bar in enumerate(bars_ret):
            total = totals[i]
            retained = retained_counts[i]
            if total > 0 and (retained / total) < 0.05:
                x_pos = bar.get_x() + bar.get_width() / 2
                y_pos = bar.get_y() + bar.get_height() + 0.02 * max(totals)  # small offset above top
                plt.text(x_pos, y_pos, f"{retained}", ha='center', va='bottom', fontsize=8, fontweight='bold', color='black')

        plt.xlabel('Layer')
        plt.ylabel('Neuron Count')
        # Title uses threshold value if available, else experiment name
        threshold_value = data.get('threshold')
        if isinstance(threshold_value, (int, float)):
            title_text = f'{threshold_value:.0f}'
        else:
            title_text = experiment_name
        plt.title(f'Layer-wise Retained vs Ablated Neurons\nExperiment: {title_text}')
        plt.legend(loc='upper right')
        plt.tight_layout()

        out_path = save_dir / f"per_experiment_layers_{experiment_name}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-layer chart for {experiment_name} -> {out_path}")
        if show_plot:
            plt.show()
        else:
            plt.close()

def create_all_experiments_layer_grid(experiments, save_path="all_experiments_layers.png", show_plot=True):
    """
    Create a single figure with a grid of subplots (one per experiment),
    each showing stacked bars per layer (ablated bottom, retained top).
    Label retained count on top of a bar if retained < 5% of that layer's total.

    Args:
        experiments (dict): Mapping experiment_name -> data dict
        save_path (str): Output image path
        show_plot (bool): Whether to display the figure
    """
    print("\nCreating all-experiments layer grid...")

    # Order experiments by threshold value when available
    exp_items = []
    for name, data in experiments.items():
        thr = data.get('threshold')
        exp_items.append((name, thr))

    # Sort: experiments with a valid numeric threshold first (ascending),
    # then experiments without threshold (keep alphabetical by name for determinism)
    exp_items.sort(key=lambda item: (0, float(item[1])) if isinstance(item[1], (int, float)) else (1, item[0]))

    experiment_names = [name for name, _ in exp_items]
    num_experiments = len(experiment_names)
    if num_experiments == 0:
        print("No experiments to plot.")
        return

    # Grid size: try to make it roughly square
    cols = int(np.ceil(np.sqrt(num_experiments)))
    rows = int(np.ceil(num_experiments / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, experiment_name in enumerate(experiment_names):
        r = idx // cols
        c = idx % cols
        ax = axes[r][c]

        data = experiments[experiment_name]
        layer_stats = defaultdict(lambda: {'ablated': 0, 'retained': 0})
        for layer, _ in data['ablated_neurons']:
            layer_stats[layer]['ablated'] += 1
        for layer, _ in data['cannot_ablate_neurons']:
            layer_stats[layer]['retained'] += 1

        if not layer_stats:
            ax.set_title(f"{experiment_name} (no data)")
            ax.axis('off')
            continue

        layers = sorted(layer_stats.keys())
        ablated_counts = [layer_stats[layer]['ablated'] for layer in layers]
        retained_counts = [layer_stats[layer]['retained'] for layer in layers]
        totals = [ablated_counts[i] + retained_counts[i] for i in range(len(layers))]

        ax.bar(layers, ablated_counts, label='Ablated', color='skyblue', alpha=0.8)
        bars_ret = ax.bar(layers, retained_counts, bottom=ablated_counts, label='Retained', color='lightcoral', alpha=0.8)

        for i, bar in enumerate(bars_ret):
            total = totals[i]
            retained = retained_counts[i]
            if total > 0 and (retained / total) < 0.05:
                x_pos = bar.get_x() + bar.get_width() / 2
                y_pos = bar.get_y() + bar.get_height()
                ax.text(x_pos, y_pos, f"{retained}", ha='center', va='bottom', fontsize=7, fontweight='bold')

        threshold_value = data.get('threshold')
        if isinstance(threshold_value, (int, float)):
            ax.set_title(f'{threshold_value:.0f}')
        else:
            ax.set_title(experiment_name)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Neuron Count')

    # Hide any unused subplots
    for j in range(num_experiments, rows * cols):
        r = j // cols
        c = j % cols
        axes[r][c].axis('off')

    # Create a single legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=2)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"All-experiments layer grid saved to: {save_path}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

def create_topic_statistics_plot(save_path="topic_statistics.png", show_plot=True):
    """
    Create visualizations showing the topic statistics from select_top_k_confidence_topics.py.
    This shows the information printed at line 54 of that file.

    Args:
        save_path (str): Path to save the visualization
        show_plot (bool): If True, show the plot, otherwise just save
    """
    print("\nCreating topic statistics plot...")

    # Data from select_top_k_confidence_topics.py line 54 output
    topic_data = [
        {'topic_name': '0_song_album_singles_chart', 'queries': 89, 'documents': 1000, 'confidence': 0.998,
         'topic_id': 0},
        {'topic_name': '1_film_films_disney_movie', 'queries': 47, 'documents': 501, 'confidence': 0.978,
         'topic_id': 1},
        {'topic_name': '14_bank_financial_business_banking', 'queries': 13, 'documents': 78, 'confidence': 0.894,
         'topic_id': 14},
        {'topic_name': '2_season_series_tv_2017', 'queries': 17, 'documents': 205, 'confidence': 0.871, 'topic_id': 2},
        {'topic_name': '24_game_games_nintendo_xbox', 'queries': 14, 'documents': 59, 'confidence': 0.708,
         'topic_id': 24},
        {'topic_name': '12_hai_dil_singh_ki', 'queries': 10, 'documents': 80, 'confidence': 0.698, 'topic_id': 12},
        {'topic_name': '9_india_indian_gandhi_party', 'queries': 12, 'documents': 89, 'confidence': 0.684,
         'topic_id': 9},
        {'topic_name': '10_nerve_muscle_anterior_thyroid', 'queries': 12, 'documents': 85, 'confidence': 0.604,
         'topic_id': 10},
        {'topic_name': '3_actor_he_episode_film', 'queries': 21, 'documents': 165, 'confidence': 0.587, 'topic_id': 3},
        {'topic_name': '21_olympic_olympics_games_skating', 'queries': 13, 'documents': 63, 'confidence': 0.553,
         'topic_id': 21}
    ]

    # Extract data for plotting
    topic_names = [d['topic_name'] for d in topic_data]
    queries = [d['queries'] for d in topic_data]
    documents = [d['documents'] for d in topic_data]
    confidences = [d['confidence'] for d in topic_data]

    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

    # Plot 1: Queries per topic
    ax1.bar(range(len(topic_names)), queries, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Topic')
    ax1.set_ylabel('Number of Queries')
    ax1.set_title('Queries per Topic')
    ax1.set_xticks(range(len(topic_names)))
    ax1.set_xticklabels([name.split('_', 1)[1] if '_' in name else name for name in topic_names],
                        rotation=45, ha='right')

    # Plot 2: Documents per topic
    ax2.bar(range(len(topic_names)), documents, color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Topic')
    ax2.set_ylabel('Number of Documents')
    ax2.set_title('Documents per Topic')
    ax2.set_xticks(range(len(topic_names)))
    ax2.set_xticklabels([name.split('_', 1)[1] if '_' in name else name for name in topic_names],
                        rotation=45, ha='right')

    # Plot 3: Confidence per topic
    ax3.bar(range(len(topic_names)), confidences, color='orange', alpha=0.7)
    ax3.set_xlabel('Topic')
    ax3.set_ylabel('Confidence Score')
    ax3.set_title('Confidence per Topic')
    ax3.set_xticks(range(len(topic_names)))
    ax3.set_xticklabels([name.split('_', 1)[1] if '_' in name else name for name in topic_names],
                        rotation=45, ha='right')

    # Plot 4: Scatter plot of queries vs confidence
    ax4.scatter(queries, confidences, s=100, alpha=0.7, color='red')
    ax4.set_xlabel('Number of Queries')
    ax4.set_ylabel('Confidence Score')
    ax4.set_title('Queries vs Confidence')

    # Add trend line
    if len(queries) > 1:
        z = np.polyfit(queries, confidences, 1)
        p = np.poly1d(z)
        ax4.plot(queries, p(queries), "b--", alpha=0.8, label=f'Trend line')
        ax4.legend()

    # Add topic labels to scatter plot
    for i, (query, conf, name) in enumerate(zip(queries, confidences, topic_names)):
        ax4.annotate(f'{name.split("_", 1)[1] if "_" in name else name}',
                     (query, conf),
                     xytext=(5, 5),
                     textcoords='offset points',
                     fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Topic statistics plot saved to: {save_path}")
    if show_plot:
        plt.show()


def main():
    """
    Main function to run the complete analysis and visualization.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize ablation experiment results')
    parser.add_argument('--experiments_path',
                        type=str,
                        default='results',
                        help='Path to the folder containing experiment folders (default: results)')
    parser.add_argument('--visualizations',
                        type=str,
                        nargs='+',
                        choices=['all', 'heatmap', 'layer_dist', 'similarity', 'ratio_dist', 'best_experiment', 'per_experiment_layers', 'all_experiments_layers',
                                 'threshold_vs_cannot_ablate', 'topic_vs_retained', 'topic_stats',
                                 'evaluation_hits_at_1'],
                        default=['all'],
                        help='Which visualizations to create (default: all)')
    parser.add_argument('--show_val',
                        action='store_true',
                        help='If set, load experiments starting with "v_" instead of "t_"')
    parser.add_argument('--all',
                        action='store_true',
                        help='If set, load all experiments starting with "v_" or "t_"')
    parser.add_argument('--no-show',
                        action='store_true',
                        help='If set, disable showing graphs (they will still be saved)')
    args = parser.parse_args()

    print(f"Loading experiment data from: {args.experiments_path}")

    # Create visualizations directory in experiments_path
    experiments_path = Path(args.experiments_path)
    visualizations_dir = experiments_path / "visualizations"
    visualizations_dir.mkdir(exist_ok=True)
    print(f"Visualizations will be saved to: {visualizations_dir}")

    # Load experiments based on the show_val flag
    experiments = load_all_experiments(args.experiments_path, show_val=args.show_val, load_all=args.all)

    if not experiments:
        print("No experiments found!")
        return

    # Validate neurons_not_checked.json files
    validate_neurons_not_checked(experiments)

    # Validate no intersection between ablated and cannot_ablate neurons
    validate_no_intersection(experiments)

    # Validate total neuron count
    validate_total_neurons(experiments)

    # Print statistics
    print_experiment_statistics(experiments)

    # Determine which visualizations to run
    if 'all' in args.visualizations:
        visualizations_to_run = ['heatmap', 'layer_dist', 'similarity', 'ratio_dist', 'best_experiment',
                                 'threshold_vs_cannot_ablate', 'topic_vs_retained', 'topic_stats',
                                 'evaluation_hits_at_1', 'per_experiment_layers', 'all_experiments_layers']
    else:
        visualizations_to_run = args.visualizations

    print(f"\nCreating visualizations: {', '.join(visualizations_to_run)}")

    # Create visualizations based on user choice
    if 'heatmap' in visualizations_to_run:
        create_cannot_ablate_heatmap(experiments, str(visualizations_dir / "cannot_ablate_heatmap.png"),
                                     show_plot=not args.no_show)

    if 'layer_dist' in visualizations_to_run:
        create_layer_ablation_distribution(experiments, str(visualizations_dir / "layer_ablation_distribution.png"),
                                           show_plot=not args.no_show)

    if 'similarity' in visualizations_to_run:
        create_experiment_similarity_matrix(experiments, str(visualizations_dir / "experiment_similarity.png"),
                                            show_plot=not args.no_show)

    if 'ratio_dist' in visualizations_to_run:
        create_ablation_ratio_distribution(experiments, str(visualizations_dir / "ablation_ratio_distribution.png"),
                                           show_plot=not args.no_show)

    if 'best_experiment' in visualizations_to_run:
        create_best_experiment_layer_distribution(experiments,
                                                  str(visualizations_dir / "best_experiment_layer_distribution.png"),
                                                  show_plot=not args.no_show)

    if 'threshold_vs_cannot_ablate' in visualizations_to_run:
        create_threshold_vs_cannot_ablate_plot(experiments, str(visualizations_dir / "threshold_vs_cannot_ablate.png"),
                                               show_plot=not args.no_show)

    if 'topic_vs_retained' in visualizations_to_run:
        create_topic_vs_retained_plot(experiments, str(visualizations_dir / "topic_vs_retained.png"),
                                           show_plot=not args.no_show)

    if 'topic_stats' in visualizations_to_run:
        create_topic_statistics_plot(str(visualizations_dir / "topic_statistics.png"), show_plot=not args.no_show)

    if 'evaluation_hits_at_1' in visualizations_to_run:
        create_evaluation_hits_at_1_plot(experiments, str(visualizations_dir / "evaluation_hits_at_1.png"),
                                         show_plot=not args.no_show)

    if 'per_experiment_layers' in visualizations_to_run:
        # Save into a dedicated subfolder under visualizations
        per_exp_dir = visualizations_dir / "per_experiment_layers"
        create_per_experiment_layer_distribution(experiments, per_exp_dir, show_plot=not args.no_show)

    if 'all_experiments_layers' in visualizations_to_run:
        create_all_experiments_layer_grid(experiments, str(visualizations_dir / "all_experiments_layers.png"), show_plot=not args.no_show)

    print("\nAnalysis complete!")
    print(f"All visualizations have been saved to: {visualizations_dir}")


if __name__ == "__main__":
    main()
