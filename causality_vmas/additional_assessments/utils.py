import time
from collections import Counter
from itertools import combinations
from typing import Dict, Tuple, List, Union, Set
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
from scipy.stats import ks_2samp
from sklearn.preprocessing import LabelEncoder

"""def evaluate_results_sensitive_analysis_causal_graphs(json_file_folder: str, employed_dataframe: pd.DataFrame,
                                                      x_semidim: float = None, y_semidim: float = None):
    graphs = []
    # Load all graphs from JSON files
    for file_name in os.listdir(json_file_folder):
        if file_name.endswith('.json'):
            with open(os.path.join(json_file_folder, file_name), 'r') as file:
                data = json.load(file)
                graph = data['causal_graph']
                file_name_to_save = file_name.replace('causal_graph_', '').replace('.json', '')
                graphs.append((file_name_to_save, nx.DiGraph(graph)))

    results = []

    # Compare each graph with every other graph
    pbar = tqdm(range(len(graphs)), 'Evaluation...')
    for i in pbar:
        for j in range(i, len(graphs)):
            file_name_1, G1 = graphs[i]
            file_name_2, G2 = graphs[j]

            n_bins1, n_sensors1 = _get_n_bins_and_n_sensors(file_name_1)
            n_bins2, n_sensors2 = _get_n_bins_and_n_sensors(file_name_2)

            single_row_results = compare_two_graphs(G1, G2, employed_dataframe, n_bins1, n_sensors1, n_bins2,
                                                    n_sensors2, x_semidim, y_semidim)

            single_row_results['n_bins_g1'] = n_bins1
            single_row_results['n_sensors_g1'] = n_sensors1
            single_row_results['n_bins_g2'] = n_bins2
            single_row_results['n_sensors_g2'] = n_sensors2

            results.append(single_row_results)

    results_df = pd.DataFrame(results)

    # get_df_boundaries(results_df)

    metrics = list(metrics_to_compare_causal_graphs().keys())
    for metric in metrics:
        plot_relationships(results_df, metric)

    results_df = group_as_similarity_score(results_df)

    return results_df


def group_as_similarity_score(data: pd.DataFrame):
    thresholds = {
        'shd': 5,
        'js': 0.8,
        'dds': 0.1,
        'ccs': 0.1,
        'kld': 0.1
    }

    def similarity_score(row, thresholds):
        score = 0
        score += (row['shd'] <= thresholds['shd'])
        score += (row['js'] >= thresholds['js'])
        score += (row['dds'] <= thresholds['dds'])
        score += (row['ccs'] <= thresholds['ccs'])
        score += (row['kld'] <= thresholds['kld'])
        return score

    # Generate all possible combinations of n_bins and n_sensors
    bins_sensors_combinations = list(
        itertools.product(data['n_bins_g1'].unique(), data['n_sensors_g1'].unique()))

    # Create a DataFrame to store the similarity scores for each combination
    results = []

    # Iterate over all combinations of bins and sensors
    for (bins1, sensors1), (bins2, sensors2) in itertools.product(bins_sensors_combinations, repeat=2):
        if bins1 == bins2 and sensors1 == sensors2:
            continue  # Skip comparing the same configuration

        subset1 = data[(data['n_bins_g1'] == bins1) & (data['n_sensors_g1'] == sensors1)]
        subset2 = data[(data['n_bins_g2'] == bins2) & (data['n_sensors_g2'] == sensors2)]

        if subset1.empty or subset2.empty:
            continue

        # Calculate the mean similarity score between the two subsets
        mean_score = (subset1.apply(similarity_score, axis=1, thresholds=thresholds).mean() +
                      subset2.apply(similarity_score, axis=1, thresholds=thresholds).mean()) / 2

        results.append({'n_bins1': bins1, 'n_sensors1': sensors1, 'n_bins2': bins2, 'n_sensors2': sensors2,
                        'mean_similarity_score': mean_score})

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Aggregate by averaging the mean_similarity_score for each unique combination
    aggregated_results = results_df.groupby(['n_bins1', 'n_sensors1']).agg(
        {'mean_similarity_score': 'mean'}).reset_index()

    # Sort by similarity scores to identify the best combinations
    best_combinations = aggregated_results.sort_values(by='mean_similarity_score', ascending=False)

    print(best_combinations.head(5))

    # Plot the best combinations for visual analysis
    plt.figure(figsize=(12, 8), dpi=1000)
    sns.heatmap(best_combinations.pivot(index='n_bins1', columns='n_sensors1', values='mean_similarity_score'),
                annot=True,
                fmt='.2f', cmap='Blues')
    plt.title('#bins and #sensors vs. similarity score')
    plt.xlabel('#sensors')
    plt.ylabel('#bins')
    plt.savefig('scores.png')


def metrics_to_compare_causal_graphs() -> Dict:
    metrics = {'shd': shd,
               # "sid": sid(G1, G2),
               # "ged": ged(G1, G2),
               "js": jaccard_similarity,
               "dds": degree_distribution_similarity,
               "ccs": clustering_coefficient_similarity,
               "kld": kullback_leibler_divergence,
               "gdda": gdda
               }
    return metrics


def compare_two_graphs(G1: nx.DiGraph, G2: nx.DiGraph, data: pd.DataFrame, n_bins1: int, n_sensors1: int, n_bins2: int,
                       n_sensors2: int, x_semidim: float = None, y_semidim: float = None):
    metrics = metrics_to_compare_causal_graphs()

    new_data1 = discretize_df(data, n_bins1, n_sensors1, x_semidim, y_semidim)
    new_data2 = discretize_df(data, n_bins2, n_sensors2, x_semidim, y_semidim)
    markov_blanket_and_equivalence_class(G1, G2, new_data1, new_data2)

    comparison_results = {}
    for metric_name, metric_func in metrics.items():
        try:
            comparison_results[metric_name] = metric_func(G1, G2)
        except Exception as e:
            comparison_results[metric_name] = str(e)

    return comparison_results


def _get_n_bins_and_n_sensors(input_string: str) -> Tuple[int, int]:
    # Define the regex pattern
    pattern = r"(\d+)bins_(\d+)sensors"

    # Search the string for the pattern
    match = re.search(pattern, input_string)

    if match:
        bins = int(match.group(1))
        sensors = int(match.group(2))
        # print(f"Input: {input_string}, Bins: {bins}, Sensors: {sensors}")
        return bins, sensors
    else:
        print("No match found")
        return None, None


def plot_relationships(data: pd.DataFrame, metric: str):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=500)

    # Plot metric vs. number of bins
    axes[0].scatter(data['n_bins_g1'], data[metric], label='g1', s=60)
    axes[0].scatter(data['n_bins_g2'], data[metric], label='g2', alpha=0.7, s=30)
    axes[0].set_title(f'{metric} vs. #bins')
    axes[0].set_xlabel('#bins')
    axes[0].set_ylabel(metric)
    axes[0].legend()

    # Plot metric vs. number of sensors
    axes[1].scatter(data['n_sensors_g1'], data[metric], label='g1', s=60)
    axes[1].scatter(data['n_sensors_g2'], data[metric], label='g2', alpha=0.7, s=30)
    axes[1].set_title(f'{metric} vs. #sensors')
    axes[1].set_xlabel('#sensors')
    axes[1].set_ylabel(metric)
    axes[1].legend()

    plt.tight_layout()
    plt.show()"""



