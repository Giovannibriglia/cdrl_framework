from collections import Counter
from decimal import Decimal
from itertools import combinations
from typing import Dict
import random
import json
import ijson
import os

import networkx as nx
import numpy as np
import pandas as pd
import torch
from causalnex.structure import StructureModel
from matplotlib import pyplot as plt
from scipy.stats import ks_2samp, entropy

from path_repo import GLOBAL_PATH_REPO

" ******************************************************************************************************************** "


def detach_and_cpu(tensor):
    """Detach a tensor from the computation graph and move it to the CPU."""
    return tensor.detach().cpu()


def detach_dict(d, func=detach_and_cpu):
    """Recursively apply a function to all tensors in a dictionary, list, or tuple."""
    if isinstance(d, dict):
        return {k: detach_dict(v, func) for k, v in d.items()}
    elif isinstance(d, list):
        return [detach_dict(v, func) for v in d]
    elif isinstance(d, tuple):
        return tuple(detach_dict(v, func) for v in d)
    elif isinstance(d, torch.Tensor):
        return func(d)
    else:
        return d


" ******************************************************************************************************************** "


def define_causal_graph(list_for_causal_graph: list) -> StructureModel:
    # Create a StructureModel
    sm = StructureModel()

    # Add edges to the StructureModel
    for relationship in list_for_causal_graph:
        cause, effect = relationship
        sm.add_edge(cause, effect)

    return sm


" ******************************************************************************************************************** "


def IQM_mean_std(data: list) -> tuple:
    # Convert data to a numpy array
    data_array = np.array(data)

    # Sort the data
    sorted_data = np.sort(data_array)

    # Calculate quartiles
    Q1 = np.percentile(sorted_data, 25)
    Q3 = np.percentile(sorted_data, 75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Find indices of data within 1.5*IQR from Q1 and Q3
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    within_iqr_indices = np.where((sorted_data >= lower_bound) & (sorted_data <= upper_bound))[0]

    # Calculate IQM (Interquartile Mean)
    try:
        iq_mean = Decimal(np.mean(sorted_data[within_iqr_indices])).quantize(Decimal('0.001'))
    except:
        iq_mean = np.mean(sorted_data[within_iqr_indices])

    # Calculate IQM standard deviation (IQM_std)
    try:
        iq_std = Decimal(np.std(sorted_data[within_iqr_indices])).quantize(Decimal('0.001'))
    except:
        iq_std = np.std(sorted_data[within_iqr_indices])

    return iq_mean, iq_std


"""def compute_avg_series_for_agent(agent_data, metric_key):
    episode_mean_series = []

    timestep_data = {}
    # Iterate through each episode
    for episode in range(len(agent_data[metric_key])):
        # Collect data for each timestep across all environments within the current episode
        for step in range(len(agent_data[metric_key][episode])):
            for env in range(len(agent_data[metric_key][episode][step])):
                data_series = agent_data[metric_key][episode][step][env]
                if step not in timestep_data:
                    timestep_data[step] = []
                if data_series:
                    timestep_data[step].append(data_series)

    mean_list = []

    for step, data_series_list in timestep_data.items():
        combined_data = [value for series in data_series_list for value in series]

        if combined_data:
            if metric_key == 'rewards':
                mean_list.append(np.mean(combined_data))
            elif metric_key == 'time':
                mean_list.append(np.sum(combined_data))
            elif metric_key == 'n_collisions':
                mean_list.append(len([element for element in combined_data if element != 0]))
            elif metric_key == 'actions':
                mean_list.append(np.sum([element for element in combined_data if element != 0]))

    episode_mean_series.append(mean_list)

    avg_mean_series = np.mean(np.array(episode_mean_series), axis=0).tolist()

    return avg_mean_series"""


def compute_avg_series_for_agent(agent_data, metric_key):
    n_episodes = len(agent_data[metric_key])
    n_steps = len(agent_data[metric_key][0])
    n_env = len(agent_data[metric_key][0][0])

    list_episodes_series = []
    for episode in range(n_episodes):
        episode_series = []
        for step in range(n_steps):
            value_env = 0
            for env in range(n_env):
                for value in agent_data[metric_key][episode][step][env]:
                    if metric_key == 'rewards' or metric_key == 'time':
                        value_env += value
                    elif metric_key == 'actions':
                        value_env += 1
                    elif metric_key == 'n_collisions':
                        value_env += 1 if value != 0 else 0
            episode_series.append(value_env / n_env)
        list_episodes_series.append(episode_series)

    mean_list = np.mean(np.array(list_episodes_series), axis=0).tolist()

    return mean_list


" ******************************************************************************************************************** "


def _state_to_tuple(state):
    """Convert tensor state to a tuple to be used as a dictionary key."""
    return tuple(state.cpu().numpy())


def exploration_action(action_reward_values: Dict) -> int:
    weights = [s / (sum(action_reward_values.values())) for s in action_reward_values.values()]
    return random.choices(list(action_reward_values.keys()), weights=weights, k=1)[0]


def get_rl_knowledge(filepath, agent_id):
    with open(f'{GLOBAL_PATH_REPO}/{filepath}', 'r') as file:
        data = json.load(file)

    # Generate the agent key
    agent_key = f'agent_{agent_id}'

    # Retrieve the rl_knowledge for the given agent
    if agent_key in data:
        return data[agent_key]['rl_knowledge']
    else:
        raise KeyError(f'Agent ID {agent_id} not found in the data')


" ******************************************************************************************************************** "


# Function to calculate Structural Hamming Distance (SHD)
def shd(graph1, graph2):
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())

    # Edge additions and deletions
    additions = edges2 - edges1
    deletions = edges1 - edges2

    # Edge reversals
    reversals = set((v, u) for (u, v) in edges1).intersection(edges2)

    shd_value = len(additions) + len(deletions) + len(reversals)
    return shd_value


# Placeholder for Structural Intervention Distance (SID)
# Implementation of SID requires causal inference capabilities, not included in standard libraries
def sid(graph1, graph2):
    raise NotImplementedError("SID computation is not implemented")


# Function to calculate Graph Edit Distance (GED)
def ged(graph1, graph2):
    return nx.graph_edit_distance(graph1, graph2)


# Function to calculate Jaccard Similarity
def jaccard_similarity(graph1, graph2):
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())
    intersection = edges1.intersection(edges2)
    union = edges1.union(edges2)
    return len(intersection) / len(union)


# Function to calculate Degree Distribution Similarity (Kolmogorov-Smirnov)
def degree_distribution_similarity(graph1, graph2):
    degrees1 = [d for n, d in graph1.degree()]
    degrees2 = [d for n, d in graph2.degree()]
    return ks_2samp(degrees1, degrees2).statistic


# Function to calculate Clustering Coefficient Similarity
def clustering_coefficient_similarity(graph1, graph2):
    cc1 = nx.average_clustering(graph1)
    cc2 = nx.average_clustering(graph2)
    return abs(cc1 - cc2)


# Function to calculate Kullback-Leibler Divergence
def kullback_leibler_divergence(graph1, graph2):
    degrees1 = [d for n, d in graph1.degree()]
    degrees2 = [d for n, d in graph2.degree()]
    hist1 = np.histogram(degrees1, bins=range(max(degrees1) + 2), density=True)[0]
    hist2 = np.histogram(degrees2, bins=range(max(degrees2) + 2), density=True)[0]
    kl_div = entropy(hist1 + 1e-9, hist2 + 1e-9)  # Adding a small value to avoid division by zero
    return kl_div


# Function to calculate Graphlet Degree Distribution Agreement (GDDA)
def gdda(graph1, graph2, k=3):
    def count_graphlets(graph, k):
        graphlets = [graph.subgraph(nodes).copy() for nodes in combinations(graph.nodes(), k)]
        graphlet_degrees = [nx.degree_histogram(graphlet) for graphlet in graphlets]
        return Counter(tuple(map(tuple, graphlet_degrees)))

    count1 = count_graphlets(graph1, k)
    count2 = count_graphlets(graph2, k)

    all_graphlets = set(count1.keys()).union(set(count2.keys()))
    gdda_value = sum(abs(count1[g] - count2[g]) for g in all_graphlets)
    return gdda_value


" ******************************************************************************************************************** "


def _rescale_value(kind: str, value: float | int, n_bins: int, x_semidim: float = None, y_semidim: float = None):
    def discretize_value(value, intervals):
        # Find the interval where the value fits
        for i in range(len(intervals) - 1):
            if intervals[i] <= value < intervals[i + 1]:
                return (intervals[i] + intervals[i + 1]) / 2
        # Handle the edge cases
        if value < intervals[0]:
            return intervals[0]
        elif value >= intervals[-1]:
            return intervals[-1]

    def create_intervals(min_val, max_val, n_intervals, scale='linear'):
        if scale == 'exponential':
            # Generate n_intervals points using exponential scaling
            intervals = np.logspace(0, 1, n_intervals, base=10) - 1
            intervals = intervals / (10 - 1)  # Normalize to range 0-1
            intervals = min_val + (max_val - min_val) * intervals
        elif scale == 'linear':
            intervals = np.linspace(min_val, max_val, n_intervals)
        else:
            raise ValueError("Unsupported scale type. Use 'exponential' or 'linear'.")
        return intervals

    if kind == 'reward':
        max_value = 0.1
        min_value = -1 * 4 - max(x_semidim, y_semidim)

    elif kind == 'DX' or kind == 'DY':
        if x_semidim is None and y_semidim is None:
            return value
        else:
            max_value = x_semidim * 2 if kind == 'posX' else y_semidim * 2
            min_value = -x_semidim * 2 if kind == 'posX' else -y_semidim * 2

    elif kind == 'VX' or kind == 'VY':
        max_value = 0.5
        min_value = -0.5

    elif kind == 'sensor':
        max_value = 0.35
        min_value = 0.0

    elif kind == 'posX' or kind == 'posY':
        if x_semidim is None and y_semidim is None:
            return value
        else:
            max_value = x_semidim if kind == 'posX' else y_semidim
            min_value = -x_semidim if kind == 'posX' else y_semidim

    intervals = create_intervals(min_value, max_value, n_bins, scale='linear')

    index = discretize_value(value, intervals)
    rescaled_value = index

    return rescaled_value


def discretize_df(dataframe: pd.DataFrame, n_bins: int, n_sensor_to_consider: int, x_semidim: float = None,
                  y_semidim: float = None):
    dataframe = _group_variables(dataframe, variable_to_group='sensor', N=n_sensor_to_consider)
    std_dev = dataframe.std()
    non_zero_std_columns = std_dev[std_dev != 0].index

    df_filtered = dataframe[non_zero_std_columns]

    new_dataframe = pd.DataFrame(columns=df_filtered.columns, index=df_filtered.index)
    for col in df_filtered.columns:
        if 'action' not in col:
            if 'DX' in col:
                kind = 'DX'
            elif 'DY' in col:
                kind = 'DY'
            elif 'reward' in col:
                kind = 'reward'
            elif 'PX' in col:
                kind = 'posX'
            elif 'PY' in col:
                kind = 'posY'
            elif 'VX' in col:
                kind = 'VX'
            elif 'VY' in col:
                kind = 'VY'
            # elif 'sensor' in col:
            #    kind = 'sensor'
            else:
                kind = None

            if kind:
                new_dataframe[col] = df_filtered[col].apply(lambda value: _rescale_value(kind, value, n_bins, x_semidim, y_semidim))
            else:
                new_dataframe[col] = df_filtered[col]
        else:
            new_dataframe[col] = df_filtered[col]

    # print(new_dataframe['agent_0_next_DX'].std())
    # new_dataframe = new_dataframe.loc[:, new_dataframe.std() != 0]

    return new_dataframe


def _group_variables(dataframe: pd.DataFrame, variable_to_group: str, N: int = 1) -> pd.DataFrame:
    # Step 1: Identify columns to group
    variable_columns = [col for col in dataframe.columns if variable_to_group in col]

    # Step 2: Create columns for the top N variables
    for i in range(N):
        dataframe[f'agent_0_{variable_to_group}_on_{i}'] = None

    # Step 3: Determine top N variable values per row
    for index, row in dataframe.iterrows():
        sorted_variables = row[variable_columns].sort_values(ascending=False).index
        for i in range(N):
            if i <= len(sorted_variables):
                variable_number = sorted_variables[i - 1].split(variable_to_group)[1]
                dataframe.at[index, f'agent_0_{variable_to_group}_on_{i}'] = int(variable_number)
            else:
                dataframe.at[index, f'agent_0_{variable_to_group}_on_{i}'] = None

    # Step 4: Drop original variable columns
    dataframe.drop(columns=variable_columns, inplace=True)

    return dataframe


def get_df_boundaries(dataframe: pd.DataFrame):
    for col in dataframe.columns.to_list():
        iqm_mean, iqm_std = IQM_mean_std(dataframe[col])
        print(
            f'{col} -> max: {dataframe[col].max()}, min: {dataframe[col].min()}, mean: {dataframe[col].mean()}, std: {dataframe[col].std()}, iqm_mean: {iqm_mean}, iqm_std: {iqm_std}')

    fig = plt.figure(dpi=500, figsize=(16, 9))
    plt.plot(dataframe['agent_0_reward'])
    plt.show()