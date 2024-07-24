from collections import Counter
from decimal import Decimal
from itertools import combinations
import multiprocessing
from typing import Dict, Tuple, List
import random
import json
import os
import re
import networkx as nx
import seaborn as sns
import itertools
import numpy as np
import pandas as pd
import torch
import yaml
from causalnex.structure import StructureModel
from matplotlib import pyplot as plt

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


def list_to_causal_graph(list_for_causal_graph: list) -> nx.DiGraph:
    # Create a new directed graph
    dg = nx.DiGraph()

    # Add edges to the directed graph
    for cause, effect in list_for_causal_graph:
        dg.add_edge(cause, effect)

    return dg


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

"""def _rescale_value(kind: str, value: float | int, n_bins: int, x_semidim: float = None, y_semidim: float = None):
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


def discretize_df(dataframe: pd.DataFrame, n_bins: int):

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
            elif 'sensor' in col:
                kind = 'sensor'
            else:
                kind = None

            if kind:
                new_dataframe[col] = df_filtered[col].apply(
                    lambda value: _rescale_value(kind, value, n_bins, x_semidim, y_semidim))
            else:
                new_dataframe[col] = df_filtered[col]
        else:
            new_dataframe[col] = df_filtered[col]

    # print(new_dataframe['agent_0_next_DX'].std())
    # new_dataframe = new_dataframe.loc[:, new_dataframe.std() != 0]
   
    return new_dataframe"""


def get_df_boundaries(dataframe: pd.DataFrame):
    for col in dataframe.columns.to_list():
        # if isinstance(dataframe[col][0], (int, float, np.int64, np.float64)):
        iqm_mean, iqm_std = IQM_mean_std(dataframe[col])
        print(
            f'{col} -> max: {dataframe[col].max()}, min: {dataframe[col].min()}, mean: {dataframe[col].mean()}, std: {dataframe[col].std()}, iqm_mean: {iqm_mean}, iqm_std: {iqm_std}')

    # fig = plt.figure(dpi=500, figsize=(16, 9))
    # plt.plot(dataframe['agent_0_reward'])
    # plt.show()


" ******************************************************************************************************************** "


def _discretize_value(value, intervals):
    # Find the interval where the value fits
    for i in range(len(intervals) - 1):
        if intervals[i] <= value < intervals[i + 1]:
            return (intervals[i] + intervals[i + 1]) / 2
    # Handle the edge cases
    if value < intervals[0]:
        return intervals[0]
    elif value >= intervals[-1]:
        return intervals[-1]


def _create_intervals(min_val, max_val, n_intervals, scale='linear'):
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


def discretize_dataframe(df, n_bins=50, scale='linear'):
    discrete_df = df.copy()
    for column in df.columns:
        if 'action' not in column:
            min_value = df[column].min()
            max_value = df[column].max()
            intervals = _create_intervals(min_value, max_value, n_bins, scale)
            discrete_df[column] = df[column].apply(lambda x: _discretize_value(x, intervals))
    return discrete_df


" ******************************************************************************************************************** "


def group_variables(dataframe: pd.DataFrame, variable_to_group: str, N: int = 1) -> pd.DataFrame:
    # Step 1: Identify columns to group
    variable_columns = [col for col in dataframe.columns if variable_to_group in col]

    # Step 2: Create columns for the top N variables
    for i in range(N):
        dataframe[f'agent_0_{variable_to_group}_on_{i}'] = None

    # Step 3: Determine top N variable values per row
    for index, row in dataframe.iterrows():
        sorted_variables = row[variable_columns].sort_values(ascending=False).index
        for i in range(N):
            if i < len(sorted_variables):
                # Split and handle possible non-numeric values gracefully
                try:
                    variable_number = sorted_variables[i].split(variable_to_group)[1]
                    variable_number = ''.join(filter(str.isdigit, variable_number))  # Keep only numeric characters
                    dataframe.at[index, f'agent_0_{variable_to_group}_on_{i}'] = int(variable_number)
                except (IndexError, ValueError) as e:
                    print(e)
                    dataframe.at[index, f'agent_0_{variable_to_group}_on_{i}'] = None
            else:
                dataframe.at[index, f'agent_0_{variable_to_group}_on_{i}'] = None

    # Step 4: Add column with the max value for each row
    dataframe[f'max_value_{variable_to_group}'] = dataframe[variable_columns].max(axis=1)

    # Step 5: Drop original variable columns if needed
    dataframe.drop(columns=variable_columns, inplace=True)

    return dataframe


" ******************************************************************************************************************** "


def constraints_causal_graph(causal_graph: nx.DiGraph):
    edges_to_remove = [(u, v) for u, v in causal_graph.edges() if 'sensor' in u and 'sensor' in v]
    causal_graph.remove_edges_from(edges_to_remove)
    return causal_graph


" ******************************************************************************************************************** "


def _process_approximation(params):
    df, n_bins, n_sensors, n_rows = params
    new_df = discretize_dataframe(df, n_bins)
    new_df = group_variables(new_df, 'sensor', n_sensors)
    agent0_columns = [col for col in new_df.columns if 'agent_0' in col]
    new_df = new_df.loc[:n_rows, agent0_columns]
    approx_dict = {'new_df': new_df, 'n_bins': n_bins, 'n_sensors': n_sensors}
    return approx_dict


def my_approximation(df: pd.DataFrame) -> List[Dict]:
    with open(f'{GLOBAL_PATH_REPO}/causality_vmas/params_approximations.yaml', 'r') as file:
        config_approximation = yaml.safe_load(file)

    N_ROWS_APPROXIMATION = config_approximation['N_ROWS_APPROXIMATION']
    N_BINS_DISCR_LIST = config_approximation['N_BINS_DISCR_LIST']
    N_SENSORS_DISCR_LIST = config_approximation['N_SENSORS_DISCR_LIST']
    params_list = [(df, n_bins, n_sensors, n_rows) for n_bins in N_BINS_DISCR_LIST for n_sensors in N_SENSORS_DISCR_LIST for n_rows in N_ROWS_APPROXIMATION]

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        approximations = pool.map(_process_approximation, params_list)

    return approximations
