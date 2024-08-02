import gc
import itertools
import json
import multiprocessing
import os
import pickle
import random
import re
from decimal import Decimal
from typing import Dict, Tuple, List

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from matplotlib import pyplot as plt
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import ks_2samp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from causality_vmas import LABEL_kind_group_var, LABEL_value_group_var, LABEL_approximation_parameters, \
    LABEL_dataframe_approximated, LABEL_discrete_intervals

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


def list_to_graph(graph: list) -> nx.DiGraph:
    # Create a new directed graph
    dg = nx.DiGraph()

    # Add edges to the directed graph
    for cause, effect in graph:
        dg.add_edge(cause, effect)

    return dg


def graph_to_list(digraph: nx.DiGraph) -> list:
    return list(digraph.edges())


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
    with open(f'{filepath}', 'r') as file:
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
            f'{col} -> unique_values: {len(dataframe[col].value_counts())}, max: {dataframe[col].max()}, min: {dataframe[col].min()}, mean: {dataframe[col].mean()}, std: {dataframe[col].std()}, iqm_mean: {iqm_mean}, iqm_std: {iqm_std}')

    # fig = plt.figure(dpi=500, figsize=(16, 9))
    # plt.plot(dataframe['agent_0_reward'])
    # plt.show()


" ******************************************************************************************************************** "
FONT_SIZE_NODE_GRAPH = 7
ARROWS_SIZE_NODE_GRAPH = 30
NODE_SIZE_GRAPH = 1000


def plot_graph(graph: nx.DiGraph, title: str, if_show: bool = False, path_name_save: str = None):
    import warnings
    warnings.filterwarnings("ignore")

    fig = plt.figure(dpi=1000)
    plt.title(title, fontsize=16)
    nx.draw(graph, with_labels=True, font_size=FONT_SIZE_NODE_GRAPH,
            arrowsize=ARROWS_SIZE_NODE_GRAPH, arrows=True,
            edge_color='orange', node_size=NODE_SIZE_GRAPH, font_weight='bold', node_color='skyblue',
            pos=nx.circular_layout(graph))

    structure_to_save = graph_to_list(graph)
    if path_name_save is not None:
        plt.savefig(f'{path_name_save}.png')
        with open(f'{path_name_save}.json', 'w') as json_file:
            json.dump(structure_to_save, json_file)

    if if_show:
        plt.show()

    plt.close(fig)


def _get_numeric_suffix(var, label):
    if label in var:
        try:
            return int(var.split('_')[-1])
        except ValueError:
            return None
    return None


def constraints_causal_graph(causal_graph: nx.DiGraph):
    # TODO: procedura generale?
    # no value_sensor -> value_sensor or kind_sensor -> kind_sensor
    # no value_sensor1 -> kind_sensor0 or kind_sensor1 -> value_sensor0
    # no vel -> distGoal or distGoal -> vel
    # no distGoal -> value_sensor or value_sensor -> distGoal
    # no distGoal -> kind_sensor or kind_sensor -> distGoal

    # plot_graph(causal_graph, 'before constraints', True)

    edges_to_remove_sensor = [(u, v) for u, v in causal_graph.edges() if
                              (LABEL_value_group_var in u and LABEL_value_group_var in v) or
                              (LABEL_kind_group_var in u and LABEL_kind_group_var in v)]
    causal_graph.remove_edges_from(edges_to_remove_sensor)
    # print("Removed sensor edges:", [s for s in edges_to_remove_sensor if s not in causal_graph.edges()])

    edges_to_remove_sensor = [(u, v) for u, v in causal_graph.edges() if
                              ((_get_numeric_suffix(u, LABEL_value_group_var) is not None and
                                _get_numeric_suffix(v, LABEL_kind_group_var) is not None) or
                               (_get_numeric_suffix(v, LABEL_value_group_var) is not None and
                                _get_numeric_suffix(u, LABEL_kind_group_var) is not None))
                              ]
    causal_graph.remove_edges_from(edges_to_remove_sensor)
    # print("Removed sensor edges:", [s for s in edges_to_remove_sensor if s not in causal_graph.edges()])

    edges_to_remove_sensor = [(u, v) for u, v in causal_graph.edges() if
                              ('Vel' in u and 'Dist' in v) or
                              ('Dist' in u and 'Vel' in v)]
    causal_graph.remove_edges_from(edges_to_remove_sensor)
    # print("Removed sensor edges:", [s for s in edges_to_remove_sensor if s not in causal_graph.edges()])

    edges_to_remove_sensor = [(u, v) for u, v in causal_graph.edges() if
                              (LABEL_value_group_var in u and 'Dist' in v) or
                              ('Dist' in u and LABEL_value_group_var in v) or
                              (LABEL_kind_group_var in u and 'Dist' in v) or
                              ('Dist' in u and LABEL_kind_group_var in v)]
    causal_graph.remove_edges_from(edges_to_remove_sensor)
    # print("Removed sensor edges:", [s for s in edges_to_remove_sensor if s not in causal_graph.edges()])

    # plot_graph(causal_graph, 'after constraints', True)
    # print(causal_graph)
    return causal_graph


" ******************************************************************************************************************** "


def values_to_bins(values: List[float], intervals: List[float]) -> List[int]:
    # Sort intervals to ensure they are in ascending order
    intervals = sorted(intervals)

    # Initialize the list to store the bin index for each value
    new_values = []

    # Iterate through each value and determine its bin
    for value in values:
        for i in range(len(intervals) - 1):
            if intervals[i] <= value < intervals[i + 1]:
                new_values.append(i)
                break
        # To handle the case where the value is exactly equal to the last interval's end
        if value == intervals[-1]:
            new_values.append(len(intervals) - 2)

    return new_values


def discretize_value(value, intervals):
    idx = np.digitize(value, intervals, right=False)
    if idx == 0:
        return intervals[0]
    elif idx >= len(intervals):
        return intervals[-1]
    else:
        if abs(value - intervals[idx - 1]) <= abs(value - intervals[idx]):
            return intervals[idx - 1]
        else:
            return intervals[idx]


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


def discretize_dataframe(df, n_bins=50, scale='linear', not_discretize_these: List = None):
    discrete_df = df.copy()
    variable_discrete_intervals = {}
    for column in df.columns:
        if column not in not_discretize_these:
            min_value = df[column].min()
            max_value = df[column].max()
            intervals = _create_intervals(min_value, max_value, n_bins, scale)
            variable_discrete_intervals[column] = intervals.tolist()
            discrete_df[column] = df[column].apply(lambda x: discretize_value(x, intervals))
            # discrete_df[column] = np.vectorize(lambda x: intervals[_discretize_value(x, intervals)])(df[column])
    return discrete_df, variable_discrete_intervals


def group_variables(dataframe: pd.DataFrame, variable_to_group: str, N: int = 1) -> pd.DataFrame:
    # TODO: servirebbe qualcosa di automatico
    first_key = LABEL_kind_group_var
    second_key = LABEL_value_group_var

    # Step 1: Identify columns to group
    variable_columns = [col for col in dataframe.columns if variable_to_group in col]

    # Step 2: Create columns for the top N variables and their max values
    for i in range(N):
        dataframe[f'{first_key}_{variable_to_group}_{i}'] = None
        dataframe[f'{second_key}_{variable_to_group}_{i}'] = None

    # Step 3: Determine top N variable values per row
    for index, row in dataframe.iterrows():
        sorted_variables = row[variable_columns].sort_values(ascending=False).index
        for i in range(N):
            if i < len(sorted_variables):
                # Split and handle possible non-numeric values gracefully
                try:
                    variable_number = sorted_variables[i].split(variable_to_group)[1]
                    variable_number = ''.join(filter(str.isdigit, variable_number))  # Keep only numeric characters
                    dataframe.at[index, f'{first_key}_{variable_to_group}_{i}'] = int(variable_number)
                    dataframe.at[index, f'{second_key}_{variable_to_group}_{i}'] = row[sorted_variables[i]]
                except (IndexError, ValueError) as e:
                    print(e)
                    dataframe.at[index, f'{first_key}_{variable_to_group}_{i}'] = None
                    dataframe.at[index, f'{second_key}_{variable_to_group}_{i}'] = None
            else:
                dataframe.at[index, f'{first_key}_{variable_to_group}_{i}'] = None
                dataframe.at[index, f'{second_key}_{variable_to_group}_{i}'] = None

    # Step 4: Discretize max value columns
    for i in range(N):
        max_col_name = f'{second_key}_{variable_to_group}_{i}'
        new_col = dataframe[max_col_name]
        min_value = new_col.min()
        max_value = new_col.max()
        intervals = _create_intervals(min_value, max_value, 4, 'linear')
        dataframe[max_col_name] = new_col.apply(lambda x: discretize_value(x, intervals))

    # Step 5: Drop original variable columns if needed
    dataframe.drop(columns=variable_columns, inplace=True)
    return dataframe


def _navigation_approximation(input_elements: Tuple[pd.DataFrame, Dict]) -> Dict:
    df, params = input_elements
    n_bins = params.get('BINS', 20)
    n_rays = params.get('RAYS', 1)
    n_rows = params.get('ROWS', int(len(df) / 2))

    agent0_columns = [col for col in df.columns if 'agent_0' in col]
    df = df.loc[:, agent0_columns]

    not_discretize = [s for s in df.columns.to_list() if len(df[s].unique()) <= n_bins]
    new_df, discrete_intervals = discretize_dataframe(df, n_bins, not_discretize_these=not_discretize)
    new_df = group_variables(new_df, 'ray', n_rays)
    new_df = new_df.loc[:n_rows - 1, :]  # new_df.sample(n_rows-1, random_state=42)

    for col in new_df.columns.to_list():
        if len(new_df[col].unique()) > n_bins and 'kind' not in col:
            print(
                f'*** {n_bins} bins) Discretization problem in {col}: {len(new_df[col].unique())}, {new_df[col].unique()} *** ')

    approx_dict = {LABEL_approximation_parameters: {'n_bins': n_bins, 'n_rays': n_rays, 'n_rows': n_rows},
                   LABEL_discrete_intervals: discrete_intervals,
                   LABEL_dataframe_approximated: new_df}
    return approx_dict


def _navigation_inverse_approximation(input_obs: Dict, **kwargs) -> Dict:
    intervals = kwargs['discrete_intervals']
    features_bn = list(intervals.keys())

    n_rays = 0
    for feat in features_bn:
        if LABEL_kind_group_var in feat:
            n_rays += 1

    sensors_keys = [s for s in list(input_obs.keys()) if 'ray' in s]

    kind_rays = [0] * n_rays
    value_rays = [0.0] * n_rays

    for key in sensors_keys:
        for i in range(n_rays):
            if f'ray_{i}' in key:
                value = input_obs[key]
                if value > value_rays[i]:
                    value_rays[i] = value
                    kind_rays[i] = 1
                elif value == value_rays[i]:
                    kind_rays[i] += 1

    obs = {key: value for key, value in input_obs.items() if 'ray' not in key}

    for i in range(n_rays):
        obs[f'kind_ray_{i}'] = kind_rays[i]
        obs[f'value_ray_{i}'] = value_rays[i]

    final_obs = {}
    for key, value in obs.items():
        if 'ray' not in key and 'agent_0' not in key:
            final_obs[f'agent_0_{key}'] = value
        else:
            final_obs[key] = value

    final_obs = {key: discretize_value(value, intervals[key]) for key, value in final_obs.items()}

    return final_obs


def inverse_approximation_function(task: str):
    if task == 'navigation':
        return _navigation_inverse_approximation
    # TODO: others
    else:
        raise NotImplementedError("The inverse approximation function for this task has not been implemented")


def my_approximation(df: pd.DataFrame, task_name: str) -> List[Dict]:
    with open(f'./s2_0_params_sensitive_analysis.yaml', 'r') as file:
        config_approximation = yaml.safe_load(file)

        approx_params_task = config_approximation[task_name]
        params = {key: values for key, values in approx_params_task.items()}

        all_combs = list(itertools.product(*params.values()))
        formatted_combinations = [dict(zip(params.keys(), comb)) for comb in all_combs]

        all_params_list = [(df, single_task_combo_params) for single_task_combo_params in formatted_combinations]

        if task_name == 'navigation':
            approximator = _navigation_approximation
        # TODO: others
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            approximations = pool.map(approximator, all_params_list)
            gc.collect()

    return approximations


" ******************************************************************************************************************** "


def _get_next_index(folder_path, prefix):
    existing_files = os.listdir(folder_path)

    indices = []

    for filename in existing_files:
        if filename.startswith(prefix):
            idx_str = filename[len(prefix):].split('.')[0]
            if idx_str.isdigit():
                indices.append(int(idx_str))

    return max(indices) + 1 if indices else 0


def save_file_incrementally(file_content, folder_path, prefix, extension, is_binary=False):
    os.makedirs(folder_path, exist_ok=True)

    next_index = _get_next_index(folder_path, prefix)
    file_path = os.path.join(folder_path, f'{prefix}{next_index}.{extension}')

    if extension == 'pkl':
        with open(file_path, 'wb') as f:
            pickle.dump(file_content, f)
    elif is_binary:
        with open(file_path, 'wb') as f:
            f.write(file_content)
    else:
        with open(file_path, 'w') as f:
            f.write(file_content)

    # print(f'Saved file to {file_path}')


def save_json_incrementally(data, folder_path, prefix):
    os.makedirs(folder_path, exist_ok=True)

    next_index = _get_next_index(folder_path, prefix)
    file_path = os.path.join(folder_path, f'{prefix}{next_index}.json')
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=8)

    # print(f'Saved JSON to {file_path}')


" ******************************************************************************************************************** "


def is_folder_empty(folder_path):
    # Get the list of files and directories in the specified folder
    contents = os.listdir(folder_path)

    # Check if the list is empty
    if not contents:
        return True
    else:
        return False


" ******************************************************************************************************************** "


def bn_to_dict(model: BayesianNetwork):
    model_data = {
        "nodes": list(model.nodes()),
        "edges": list(model.edges()),
        "cpds": {}
    }

    for cpd in model.get_cpds():
        cpd_data = {
            "variable": cpd.variable,
            "variable_card": cpd.variable_card,
            "values": cpd.values.tolist(),
            "evidence": cpd.variables[1:],
            "evidence_card": cpd.cardinality[1:].tolist() if len(cpd.variables) > 1 else [],
            "state_names": cpd.state_names
        }
        model_data["cpds"][cpd.variable] = cpd_data

    return model_data


def dict_to_bn(model_data) -> BayesianNetwork:
    model = BayesianNetwork()
    model.add_nodes_from(model_data["nodes"])
    model.add_edges_from(model_data["edges"])

    for variable, cpd_data in model_data["cpds"].items():
        variable_card = cpd_data["variable_card"]
        evidence_card = cpd_data["evidence_card"]

        values = np.array(cpd_data["values"])
        if evidence_card:
            values = values.reshape(variable_card, np.prod(evidence_card))
        else:
            values = values.reshape(variable_card, 1)

        cpd = TabularCPD(
            variable=cpd_data["variable"],
            variable_card=variable_card,
            values=values.tolist(),
            evidence=cpd_data["evidence"],
            evidence_card=evidence_card,
            state_names=cpd_data["state_names"]
        )
        model.add_cpds(cpd)

    model.check_model()
    return model


def extract_intervals_from_bn(model: BayesianNetwork):
    intervals_dict = {}
    for node in model.nodes():
        cpd = model.get_cpds(node)
        if cpd:
            # Assuming discrete nodes with states
            intervals_dict[node] = cpd.state_names[node]
    return intervals_dict


" ******************************************************************************************************************** "


def split_dataframe(df: pd.DataFrame, num_splits: int) -> List:
    return [df.iloc[i::num_splits, :] for i in range(num_splits)]


" ******************************************************************************************************************** "


def group_features_by_distribution(df, auto_threshold=True, threshold=None):
    # TODO: fai da -1 a 1, così capisci la direzione
    """
    Groups features with similar distributions in a dataframe.

    Parameters:
    - df: pandas.DataFrame
        The input dataframe containing the features to be grouped.
    - auto_threshold: bool (default=True)
        Whether to automatically determine the threshold for clustering.
    - threshold: float (default=None)
        The threshold for the clustering algorithm to determine the number of clusters. Ignored if auto_threshold is True.

    Returns:
    - feature_clusters: dict
        A dictionary mapping each feature to a cluster.
    """

    # Step 1: Standardize the Data
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df, columns=df.columns)

    # Step 2: Measure Similarity
    correlation_matrix = scaled_df.corr().abs()

    # Handle NaN values in the correlation matrix
    correlation_matrix = correlation_matrix.fillna(0)

    # Step 3: Cluster Features
    distance_matrix = 1 - correlation_matrix

    # Ensure the distance matrix is symmetric
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Set diagonal elements to zero
    np.fill_diagonal(distance_matrix.values, 0)

    # Check if the distance matrix is symmetric
    if not np.allclose(distance_matrix, distance_matrix.T, atol=1e-8):
        print("Distance matrix:\n", distance_matrix)
        print("Transpose of distance matrix:\n", distance_matrix.T)
        raise ValueError("Distance matrix is not symmetric")

    linked = linkage(squareform(distance_matrix), 'ward')

    if auto_threshold:
        # Create dendrogram and determine the threshold automatically
        dendro = dendrogram(linked, no_plot=True)
        distances = dendro['dcoord']
        distances = sorted([y for x in distances for y in x[1:3]], reverse=True)
        diff = np.diff(distances)
        threshold = distances[np.argmax(diff) + 1]

    # Plot the dendrogram
    plt.figure(figsize=(10, 7), dpi=1000)
    sns.clustermap(correlation_matrix, row_linkage=linked, col_linkage=linked, cmap='coolwarm', annot=True)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.show()

    # Define clusters
    clusters = fcluster(linked, t=threshold, criterion='distance')

    # Map features to clusters
    feature_clusters = {df.columns[i]: clusters[i] for i in range(len(df.columns))}

    return feature_clusters


def plot_distributions(df):
    """
    Plots the distribution of each variable in the dataframe.

    Parameters:
    - df: pandas.DataFrame
        The input dataframe containing the variables to be plotted.
    """
    num_columns = len(df.columns)
    num_rows = (num_columns + 2) // 3  # Adjust rows for a better fit

    plt.figure(figsize=(15, num_rows * 5))

    for i, column in enumerate(df.columns, 1):
        plt.subplot(num_rows, 3, i)
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def get_numeric_part(input_string: str) -> int:
    numeric_part = re.findall(r'\d+', input_string)
    out_number = int(numeric_part[0]) if numeric_part else None
    return out_number


" ******************************************************************************************************************** "


def get_MeanAbsoluteError(errors):
    return np.mean(np.abs(errors))


def get_MeanSquaredError(errors):
    return np.mean(errors ** 2)


def get_RootMeanSquaredError(errors):
    return np.sqrt(get_MeanSquaredError(errors))


def get_MedianAbsoluteError(errors):
    return np.median(np.abs(errors))


" ******************************************************************************************************************** "


def get_adjacency_matrix(graph, order_nodes=None, weight=False) -> np.array:
    """Retrieve the adjacency matrix from the nx.DiGraph or numpy array."""
    if isinstance(graph, np.ndarray):
        return graph
    elif isinstance(graph, nx.DiGraph):
        if order_nodes is None:
            order_nodes = graph.nodes()
        if not weight:
            return np.array(nx.adjacency_matrix(graph, order_nodes, weight=None).todense())
        else:
            return np.array(nx.adjacency_matrix(graph, order_nodes).todense())
    else:
        raise TypeError("Only networkx.DiGraph and np.ndarray (adjacency matrixes) are supported.")


def get_StructuralHammingDistance(target, pred, double_for_anticausal=True) -> float:
    r"""Compute the Structural Hamming Distance.

    The Structural Hamming Distance (SHD) is a standard distance to compare
    graphs by their adjacency matrix. It consists in computing the difference
    between the two (binary) adjacency matrixes: every edge that is either
    missing or not in the target graph is counted as a mistake. Note that
    for directed graph, two mistakes can be counted as the edge in the wrong
    direction is false and the edge in the good direction is missing ; the
    `double_for_anticausal` argument accounts for this remark. Setting it to
    `False` will count this as a single mistake.

    Args:
        target (numpy.ndarray or networkx.DiGraph): Target graph, must be of
            ones and zeros.
        pred (numpy.ndarray or networkx.DiGraph): Prediction made by the
            algorithm to evaluate.
        double_for_anticausal (bool): Count the badly oriented edges as two
            mistakes. Default: True

    Returns:
        int: Structural Hamming Distance (int).

            The value tends to zero as the graphs tend to be identical."""

    true_labels = get_adjacency_matrix(target)
    predictions = get_adjacency_matrix(pred)  # , target.nodes() if isinstance(target, nx.DiGraph) else None)

    # Padding predictions to match the shape of true_labels
    pad_size = (true_labels.shape[0] - predictions.shape[0], true_labels.shape[1] - predictions.shape[1])
    padded_predictions = np.pad(predictions, ((0, pad_size[0]), (0, pad_size[1])), mode='constant', constant_values=0)

    diff = np.abs(true_labels - padded_predictions)
    if double_for_anticausal:
        return np.sum(diff)
    else:
        diff = diff + diff.transpose()
        diff[diff > 1] = 1  # Ignoring the double edges.
        return np.sum(diff) / 2


def get_StructuralInterventionDistance(target, pred) -> int:
    def find_intervention_distances(graph):
        distances = {}
        nodes = list(graph.nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                try:
                    dist = nx.shortest_path_length(graph, nodes[i], nodes[j])
                    distances[(nodes[i], nodes[j])] = dist
                    distances[(nodes[j], nodes[i])] = dist
                except nx.NetworkXNoPath:
                    distances[(nodes[i], nodes[j])] = float('inf')
                    distances[(nodes[j], nodes[i])] = float('inf')
        return distances

    true_distances = find_intervention_distances(target)
    estimated_distances = find_intervention_distances(pred)

    all_keys = set(true_distances.keys()).union(set(estimated_distances.keys()))

    sid = 0
    for key in all_keys:
        true_dist = true_distances.get(key, float('inf'))
        estimated_dist = estimated_distances.get(key, float('inf'))
        if true_dist != estimated_dist:
            sid += 1

    return sid


def get_FrobeniusNorm(target_graph, pred_graph) -> float:
    """
    Compute the Frobenius norm between two matrices.

    Parameters:
    target (np.ndarray): First matrix.
    pred (np.ndarray): Second matrix.

    Returns:
    float: Frobenius norm between target and pred.
    """

    target = get_adjacency_matrix(target_graph)
    pred = get_adjacency_matrix(pred_graph)

    def _resize_matrix(matrix: np.ndarray, new_shape: tuple) -> np.ndarray:
        """
        Resize a matrix to the new shape by padding with zeros if necessary.

        Parameters:
        matrix (np.ndarray): The matrix to resize.
        new_shape (tuple): The desired shape.

        Returns:
        np.ndarray: The resized matrix.
        """
        resized_matrix = np.zeros(new_shape)
        original_shape = matrix.shape
        resized_matrix[:original_shape[0], :original_shape[1]] = matrix
        return resized_matrix

    if target.shape != pred.shape:
        new_shape = (max(target.shape[0], pred.shape[0]), max(target.shape[1], pred.shape[1]))
        target = _resize_matrix(target, new_shape)
        pred = _resize_matrix(pred, new_shape)

    return np.linalg.norm(target - pred, 'fro')


def get_JaccardSimilarity(target, pred) -> float:
    def _intersection(a, b):
        return list(set(a) & set(b))

    def _union(a, b):
        return list(set(a) | set(b))

    nodes_target = set(target.nodes)
    nodes_pred = set(pred.nodes)
    all_nodes = nodes_target | nodes_pred
    similarity = 0
    num_pairs = 0

    for node1 in all_nodes:
        neighbors1 = list(target.successors(node1)) if node1 in target else []
        for node2 in all_nodes:
            neighbors2 = list(pred.successors(node2)) if node2 in pred else []

            intersection_size = len(_intersection(neighbors1, neighbors2))
            union_size = len(_union(neighbors1, neighbors2))

            if union_size > 0:
                similarity += intersection_size / union_size
                num_pairs += 1

    return (similarity / num_pairs) if num_pairs > 0 else 0


def get_DegreeDistributionSimilarity(target, pred) -> float:
    # Function to calculate Degree Distribution Similarity (Kolmogorov-Smirnov)
    # max = 1 = max diversity -> with "1-", 1 = max similarity
    degrees1 = [d for n, d in target.degree()]
    degrees2 = [d for n, d in pred.degree()]
    return 1 - ks_2samp(degrees1, degrees2).statistic


def get_ClusteringCoefficientSimilarity(target, pred) -> float:
    # Function to calculate Clustering Coefficient Similarity
    # max = 1 = max diversity -> with "1-", 1 = max similarity
    cc1 = nx.average_clustering(target)
    cc2 = nx.average_clustering(pred)
    return 1 - abs(cc1 - cc2)
