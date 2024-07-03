from decimal import Decimal
from typing import Dict
import random
import json
import ijson
import os
import numpy as np
import torch
from causalnex.structure import StructureModel
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


def exploration_action(reward_action_values: Dict) -> int:
    return random.choices(list(reward_action_values.keys()), weights=list(reward_action_values.values()), k=1)[0]


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
