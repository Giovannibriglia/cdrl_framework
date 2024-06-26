from decimal import Decimal
from typing import Dict
import random
import numpy as np
import torch
from causalnex.structure import StructureModel

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
    iq_mean = Decimal(np.mean(sorted_data[within_iqr_indices])).quantize(Decimal('0.01'))

    # Calculate IQM standard deviation (IQM_std)
    iq_std = Decimal(np.std(sorted_data[within_iqr_indices])).quantize(Decimal('0.01'))

    return iq_mean, iq_std


def compute_iqm_and_std_for_agent(agent_data, metric_key):
    episode_iqm_means = []
    episode_iqm_stds = []

    # Iterate through each episode
    for episode in range(len(agent_data[metric_key])):
        timestep_data = {}

        # Collect data for each timestep across all environments within the current episode
        for step in range(len(agent_data[metric_key][episode])):
            for env in range(len(agent_data[metric_key][episode][step])):
                data_series = agent_data[metric_key][episode][step][env]
                if step not in timestep_data:
                    timestep_data[step] = []
                if data_series:
                    timestep_data[step].append(data_series)

        mean_list = []

        # Compute the mean for each timestep in the current episode
        for step, data_series_list in timestep_data.items():
            combined_data = [value for series in data_series_list for value in series]
            if combined_data:
                mean_list.append(np.mean(combined_data))

        # Compute IQM and STD for the current episode
        if mean_list:
            episode_iqm_mean, episode_iqm_std = IQM_mean_std(mean_list)
            episode_iqm_means.append(episode_iqm_mean)
            episode_iqm_stds.append(episode_iqm_std)

    return episode_iqm_means, episode_iqm_stds


" ******************************************************************************************************************** "

def _state_to_tuple(state):
    """Convert tensor state to a tuple to be used as a dictionary key."""
    return tuple(state.cpu().numpy())


def exploration_action(reward_action_values: Dict) -> int:
    return random.choices(list(reward_action_values.keys()), weights=list(reward_action_values.values()), k=1)[0]
