from typing import Dict, List, Union
import networkx as nx
import numpy as np
import pandas as pd
import json
import torch

from causality_vmas import  LABEL_reward_action_values
from causality_vmas.s4_0_compute_causal_knowledge_for_rl import CausalInferenceForRL
from causality_vmas.utils import list_to_graph, _discretize_value, get_numeric_part, extract_intervals_from_bn, \
    dict_to_bn
from obs_ex import obs_example

with open(f'causality_vmas/results_sensitive_analysis_navigation/best/best_approx_params.json', 'r') as file:
    approx_params = json.load(file)

online = True


class CausalityDrivenActionsFilter:
    def __init__(self, online: bool, features_mapping: Dict, discrete_intervals: Dict,
                 causal_table: pd.DataFrame = None, df: pd.DataFrame = None,
                 causal_graph: nx.DiGraph = None, dict_bn: Dict = None):
        self.online = online
        self.features_mapping = features_mapping

        if online:
            self.df = df
            self.bn = dict_bn
            self.causal_graph = causal_graph
            self.ci = CausalInferenceForRL(self.df, self.causal_graph, self.bn)  # Assuming this is defined elsewhere
        else:
            self.causal_table = causal_table

        self.last_obs_continuous = None

        self.intervals = discrete_intervals

    def get_actions_from_causality(self, multiple_observation: Union[List, torch.Tensor]) -> Dict:

        if isinstance(multiple_observation, list):
            multiple_observation = torch.tensor(multiple_observation)

        if not isinstance(multiple_observation, torch.Tensor):
            raise ValueError('multiple_observation must be list or tensor')

        num_envs, num_agents, obs_size = multiple_observation.shape

        if self.last_obs_continuous is None:
            self.last_obs_continuous = {}
            for env_index in range(num_envs):
                self.last_obs_continuous[env_index] = {}
                for agent_index in range(num_agents):
                    self.last_obs_continuous[env_index][agent_index] = None

        actions_to_discard = {}
        for env_index in range(num_envs):
            actions_to_discard[env_index] = {}
            for agent_index in range(num_agents):
                actions_to_discard[env_index][agent_index] = []

        for env_idx in range(num_envs):
            for agent_idx in range(num_agents):
                current_obs_continuous = {f'{name}': multiple_observation[env_idx, agent_idx][n].item() for n, name in
                                          enumerate(self.features_mapping.values())}
                if self.last_obs_continuous[env_idx][agent_idx] is not None:
                    delta_obs_continuous = {
                        key: (current_obs_continuous[key] - self.last_obs_continuous[env_idx][agent_idx][
                            key]) if "ray" not in key else current_obs_continuous[key] for key in
                        current_obs_continuous}

                    delta_obs_discrete = self._process_obs(delta_obs_continuous)

                    reward_action_values = self._get_reward_action_values(delta_obs_discrete)

                    action_reward_scores = self._weighted_action_filter(reward_action_values)
                    print('Action-Reward-Scores: ', action_reward_scores)
                    actions_to_discard[env_idx][agent_idx] = self._select_actions_to_remove(action_reward_scores)

                self.last_obs_continuous[env_idx][agent_idx] = current_obs_continuous
        print('Actions to discard: ', actions_to_discard)
        print('\n')
        return actions_to_discard

    def _get_reward_action_values(self, obs: Dict) -> Dict:

        def find_missing_values(input_dict):
            missing_values = {}
            for feature, value in input_dict.items():
                if feature in self.causal_table.columns:
                    unique_values = set(self.causal_table[feature].dropna().unique())
                    if value not in unique_values:
                        missing_values[feature] = value
            return missing_values

        # print(find_missing_values(obs))

        if self.online:
            reward_action_values = self.ci.single_query(obs)
        else:
            filtered_df = self.causal_table.copy()
            for feature, value in obs.items():
                filtered_df = filtered_df[filtered_df[feature] == value]

            reward_action_values = filtered_df[LABEL_reward_action_values].to_dict()

        return reward_action_values

    @staticmethod
    def _weighted_action_filter(reward_action_values: Dict) -> Dict:
        def rescale_reward(past_value: float, old_min: float, old_max: float, new_min=0, new_max=1):
            new_value = new_min + ((past_value - old_min) / (old_max - old_min)) * (new_max - new_min)
            return new_value

        all_rewards = list(reward_action_values.keys())

        old_min = float(min(all_rewards))
        old_max = float(max(all_rewards))

        averaged_mean_dict = {}
        for reward_value, action_dict in reward_action_values.items():
            for action, prob in action_dict.items():
                if action not in averaged_mean_dict:
                    averaged_mean_dict[action] = 0

                averaged_mean_dict[action] += prob * rescale_reward(reward_value, old_min, old_max)

        num_reward_entries = len(reward_action_values)
        for action in averaged_mean_dict:
            averaged_mean_dict[action] = round(averaged_mean_dict[action] / num_reward_entries, 4)

        return averaged_mean_dict

    @staticmethod
    def _select_actions_to_remove(action_reward_scores: Dict) -> List:
        # Extract the values and calculate the 25th percentile
        values = list(action_reward_scores.values())
        percentile_25 = np.percentile(values, 25)

        return [key for key, value in action_reward_scores.items() if value <= percentile_25]

    def _process_obs(self, input_obs: Dict) -> Dict:

        n_rays = approx_params['n_rays']
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

        def update_keys(original_dict):
            new_dict = {}
            for key, value in original_dict.items():
                new_key = f'agent_0_{key}' if 'ray' not in key else key
                new_dict[new_key] = value
            return new_dict

        new_obs = update_keys(obs)
        obs = {key: _discretize_value(value, self.intervals[key]) for key, value in new_obs.items()}

        return obs


def main(task='navigation'):
    path_best = f'./causality_vmas/results_sensitive_analysis_{task}/best'

    df = pd.read_pickle(f'{path_best}/best_df.pkl')

    with open(f'{path_best}/best_causal_graph.json', 'r') as file:
        list_causal_graph = json.load(file)
    causal_graph = list_to_graph(list_causal_graph)
    with open(f'{path_best}/best_bn_params.json', 'r') as file:
        dict_bn = json.load(file)
    with open(f'{path_best}/best_others.json', 'r') as file:
        others = json.load(file)

    causal_table = pd.read_pickle(f'{path_best}/causal_table.pkl')

    with open('causality_vmas/dataframes/info_navigation_pomdp_discrete_actions_0.json', 'r') as file:
        features_mapping = json.load(file)['features_mapping']

    bn = dict_to_bn(dict_bn)
    discrete_intervals = extract_intervals_from_bn(bn)

    cd_actions_filter = CausalityDrivenActionsFilter(online, features_mapping, discrete_intervals, causal_table,
                                                     df, causal_graph, dict_bn)

    df_test = pd.read_pickle('./causality_vmas/dataframes/df_navigation_pomdp_discrete_actions_0.pkl')
    agent0_columns = [col for col in df_test.columns if 'agent_0' in col and 'action' not in col and 'reward' not in col]
    df_test = df_test.loc[:, agent0_columns]

    for n, row in df_test.iterrows():
        # Convert the row to a list of values and then to a PyTorch tensor
        input_row = torch.tensor(row.values)
        input_row = input_row.unsqueeze(0).unsqueeze(0)  # Adds two dimensions of size 1
        # Ensure that cd_actions_filter.get_actions_from_causality is a callable function
        if callable(cd_actions_filter.get_actions_from_causality):
            actions = cd_actions_filter.get_actions_from_causality(input_row)
        else:
            raise TypeError("cd_actions_filter.get_actions_from_causality is not callable")


if __name__ == '__main__':
    # TODO: check if group "ray" variables works
    main()
    print(obs_example.size())
