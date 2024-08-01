from typing import Dict, List, Union
import networkx as nx
import numpy as np
import pandas as pd
import json
import torch

from causality_vmas import LABEL_reward_action_values
from causality_vmas.s4_0_compute_causal_knowledge_for_rl import CausalInferenceForRL
from causality_vmas.utils import list_to_graph, discretize_value, get_numeric_part, extract_intervals_from_bn, \
    dict_to_bn, inverse_approximation_function
from obs_ex import obs_example

with open(f'causality_vmas/results_sensitive_analysis_navigation/best/best_approx_params.json', 'r') as file:
    approx_params = json.load(file)

online = True


class CausalActionsFilter:
    def __init__(self, online: bool, features_mapping: Dict,
                 causal_table: pd.DataFrame = None, df: pd.DataFrame = None,
                 causal_graph: nx.DiGraph = None, dict_bn: Dict = None, obs_train_to_test=None):
        self.online = online
        self.features_mapping = features_mapping

        self.ci = CausalInferenceForRL(self.online, df, causal_graph, dict_bn, causal_table,
                                       obs_train_to_test=obs_train_to_test)

        self.last_obs_continuous = None

    def get_actions(self, multiple_observation: Union[List, torch.Tensor]) -> Union[List, torch.Tensor]:
        if not isinstance(multiple_observation, (list, torch.Tensor)):
            raise ValueError('multiple_observation must be list or tensor')

        input_type = type(multiple_observation)

        if isinstance(multiple_observation, list):
            multiple_observation = torch.tensor(multiple_observation)

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

                    reward_action_values = self.ci.return_reward_action_values(delta_obs_continuous)
                    action_reward_scores = self._weighted_action_filter(reward_action_values)
                    actions_to_discard[env_idx][agent_idx] = self._select_actions_to_remove(action_reward_scores)

                self.last_obs_continuous[env_idx][agent_idx] = current_obs_continuous
        # print('Actions to discard: ', actions_to_discard)

        # Convert actions_to_discard to the same type as input
        if input_type is list:
            return [[actions_to_discard[env_idx][agent_idx] for agent_idx in range(num_agents)] for env_idx in
                    range(num_envs)]
        else:
            # Create a list to hold the action tensors for each environment and agent
            actions_list = []
            for env_idx in range(num_envs):
                env_actions = []
                for agent_idx in range(num_agents):
                    agent_actions = torch.tensor(
                        [float(action) for action in actions_to_discard[env_idx][agent_idx]], dtype=torch.float
                    )
                    env_actions.append(agent_actions)
                actions_list.append(env_actions)

            # Since we can't return a tensor of varying lengths directly, return the list of tensors
            return actions_list

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

    obs_train_to_test = inverse_approximation_function(task)

    cd_actions_filter = CausalActionsFilter(online, features_mapping, causal_table,
                                            df, causal_graph, dict_bn, obs_train_to_test)

    df_test = pd.read_pickle('./causality_vmas/dataframes/df_navigation_pomdp_discrete_actions_0.pkl')
    agent0_columns = [col for col in df_test.columns if
                      'agent_0' in col and 'action' not in col and 'reward' not in col]
    df_test = df_test.loc[:, agent0_columns]

    for n, row in df_test.iterrows():
        # Convert the row to a list of values and then to a PyTorch tensor
        input_row = torch.tensor(row.values)
        input_row = input_row.unsqueeze(0).unsqueeze(0)  # Adds two dimensions of size 1

        print(cd_actions_filter.get_actions(input_row))



if __name__ == '__main__':
    main()
    print(obs_example.size())
