from typing import Dict, List, Union, Tuple
import os
import numpy as np
import pandas as pd
import json
import torch
import time

from causality_vmas import LABEL_grouped_features
from causality_vmas.s4_0_compute_causal_knowledge_for_rl import CausalInferenceForRL
from causality_vmas.utils import list_to_graph, inverse_approximation_function


class CausalActionsFilter:
    def __init__(self, online: bool, task: str, **kwargs):
        self.online = online
        self.task = task
        self.path_best = f'./causality_vmas/results_sensitive_analysis_{self.task}/best'

        self.df_train, self.causal_graph, self.dict_bn, self.causal_table, self.obs_train_to_test, self.grouped_features = self._get_task_items()

        self.ci_agents = None

        self.last_obs_continuous = None

    def _get_task_items(self) -> Tuple:
        df_train = pd.read_pickle(f'{self.path_best}/best_df.pkl')

        with open(f'{self.path_best}/best_causal_graph.json', 'r') as file:
            list_causal_graph = json.load(file)
        causal_graph = list_to_graph(list_causal_graph)

        with open(f'{self.path_best}/best_bn_params.json', 'r') as file:
            dict_bn = json.load(file)

        if os.path.exists(f'{self.path_best}/causal_table.pkl'):
            causal_table = pd.read_pickle(f'{self.path_best}/causal_table.pkl')
        else:
            causal_table = None

        obs_train_to_test = inverse_approximation_function(self.task)

        with open(f'{self.path_best}/best_others.json', 'r') as file:
            others = json.load(file)
        grouped_features = others[LABEL_grouped_features]

        return df_train, causal_graph, dict_bn, causal_table, obs_train_to_test, grouped_features

    def _init_causal_inference_agents(self, n_envs: int, n_agents: int):
        n = 1 # min(10, max(n_envs, n_agents))

        self.ci_agents = [
            CausalInferenceForRL(online=self.online, df_train=self.df_train, causal_graph=self.causal_graph,
                                 bn_dict=self.dict_bn, causal_table=self.causal_table,
                                 obs_train_to_test=self.obs_train_to_test,
                                 grouped_features=self.grouped_features) for _ in range(n)]

    def get_actions(self, multiple_observation: Union[List, torch.Tensor]) -> Union[List, torch.Tensor]:
        def validate_input(observation):
            if not isinstance(observation, (list, torch.Tensor)):
                raise ValueError('multiple_observation must be list or tensor')
            return observation

        def convert_to_tensor(observation):
            if isinstance(observation, list):
                return torch.tensor(observation)
            return observation

        def initialize_continuous(num_envs, num_agents):
            return {env_index: {agent_index: None for agent_index in range(num_agents)} for env_index in
                    range(num_envs)}

        def initialize_actions_to_discard(num_envs, num_agents):
            return {env_index: {agent_index: [] for agent_index in range(num_agents)} for env_index in range(num_envs)}

        def convert_actions_to_input_type(actions_to_discard, input_type):
            if input_type is list:
                return [[actions_to_discard[env_idx][agent_idx] for agent_idx in range(num_agents)] for env_idx in
                        range(num_envs)]
            else:
                return [[torch.tensor(actions_to_discard[env_idx][agent_idx], dtype=torch.bool) for agent_idx in
                         range(num_agents)] for env_idx in range(num_envs)]

        multiple_observation = validate_input(multiple_observation)
        input_type = type(multiple_observation)
        multiple_observation = convert_to_tensor(multiple_observation)
        num_envs, num_agents, obs_size = multiple_observation.shape

        self._init_causal_inference_agents(num_envs, num_agents)

        if self.last_obs_continuous is None:
            self.last_obs_continuous = initialize_continuous(num_envs, num_agents)

        actions_to_discard = initialize_actions_to_discard(num_envs, num_agents)

        tasks = [
            (env_idx, agent_idx, multiple_observation[env_idx, agent_idx],
             self.ci_agents[(env_idx * num_agents + agent_idx) % len(self.ci_agents)])
            for env_idx in range(num_envs) for agent_idx in range(num_agents)
        ]

        # with ThreadPoolExecutor() as pool:
        #with Pool(15) as pool:
            # results = pool.map(self.process_env_agent_helper, tasks)

        results = list(map(self.process_env_agent_helper, tasks))
        for env_idx, agent_idx, result in results:
            actions_to_discard[env_idx][agent_idx] = result

        return convert_actions_to_input_type(actions_to_discard, input_type)

    def process_env_agent_helper(self, args):
        return self.process_env_agent(*args)

    @staticmethod
    def calculate_delta_obs_continuous(current_obs, last_obs):
        return {f'agent_0_obs_{key}': (curr_obs - last_obs[key]) for key, curr_obs in enumerate(current_obs)}

    def process_env_agent(self, env_idx, agent_idx, current_obs_continuous, causal_inference_agent):
        actions_to_discard = []
        if self.last_obs_continuous[env_idx][agent_idx] is not None:
            delta_obs_continuous = self.calculate_delta_obs_continuous(current_obs_continuous,
                                                                  self.last_obs_continuous[env_idx][agent_idx])
            reward_action_values = causal_inference_agent.return_reward_action_values(delta_obs_continuous)
            action_reward_scores = self._weight_actions_rewards(reward_action_values)
            actions_to_discard = self._actions_mask_filter(action_reward_scores)
        self.last_obs_continuous[env_idx][agent_idx] = current_obs_continuous
        return env_idx, agent_idx, actions_to_discard

    @staticmethod
    def _weight_actions_rewards(reward_action_values: Dict) -> Dict:
        def rescale_reward(past_value: float, old_min: float, old_max: float, new_min: float = 0.0,
                           new_max: float = 1.0):
            if old_max == old_min:
                return new_min  # or return past_value if no rescaling is needed
            return new_min + ((past_value - old_min) / (old_max - old_min)) * (new_max - new_min)

        all_rewards = list(reward_action_values.keys())
        old_min = float(min(all_rewards))
        old_max = float(max(all_rewards))

        def process_action(reward_value, action_prob_pair):
            action, prob = action_prob_pair
            # Ensure prob is a float and not a dictionary or other type
            if isinstance(prob, dict):
                # Handle the case where prob is a dict, e.g., sum the probabilities or extract a specific value
                prob = sum(prob.values())  # Example: summing all values in the dict
            return action, prob * rescale_reward(reward_value, old_min, old_max)

        # Flatten the dictionary into a list of tuples: (action, weighted_reward)
        flattened_action_rewards = [
            process_action(reward_value, action_prob_pair)
            for reward_value, action_dict in reward_action_values.items()
            for action_prob_pair in action_dict.items()
        ]

        # Initialize an empty dictionary to hold the aggregated sums
        aggregated_sums = {}

        # Aggregate the results manually
        for action, weighted_reward in flattened_action_rewards:
            if action in aggregated_sums:
                aggregated_sums[action] += weighted_reward
            else:
                aggregated_sums[action] = weighted_reward

        num_reward_entries = len(reward_action_values)

        # Final averaging step
        averaged_mean_dict = {
            action: round(total / num_reward_entries, 4)
            for action, total in aggregated_sums.items()
        }

        return averaged_mean_dict

    @staticmethod
    def _actions_mask_filter(action_reward_scores: Dict) -> List:
        ordered_action_reward_scores = {k: action_reward_scores[k] for k in sorted(action_reward_scores)}

        values = list(ordered_action_reward_scores.values())
        percentile_25 = np.percentile(values, 25)

        actions_mask = [
            0 if value <= percentile_25
            else 1
            for key, value in ordered_action_reward_scores.items()
        ]

        if all(x == 0 for x in actions_mask):
            actions_mask = list(map(lambda value: 0 if value <= 0 else 1, ordered_action_reward_scores.values()))
            if all(x == 0 for x in actions_mask):
                actions_mask = [1] * len(actions_mask)

        # print(f'\n {ordered_action_reward_scores} - {actions_mask}')

        return actions_mask


def main(task: str, online: bool):
    cd_actions_filter = CausalActionsFilter(online, task)

    df_test = pd.read_pickle(f'./causality_vmas/dataframes/df_{task}_pomdp_discrete_actions_0.pkl')
    agent0_columns = [col for col in df_test.columns if
                      'agent_0' in col and 'action' not in col and 'reward' not in col]
    df_test = df_test.loc[:, agent0_columns]

    df_test = df_test.loc[:, agent0_columns]

    for n, row in df_test.iterrows():
        # Convert the row to a list of values and then to a PyTorch tensor
        input_row = torch.tensor(row.values)
        input_row = input_row.unsqueeze(0).unsqueeze(0)  # Adds two dimensions of size 1
        # Repeat the tensor to obtain a size of 10, 3, X
        input_row_repeated = input_row.repeat(10, 3, 1)
        in_time = time.time()
        mask = cd_actions_filter.get_actions(input_row_repeated)
        print(f'Single timestep: computation time for action mask : {round(time.time() - in_time, 3)} secs')
        # print(f'Single timestep: computation time for action mask : {round(time.time() - in_time, 3)} secs - {mask}')


if __name__ == '__main__':
    task_name = str(input('Select task: ')).lower()
    main(task_name, True)
