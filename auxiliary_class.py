from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union, Tuple
import networkx as nx
import numpy as np
import pandas as pd
import json
import torch
import time

from causality_vmas import LABEL_grouped_features
from causality_vmas.s4_0_compute_causal_knowledge_for_rl import CausalInferenceForRL
from causality_vmas.utils import list_to_graph, inverse_approximation_function

online = True


class CausalActionsFilter:
    def __init__(self, online: bool, task: str, **kwargs):
        self.online = online

        self.task = task
        self.path_best = f'./causality_vmas/results_sensitive_analysis_{self.task}/best'

        df_train, causal_graph, dict_bn, causal_table, obs_train_to_test, grouped_features = self.get_task_items()

        self.ci = CausalInferenceForRL(self.online, df_train, causal_graph, dict_bn, causal_table,
                                       obs_train_to_test=obs_train_to_test, grouped_features=grouped_features)
        self.last_obs_continuous = None

    """def get_actions(self, multiple_observation: Union[List, torch.Tensor]) -> Union[List, torch.Tensor]:
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
                current_obs_continuous = multiple_observation[env_idx, agent_idx]

                if self.last_obs_continuous[env_idx][agent_idx] is not None:
                    delta_obs_continuous = {f'agent_0_obs_{key}': (curr_obs - self.last_obs_continuous[env_idx][agent_idx][
                            key]) for key, curr_obs in enumerate(current_obs_continuous)}

                    reward_action_values = self.ci.return_reward_action_values(delta_obs_continuous, if_parallel=False)
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
            return actions_list"""

    def get_task_items(self) -> Tuple:
        df_train = pd.read_pickle(f'{self.path_best}/best_df.pkl')

        with open(f'{self.path_best}/best_causal_graph.json', 'r') as file:
            list_causal_graph = json.load(file)
        causal_graph = list_to_graph(list_causal_graph)

        with open(f'{self.path_best}/best_bn_params.json', 'r') as file:
            dict_bn = json.load(file)

        causal_table = pd.read_pickle(f'{self.path_best}/causal_table.pkl')

        obs_train_to_test = inverse_approximation_function(self.task)

        with open(f'{self.path_best}/best_others.json', 'r') as file:
            others = json.load(file)
        grouped_features = others[LABEL_grouped_features]

        return df_train, causal_graph, dict_bn, causal_table, obs_train_to_test, grouped_features

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

        def calculate_delta_obs_continuous(current_obs, last_obs):
            return {f'agent_0_obs_{key}': (curr_obs - last_obs[key]) for key, curr_obs in enumerate(current_obs)}

        def process_env_agent(env_idx, agent_idx, current_obs_continuous):
            actions_to_discard = []
            if self.last_obs_continuous[env_idx][agent_idx] is not None:
                delta_obs_continuous = calculate_delta_obs_continuous(current_obs_continuous,
                                                                      self.last_obs_continuous[env_idx][agent_idx])
                reward_action_values = self.ci.return_reward_action_values(delta_obs_continuous, if_parallel=False)
                action_reward_scores = self._weighted_action_filter(reward_action_values)
                actions_to_discard = self._select_actions_to_remove(action_reward_scores)
            self.last_obs_continuous[env_idx][agent_idx] = current_obs_continuous
            return env_idx, agent_idx, actions_to_discard

        def convert_actions_to_input_type(actions_to_discard, input_type):
            """if input_type is list:
                return [[actions_to_discard[env_idx][agent_idx] for agent_idx in range(num_agents)] for env_idx in
                        range(num_envs)]
            else:
                return [[torch.tensor(actions_to_discard[env_idx][agent_idx], dtype=torch.float) for agent_idx in
                         range(num_agents)] for env_idx in range(num_envs)]"""
            return actions_to_discard

        multiple_observation = validate_input(multiple_observation)
        input_type = type(multiple_observation)
        multiple_observation = convert_to_tensor(multiple_observation)
        num_envs, num_agents, obs_size = multiple_observation.shape

        if self.last_obs_continuous is None:
            self.last_obs_continuous = initialize_continuous(num_envs, num_agents)

        actions_to_discard = initialize_actions_to_discard(num_envs, num_agents)

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_env_agent, env_idx, agent_idx, multiple_observation[env_idx, agent_idx])
                for env_idx in range(num_envs) for agent_idx in range(num_agents)
            ]

            for future in futures:
                env_idx, agent_idx, result = future.result()
                actions_to_discard[env_idx][agent_idx] = result

        return convert_actions_to_input_type(actions_to_discard, input_type)

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


"""class ParallelCausalActionsFilter:
    def __init__(self, online: bool, causal_table: pd.DataFrame = None, df: pd.DataFrame = None,
                 causal_graph: nx.DiGraph = None, dict_bn: Dict = None, obs_train_to_test=None,
                 grouped_features: List = None):
        self.online = online
        self.df = df
        self.causal_graph = causal_graph
        self.dict_bn = dict_bn
        self.causal_table = causal_table
        self.obs_train_to_test = obs_train_to_test
        self.grouped_features = grouped_features
        self.last_obs_continuous = None
        self.ci_agents = None

    def initialize_agents(self, num_envs: int, num_agents: int):
        n = min(num_envs, num_agents)
        self.ci_agents = [CausalInferenceForRL(self.online, self.df, self.causal_graph, self.dict_bn,
                                               self.causal_table, obs_train_to_test=self.obs_train_to_test,
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

        def calculate_delta_obs_continuous(current_obs, last_obs):
            return {f'agent_0_obs_{key}': (curr_obs - last_obs[key]) for key, curr_obs in enumerate(current_obs)}

        def process_env_agent(env_idx, agent_idx, current_obs_continuous, agent):
            actions_to_discard = []
            if self.last_obs_continuous[env_idx][agent_idx] is not None:
                delta_obs_continuous = calculate_delta_obs_continuous(current_obs_continuous,
                                                                      self.last_obs_continuous[env_idx][agent_idx])
                reward_action_values = agent.return_reward_action_values(delta_obs_continuous, if_parallel=False)
                action_reward_scores = self._weighted_action_filter(reward_action_values)
                actions_to_discard = self._select_actions_to_remove(action_reward_scores)
            self.last_obs_continuous[env_idx][agent_idx] = current_obs_continuous
            return env_idx, agent_idx, actions_to_discard

        def convert_actions_to_input_type(actions_to_discard, input_type):
            return actions_to_discard

        multiple_observation = validate_input(multiple_observation)
        input_type = type(multiple_observation)
        multiple_observation = convert_to_tensor(multiple_observation)
        num_envs, num_agents, n_actions = multiple_observation.shape

        if self.ci_agents is None:
            self.initialize_agents(num_envs, num_agents)

        if self.last_obs_continuous is None:
            self.last_obs_continuous = initialize_continuous(num_envs, num_agents)

        actions_to_discard = initialize_actions_to_discard(num_envs, num_agents)

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_env_agent, env_idx, agent_idx, multiple_observation[env_idx, agent_idx],
                                self.ci_agents[(env_idx * num_agents + agent_idx) % len(self.ci_agents)])
                for env_idx in range(num_envs) for agent_idx in range(num_agents)
            ]

            for future in futures:
                env_idx, agent_idx, result = future.result()
                actions_to_discard[env_idx][agent_idx] = result

        return convert_actions_to_input_type(actions_to_discard, input_type)

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
        values = list(action_reward_scores.values())
        percentile_25 = np.percentile(values, 25)
        return [key for key, value in action_reward_scores.items() if value <= percentile_25]"""


def main(task='navigation'):

    cd_actions_filter = CausalActionsFilter(online, task)

    df_test = pd.read_pickle('./causality_vmas/dataframes/df_navigation_pomdp_discrete_actions_0.pkl')
    agent0_columns = [col for col in df_test.columns if
                      'agent_0' in col and 'action' not in col and 'reward' not in col]
    df_test = df_test.loc[:, agent0_columns]

    df_test = df_test.loc[:, agent0_columns]

    for n, row in df_test.iterrows():
        # Convert the row to a list of values and then to a PyTorch tensor
        input_row = torch.tensor(row.values)
        input_row = input_row.unsqueeze(0).unsqueeze(0)  # Adds two dimensions of size 1
        # Repeat the tensor to obtain a size of 10, 3, 18
        input_row_repeated = input_row.repeat(10, 3, 1)
        in_time = time.time()
        cd_actions_filter.get_actions(input_row_repeated)
        print('Single timestep: computation time for action mask : ', time.time() - in_time)


if __name__ == '__main__':
    main('flocking')
