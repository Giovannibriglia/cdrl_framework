from typing import Dict, List

import numpy as np
import pandas as pd
import json

from gymnasium.spaces import Discrete, Box

from causality_vmas import LABEL_dir_storing_dict_and_info, LABEL_dict_causality, LABEL_causal_graph, LABEL_bn_dict, \
    LABEL_reward_action_values
from causality_vmas.causality_algos import SingleCausalInference
from causality_vmas.utils import list_to_graph


class CausalityDrivenActionSpaceFilter:
    def __init__(self, action_space_type, online: bool, path_previous_knowledge: bool):
        self.original_action_space_type = action_space_type
        self.dynamic_action_space = DynamicActionSpace(action_space_type)
        self.online = online

        if path_previous_knowledge:
            self.path_knowledge = path_previous_knowledge
            self.df_causality = pd.read_pickle(f'{self.path_knowledge}/best_df.pkl')

            if self.online:
                with open(f'{self.path_knowledge}(best_info.json', 'r') as f:
                    info_best = json.load(f)

                causal_graph_list = info_best[f'{LABEL_dict_causality}'][f'{LABEL_causal_graph}']
                causal_graph = list_to_graph(causal_graph_list)
                bn_info = info_best[f'{LABEL_dict_causality}'][f'{LABEL_bn_dict}']

                self.ci = SingleCausalInference(self.df_causality, causal_graph, bn_info)
            else:
                self.causal_table = pd.read_pickle(f'{self.path_knowledge}/causal_table.pkl')

        else:
            raise NotImplementedError('have no previous knowledge is not still implemented')

    def action_space_filter(self, obs: Dict):
        # TODO: rescale obs if is not in the bayesian network?
        reward_action_values = self._get_reward_action_values(obs)
        action_reward_scores = self.weighted_action_filter(reward_action_values)
        actions_to_discard = self.select_actions_to_remove(action_reward_scores)

        self.dynamic_action_space.constraint(actions_to_discard)
        # TODO: quale outuput?
        return self.dynamic_action_space

    def _get_reward_action_values(self, obs: Dict) -> Dict:
        if self.online:
            raise NotImplementedError
        else:
            filtered_df = self.causal_table.copy()
            for feature, value in obs.items():
                filtered_df = filtered_df[filtered_df[feature] == value]

            reward_action_values = filtered_df[LABEL_reward_action_values]

        return reward_action_values

    @staticmethod
    def weighted_action_filter(input_dict: Dict) -> Dict:
        def rescale_reward(past_value: float, old_min: float, old_max: float, new_min=0, new_max=1):
            new_value = new_min + ((past_value - old_min) / (old_max - old_min)) * (new_max - new_min)
            return new_value

        # Extract all reward values to find min and max for rescaling
        all_rewards = [reward for rewards_dict in input_dict.values() for reward in rewards_dict.keys()]
        old_min = float(min(all_rewards))
        old_max = float(max(all_rewards))

        averaged_mean_dict = {}
        for key_action in input_dict.values():
            for action, prob in key_action.items():
                if action not in averaged_mean_dict:
                    averaged_mean_dict[action] = 0

                # Here we assume that the reward_value is the key, and we are calculating a weighted sum
                reward_value = float(action)
                averaged_mean_dict[action] += prob * rescale_reward(reward_value, old_min, old_max)

        # Averaging the summed rewards based on the number of reward entries
        num_reward_entries = len(input_dict)
        for action in averaged_mean_dict:
            averaged_mean_dict[action] = round(averaged_mean_dict[action] / num_reward_entries, 4)

        return averaged_mean_dict

    @staticmethod
    def select_actions_to_remove(action_reward_scores: Dict) -> List:
        # Extract the values and calculate the 25th percentile
        values = list(action_reward_scores.values())
        percentile_25 = np.percentile(values, 25)

        return [key for key, value in action_reward_scores if value <= percentile_25]


class DynamicActionSpace(spaces.Space):
    def __init__(self, original_space):
        assert isinstance(original_space, (spaces.Discrete, spaces.Box)), "Original space must be Discrete or Box"

        self.original_space = original_space
        self.invalid_values = []
        self.invalid_intervals = []

        if isinstance(original_space, spaces.Discrete):
            self.valid_values = list(range(original_space.n))
            self.n = len(self.valid_values)
        elif isinstance(original_space, spaces.Box):
            self.low = original_space.low
            self.high = original_space.high

        super().__init__(original_space.shape, original_space.dtype)

    def sample(self):
        if isinstance(self.original_space, spaces.Discrete):
            return np.random.choice(self.valid_values)
        elif isinstance(self.original_space, spaces.Box):
            while True:
                sample = self.original_space.sample()
                if self._is_valid(sample):
                    return sample

    def contains(self, x):
        if isinstance(self.original_space, spaces.Discrete):
            return x in self.valid_values
        elif isinstance(self.original_space, spaces.Box):
            return self._is_valid(x)

    def _is_valid(self, x):
        for interval in self.invalid_intervals:
            if np.all(x >= interval[0]) and np.all(x <= interval[1]):
                return False
        return True

    def update_constraints(self, to_constraint: List[int] | List[Tuple] = None):
        if isinstance(self.original_space, spaces.Discrete):
            if to_constraint is not None:
                self.invalid_values = to_constraint
                self.valid_values = [i for i in range(self.original_space.n) if i not in self.invalid_values]
                self.n = len(self.valid_values)
        elif isinstance(self.original_space, spaces.Box):
            if to_constraint is not None:
                self.invalid_intervals = to_constraint
        else:
            raise NotImplementedError()

    def reset_constraints(self):
        self.invalid_values = []
        self.invalid_intervals = []

        if isinstance(self.original_space, spaces.Discrete):
            self.valid_values = list(range(self.original_space.n))
            self.n = len(self.valid_values)

    def __repr__(self):
        if isinstance(self.original_space, spaces.Discrete):
            return f"DynamicDiscrete({self.valid_values})"
        elif isinstance(self.original_space, spaces.Box):
            return f"DynamicBox({self.low}, {self.high}, {self.invalid_intervals})"

    def to_jsonable(self, sample_n):
        return [int(x) for x in sample_n] if isinstance(self.original_space, spaces.Discrete) else [x.tolist() for x in
                                                                                                    sample_n]
    @staticmethod
    def from_jsonable(sample_n):
        return np.array(sample_n)
