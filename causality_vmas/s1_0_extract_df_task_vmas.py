import importlib
import os
import sys
import time
from typing import Dict, Tuple

import pandas as pd
import torch
import yaml
from torch import Tensor
from tqdm import tqdm

from causality_vmas.utils import save_file_incrementally, save_json_incrementally
from vmas import make_env
from vmas.simulator.heuristic_policy import RandomPolicy, BaseHeuristicPolicy
from vmas.simulator.utils import save_video


class VMASExperiment:
    def __init__(self,
                 scenario_name: str,
                 **kwargs
                 ):

        self.df_story = None
        self.scenario_name = scenario_name
        self.dict_actions = True
        self.dict_space = True

        params_sim, params_scenario = self._get_input_params()

        self.mdp = params_sim.get('mdp', False)
        self.n_steps = params_sim.get('n_steps_for_env', 10000)
        self.n_envs = params_sim.get('n_envs', 10)

        self.continuous_actions = params_scenario.get('continuous_actions', True)

        self.n_agents = params_scenario.get('n_agents', None)
        self.seed = params_scenario.get('seed', 42)

        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.render = kwargs.get('render', False)
        self.save_render = kwargs.get('save_render', False)

        kwargs_env = params_scenario
        self.env = make_env(scenario=scenario_name, device=self.device, dict_spaces=self.dict_space,
                            num_envs=self.n_envs, **kwargs_env)
        if self.n_agents is None:
            self.n_agents = self.env.n_agents

        if self.continuous_actions:
            if params_sim.get('random_action', True):
                self.policy = RandomPolicy(self.continuous_actions)
            else:
                self.policy = self._import_heuristic_policy()
        else:
            self.policy = self.env.get_random_action

        self.list_story = []

        observation_space = self.env.get_observation_space()
        n_features = observation_space['agent_0'].shape[0]

        self.last_obs = None if self.mdp else {f'agent_{i}': [[0] * n_features] * self.n_envs for i in
                                               range(self.n_agents)}
        self.last_rewards = None if self.mdp else {f'agent_{i}': [0] * self.n_envs for i in
                                                   range(self.n_agents)}

        self.dict_info = params_scenario.copy()
        self.dict_info.update(params_sim)

        self.dir_save = f'./dataframes'
        os.makedirs(self.dir_save, exist_ok=True)

        self.feature_names = None

    def _import_heuristic_policy(self) -> BaseHeuristicPolicy:
        try:
            module_name = f'vmas.scenarios.{self.scenario_name}'
            module = importlib.import_module(module_name)
            HeuristicPolicy = getattr(module, 'HeuristicPolicy')

            kwargs = {'continuous_action': self.continuous_actions}

            policy = HeuristicPolicy(**kwargs)

            return policy
        except:
            raise NotImplementedError('No heuristic policy has been implemented for this scenario')

    def _get_input_params(self) -> Tuple[Dict, Dict]:
        with open(f's1_0_configs.yaml', 'r') as file:
            config = yaml.safe_load(file)

        params_sim = config['simulation']
        params_scenario = config['scenario'][self.scenario_name]

        return params_sim, params_scenario

    def _generate_feature_names(self, total_features: int):
        return [f'obs_{n}' for n in range(total_features)]

    def _select_actions(self, obs: Tensor):
        actions = {} if self.dict_actions else []
        for n_agent, agent in enumerate(self.env.agents):
            if self.continuous_actions:
                if self.dict_space:
                    actions_agent = self.policy.compute_action(obs[agent.name], agent.u_range)
                else:
                    actions_agent = self.policy.compute_action(obs[n_agent], agent.u_range)
                actions_agent.to(self.device)
            else:
                actions_agent = self.policy(agent)

            if self.dict_actions:
                actions[agent.name] = actions_agent
            else:
                actions.append(actions_agent)

        return actions

    def _store_data(self, obs, rews, actions):
        """print('\nobs: ', obs)
        print('rewards: ', rews)
        print('actions: ', actions)"""

        if self.feature_names is None:
            total_features = obs[next(iter(obs))][0].shape[0]
            self.feature_names = self._generate_feature_names(total_features)

        for env_index in range(self.n_envs):
            dict_row = {}
            if self.dict_space:
                for agent_index, agent_name in enumerate(obs.keys()):
                    current_obs = obs[agent_name][env_index].cpu().numpy()
                    current_reward = rews[agent_name][env_index].cpu().numpy()
                    agent_action = actions[agent_name][env_index].cpu().numpy()

                    if self.mdp:
                        for n, value in enumerate(current_obs):
                            dict_row[f'{agent_name}_{self.feature_names[n]}'] = value

                        dict_row[f'{agent_name}_reward'] = current_reward.item()
                    else:
                        obs_diff = current_obs - self.last_obs[agent_name][env_index]
                        reward_diff = current_reward - self.last_rewards[agent_name][env_index]

                        for n, value in enumerate(obs_diff):
                            dict_row[f'{agent_name}_{self.feature_names[n]}'] = value

                        dict_row[f'{agent_name}_reward'] = reward_diff.item()

                        # Update last observations and rewards
                        self.last_obs[agent_name][env_index] = current_obs
                        self.last_rewards[agent_name][env_index] = current_reward

                        for n, value in enumerate(agent_action):
                            dict_row[f'{agent_name}_action_{n}'] = value
            else:
                for agent_index in range(len(obs)):
                    agent_name = f'agent_{agent_index}'

                    current_obs = obs[agent_name][env_index].cpu().numpy()
                    current_reward = rews[agent_name][env_index].cpu().numpy()
                    agent_action = actions[agent_name][env_index].cpu().numpy()

                    if self.mdp:
                        for n, value in enumerate(current_obs):
                            dict_row[f'{agent_name}_{self.feature_names[n]}'] = value

                        dict_row[f'{agent_name}_reward'] = current_reward.item()
                    else:
                        obs_diff = current_obs - self.last_obs[agent_name][env_index]
                        reward_diff = current_reward - self.last_rewards[agent_name][env_index]

                        for n, value in enumerate(obs_diff):
                            dict_row[f'{agent_name}_{self.feature_names[n]}'] = value

                        dict_row[f'{agent_name}_reward'] = reward_diff.item()

                        # Update last observations and rewards
                        self.last_obs[agent_name][env_index] = current_obs
                        self.last_rewards[agent_name][env_index] = current_reward

                    for n, value in enumerate(agent_action):
                        dict_row[f'{agent_name}_action_{n}'] = value

            self.list_story.append(dict_row)

    def run_experiment(self) -> Tuple[pd.DataFrame, Dict, str]:
        frame_list = [] if self.render and self.save_render else None

        init_time = time.time()

        obs = self.env.reset()

        for _ in tqdm(range(self.n_steps), desc='Running experiments...'):

            actions = self._select_actions(obs)
            obs, rews, dones, info = self.env.step(actions)

            self._store_data(obs, rews, actions)

            if self.render:
                frame = self.env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=self.render,
                )
                if self.save_render:
                    frame_list.append(frame)

        total_time = round((time.time() - init_time) / 60, 2)
        # print(f"{self.scenario_name}: {total_time} minutes for {self.n_envs} parallel environments")

        self.df_story = pd.DataFrame(self.list_story)

        if self.render and self.save_render:
            save_video(self.scenario_name, frame_list, fps=1 / self.env.scenario.world.dt)

        self._store_results()

        return self.df_story, self.dict_info, self.dir_save

    def _store_results(self):
        add = 'mdp' if self.mdp else 'pomdp'
        add += '_continuous_actions' if self.continuous_actions else '_discrete_actions'
        save_file_incrementally(self.df_story, self.dir_save, prefix=f'df_{self.scenario_name}_{add}_', extension='pkl')
        save_json_incrementally(self.dict_info, self.dir_save, prefix=f'info_{self.scenario_name}_{add}_')

    def return_df_story(self) -> pd.DataFrame:
        return self.df_story

    def return_dict_info(self) -> Dict:
        return self.dict_info


def main(scenario_name: str):
    experiment = VMASExperiment(scenario_name)
    experiment.run_experiment()


if __name__ == "__main__":
    main('flocking')
