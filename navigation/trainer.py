import os
import json
from typing import List, Dict
import pandas as pd
import yaml
from vmas import make_env
from path_repo import GLOBAL_PATH_REPO
from torch import Tensor
import time
from navigation.algos_new import RandomAgentVMAS, QLearningAgent, DQNAgent
import torch
from vmas.simulator.environment import Wrapper
from tqdm.auto import tqdm


class VMASTrainer:
    def __init__(self, simulation_config: Dict = None, algo_config: Dict = None, causality_config: Dict = None,
                 env_wrapper: Wrapper | None = None, rendering: bool = False, seed: int = 42):

        self.observability = simulation_config.get('observability', 'mdp')
        self.env_wrapper = env_wrapper
        self.if_render = rendering
        self.seed = seed

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gym_wrapper = Wrapper.GYM

        if self.env_wrapper == self.gym_wrapper:
            self.n_environments = 1
        else:
            self.n_environments = int(simulation_config.get('n_environments', 10))

        self.env = self._config_env(simulation_config)
        self.algos = self._config_algos(algo_config, causality_config)

        if self.observability == 'pomdp':
            self.dict_for_pomdp = {'state': None, 'reward': None, 'next_state': None}
        else:
            self.dict_for_pomdp = None

        self._define_metrics()

    def _define_metrics(self):
        self.dict_metrics = {
            f'agent_{i}': {
                'env': {
                    'task': 'navigation',
                    'n_agents': self.n_agents,
                    'n_training_episodes': self.n_training_episodes,
                    'env_max_steps': self.max_steps_env,
                    'n_envs': self.n_environments,
                    'x_semidim': self.x_semidim,
                    'y_semidim': self.y_semidim
                },
                'rewards': [[[[] for _ in range(self.n_environments)] for _ in range(self.max_steps_env)] for _ in
                            range(self.n_training_episodes)],
                'actions': [[[[] for _ in range(self.n_environments)] for _ in range(self.max_steps_env)] for _ in
                            range(self.n_training_episodes)],
                'time': [[[[] for _ in range(self.n_environments)] for _ in range(self.max_steps_env)] for _ in
                         range(self.n_training_episodes)],
                'rl_knowledge': [[[] for _ in range(self.n_environments)] for _ in range(self.n_training_episodes)],
                'n_collisions': [[[[] for _ in range(self.n_environments)] for _ in range(self.max_steps_env)] for _ in
                         range(self.n_training_episodes)],
                'causal_graph': None,
                'df_causality': None,
                'causal_table': None
            } for i in range(self.n_agents)
        }

    def _config_env(self, simulation_config: Dict = None):
        self.n_training_episodes = int(simulation_config.get('n_episodes', 10))
        self.n_agents = int(simulation_config.get('n_agents', 10))
        self.scenario = str(simulation_config.get('task', 'navigation'))
        self.max_steps_env = int(simulation_config.get('max_steps_env', 20000))
        self.x_semidim = float(simulation_config.get('x_semidim', 0.5))
        self.y_semidim = float(simulation_config.get('y_semidim', 0.5))
        self.n_sensors = int(simulation_config.get('n_sensors', 12))

        print('Device: ', self.device)
        env = make_env(
            scenario=self.scenario,
            num_envs=self.n_environments,
            device=self.device,
            continuous_actions=False,
            dict_spaces=True,
            wrapper=self.env_wrapper,
            seed=self.seed,
            shared_rew=False,
            n_agents=self.n_agents,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            max_steps=self.max_steps_env,
            n_sensors=self.n_sensors
        )
        return env

    def _config_algo(self, agent_id: int, algo_config: Dict = None, causality_config: Dict = None):
        algo_name = str(algo_config.get('name', 'random'))

        if algo_name == 'qlearning':
            return QLearningAgent(self.env, self.device, int(self.max_steps_env / self.n_environments), agent_id,
                                  algo_config, causality_config)
        elif algo_name == 'dqn':
            return DQNAgent(self.env, self.device, int(self.max_steps_env / self.n_environments), agent_id,
                            algo_config, causality_config)
        else:
            return RandomAgentVMAS(self.env, self.device, agent_id=agent_id, algo_config=algo_config)

    def _config_algos(self, algo_config: Dict = None, causality_config: Dict = None):
        return [self._config_algo(i, algo_config, causality_config) for i in range(self.n_agents)]

    def train(self):
        if self.env_wrapper is None:
            self._native_train()
        elif self.env_wrapper == self.gym_wrapper:
            self._gym_train()

        self._save_metrics()

    def _gym_train(self):
        # training process
        pbar = tqdm(range(self.n_training_episodes), desc='Training...')
        for episode in pbar:
            observations = self.env.reset()
            done = False

            steps = 0
            initial_time = time.time()
            while not done:
                actions = [self.algos[i].choose_action(observations[f'agent_{i}']) for i in range(self.n_agents)]
                actions = [tensor.item() for tensor in actions]
                next_observations, rewards, done, info = self._trainer_step(actions, observations, episode)

                if self.if_render:
                    self.env.render()
                for i in range(self.n_agents):
                    self.algos[i].update(observations[f'agent_{i}'], actions[i], rewards[f'agent_{i}'],
                                         next_observations[f'agent_{i}'])

                    self._update_metrics(f'agent_{i}', episode, steps, 0, rewards[f'agent_{i}'],
                                         actions[i], initial_time, info[f'agent_{i}']['agent_collisions'])

                steps += 1
                observations = next_observations

            for i in range(self.n_agents):
                rl_knowledge = self.algos[i].return_RL_knowledge()
                self.dict_metrics[f'agent_{i}']['rl_knowledge'][episode][0] = rl_knowledge
                self.algos[i].reset_RL_knowledge()

            self._save_metrics()

    def _native_train(self):
        pbar = tqdm(range(self.n_training_episodes), desc='Training...')

        for episode in pbar:
            observations = self.env.reset()
            dones = torch.tensor([False * self.n_agents], device=self.device)

            steps = 0
            initial_time = time.time()
            while not any(dones):
                if self.n_environments == 1:
                    actions = [self.algos[i].choose_action(observations[f'agent_{i}'][0]) for i in
                               range(self.n_agents)]
                else:
                    actions = [
                        [self.algos[i].choose_action(observations[f'agent_{i}'][env_n]) for env_n in
                         range(self.n_environments)]
                        for i in range(self.n_agents)]

                next_observations, rewards, dones, info = self._trainer_step(actions, observations, episode)
                if self.if_render:
                    self.env.render()

                for i in range(self.n_agents):
                    if self.n_environments == 1:
                        self.algos[i].update(observations[f'agent_{i}'][0], actions[i][0],
                                             rewards[f'agent_{i}'][0],
                                             next_observations[f'agent_{i}'][0])

                        self._update_metrics(f'agent_{i}', episode, steps, 0, rewards[f'agent_{i}'][0], actions[i][0],
                                             initial_time, info[f'agent_{i}']['agent_collisions'][0])

                    else:
                        for env_n in range(self.n_environments):
                            action_scalar = actions[i][env_n].item()
                            self.algos[i].update(observations[f'agent_{i}'][env_n], action_scalar,
                                                 rewards[f'agent_{i}'][env_n],
                                                 next_observations[f'agent_{i}'][env_n])
                            self._update_metrics(f'agent_{i}', episode, steps, env_n, rewards[f'agent_{i}'][env_n],
                                                 actions[i][env_n], initial_time, info[f'agent_{i}']['agent_collisions'][env_n])
                steps += 1
                observations = next_observations

                if steps % int(self.max_steps_env / 10) == 0:
                    print(f'{steps}/{self.max_steps_env}')

            for i in range(self.n_agents):
                rl_knowledge = self.algos[i].return_RL_knowledge()
                for env_n in range(self.n_environments):
                    self.dict_metrics[f'agent_{i}']['rl_knowledge'][episode][env_n] = rl_knowledge
                self.algos[i].reset_RL_knowledge()

            self._save_metrics()

    def _trainer_step(self, actions: List, observations: Tensor, episode: int):
        next_observations, rewards, done, info = self.env.step(actions)

        if self.observability == 'mdp':
            return next_observations, rewards, done, info
        else:  # pomdp
            if self.dict_for_pomdp is not None:
                if self.dict_for_pomdp['state'] is None:  # initialization
                    self.dict_for_pomdp['state'] = observations
                    self.dict_for_pomdp['reward'] = rewards
                    self.dict_for_pomdp['next_state'] = next_observations

            # Calculate the absolute difference
            observations_relative = {agent: abs(observations[agent]) - abs(self.dict_for_pomdp['state'][agent])
                                     for agent in observations}
            next_observations_relative = {
                agent: abs(next_observations[agent]) - abs(self.dict_for_pomdp['next_state'][agent]) for agent
                in next_observations}
            rewards_relative = {agent: abs(rewards[agent]) - abs(self.dict_for_pomdp['reward'][agent]) for agent
                                in rewards}

            """print('\n obs', observations['agent_0'])
            print('new_obs ', observations_relative['agent_0'])
            print('\n')
            print('rew ', rewards['agent_0'])
            print('new_rew', rewards_relative['agent_0'])"""

            self.dict_for_pomdp['state'] = observations_relative
            self.dict_for_pomdp['reward'] = rewards_relative
            self.dict_for_pomdp['next_state'] = next_observations_relative

            return next_observations_relative, rewards_relative, done, info

    def _update_metrics(self, agent_key: str, episode_idx: int, step_idx: int, env_idx: int, reward_value: float = None,
                        action_value: float = None, initial_time_value: float = None, collision: Tensor = 0):

        self.dict_metrics[agent_key]['rewards'][episode_idx][step_idx][env_idx].append(float(reward_value))

        self.dict_metrics[agent_key]['actions'][episode_idx][step_idx][env_idx].append(int(action_value))

        self.dict_metrics[agent_key]['time'][episode_idx][step_idx][env_idx].append(
            time.time() - initial_time_value)

        if collision.item() != 0:
            collision = 1
        else:
            collision = 0
        self.dict_metrics[agent_key]['n_collisions'][episode_idx][step_idx][env_idx].append(collision)

    def _save_metrics(self):
        dir_results = f'{GLOBAL_PATH_REPO}/navigation/results'
        os.makedirs(dir_results, exist_ok=True)

        with open(f'{dir_results}/{self.algos[0].name}_{self.observability}.json', 'w') as file:
            json.dump(self.dict_metrics, file)

        if self.algos[0].name == 'random':
            if self.algos[0].save_df:
                dir_save = f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge/offline'
                os.makedirs(dir_save, exist_ok=True)
                df_final = pd.DataFrame()
                for algo in self.algos:
                    df_new = algo.return_df()
                    df_final = pd.concat([df_final, df_new], axis=1).reset_index(drop=True)
                df_final.to_pickle(f'{dir_save}/df_random_{self.observability}_{len(df_final)}.pkl')
                # df_final.to_excel(f'{GLOBAL_PATH_REPO}/navigation/df_random_{self.observability}_{len(df_final)}.xlsx')


def run_simulations(path_yaml_config: str, if_rendering: bool = False):
    with open(f'{path_yaml_config}', 'r') as file:
        config = yaml.safe_load(file)

    simulation_config = config.get('simulation_config')
    algo_config = config.get('algo_config')
    causality_config = config.get('causality_config', None)

    task = simulation_config.get('task', 'navigation')
    observability = simulation_config.get('observability', 'mdp')
    algorithm_name = algo_config.get('name', 'random')

    if causality_config is not None:
        algorithm = f'causal_{algorithm_name}'
        algorithm += '_online' if causality_config.get('online_ci', True) else '_offline'

    print(f'*** {task} - {algorithm_name} - {observability} ***')
    trainer = VMASTrainer(simulation_config=simulation_config, algo_config=algo_config,
                          causality_config=causality_config, rendering=if_rendering)
    trainer.train()


if __name__ == '__main__':
    path_file = f'{GLOBAL_PATH_REPO}/config_simulations/causal_qlearning_online.yaml'
    run_simulations(path_file, False)
