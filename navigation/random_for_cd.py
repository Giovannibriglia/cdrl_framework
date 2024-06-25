import os
import json
from typing import List
import pandas as pd
from vmas import make_env
from path_repo import GLOBAL_PATH_REPO
from torch import Tensor
import time
from navigation.algos import RandomAgentVMAS, CausalAgentVMAS, QLearningAgent, DQNAgent
import torch
from vmas.simulator.environment import Wrapper
from tqdm.auto import tqdm


class VMASTrainer:
    def __init__(self, n_training_episodes: int = 1, n_environments: int = 1, n_agents: int = 4,
                 algo_name: str = 'only_causal', env_wrapper: Wrapper | None = Wrapper.GYM, rendering: bool = False,
                 x_semidim: float = 0.5, y_semidim: float = 0.5, max_steps_env: int = None, observability: str = 'mdp',
                 seed: int = 42):
        self.n_training_episodes = n_training_episodes
        self.n_environments = n_environments
        self.n_agents = n_agents
        self.env_wrapper = env_wrapper
        self.if_render = rendering
        self.x_semidim = x_semidim
        self.y_semidim = y_semidim
        self.max_steps_env = max_steps_env
        self.observability = observability
        self.seed = seed

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gym_wrapper = Wrapper.GYM

        if self.env_wrapper == self.gym_wrapper:
            self.n_environments = 1

        self.env = self.config_env()
        self.algos = self.config_algos(algo_name)

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
                'causal_graph': None,
                'df_causality': None,
                'causal_table': None
            } for i in range(self.n_agents)
        }

    def config_env(self):
        print('Device: ', self.device)
        env = make_env(
            scenario='navigation',
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
            max_steps=self.max_steps_env
        )
        return env

    def _config_algo(self, algo: str, agent_id: int):
        if algo == 'completely_causal':
            return CausalAgentVMAS(self.env, {}, self.device, agent_id, n_steps=int(self.max_steps_env/self.max_steps_env))
        elif algo == 'qlearning':
            algo_config = {'learning_rate': 0.0001, 'discount_factor': 0.98, 'epsilon_start': 1.0, 'min_epsilon': 0.05}
            return QLearningAgent(self.env, self.device, int(self.max_steps_env/self.max_steps_env), agent_id, algo_config)
        elif algo == 'dqn':
            return DQNAgent(self.env, self.device)
        elif algo == 'causal_qlearning':
            algo_config = {'learning_rate': 0.0001, 'discount_factor': 0.98, 'epsilon_start': 1.0, 'min_epsilon': 0.05}
            causality_config = None
            return QLearningAgent(self.env, self.device, int(self.max_steps_env/self.max_steps_env), agent_id, algo_config, causality_config)
        else:
            return RandomAgentVMAS(self.env, self.device, agent_id=agent_id, save_df=True)

    def config_algos(self, algo_name: str):
        return [self._config_algo(algo_name, i) for i in range(self.n_agents)]

    def train(self):
        if self.env_wrapper is None:
            self.native_train()
        elif self.env_wrapper == self.gym_wrapper:
            self.gym_train()

        self.save_metrics()

        if self.algos[0].name == 'random':
            if self.algos[0].save_df:
                dir_save = f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge/offline'
                df_final = pd.DataFrame()
                for algo in self.algos:
                    df_new = algo.return_df()
                    df_final = pd.concat([df_final, df_new], axis=1).reset_index(drop=True)
                df_final.to_pickle(f'{dir_save}/df_random_{self.observability}_{len(df_final)}.pkl')
                # df_final.to_excel(f'{GLOBAL_PATH_REPO}/navigation/df_random_{self.observability}_{len(df_final)}.xlsx')

    def gym_train(self):
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
                next_observations, rewards, done, info = self.trainer_step(actions, observations, episode)
                if self.if_render:
                    self.env.render()
                for i in range(self.n_agents):
                    self.algos[i].update(observations[f'agent_{i}'], actions[i], rewards[f'agent_{i}'],
                                         next_observations[f'agent_{i}'])

                    self.update_metrics(f'agent_{i}', episode, steps, 0, rewards[f'agent_{i}'],
                                        actions[i], initial_time)

                steps += 1
                observations = next_observations
            for i in range(self.n_agents):
                rl_knowledge = self.algos[i].return_RL_knowledge()
                self.dict_metrics[f'agent_{i}']['rl_knowledge'][episode][0] = rl_knowledge
                self.algos[i].reset_RL_knowledge()

    def native_train(self):
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

                next_observations, rewards, dones, info = self.trainer_step(actions, observations, episode)
                if self.if_render:
                    self.env.render()

                for i in range(self.n_agents):
                    if self.n_environments == 1:
                        self.algos[i].update(observations[f'agent_{i}'][0], actions[i][0],
                                             rewards[f'agent_{i}'][0],
                                             next_observations[f'agent_{i}'][0])

                        self.update_metrics(f'agent_{i}', episode, steps, 0, rewards[f'agent_{i}'][0], actions[i][0],
                                            initial_time)

                    else:
                        for env_n in range(self.n_environments):
                            action_scalar = actions[i][env_n].item()
                            self.algos[i].update(observations[f'agent_{i}'][env_n], action_scalar,
                                                 rewards[f'agent_{i}'][env_n],
                                                 next_observations[f'agent_{i}'][env_n])

                            self.update_metrics(f'agent_{i}', episode, steps, env_n, rewards[f'agent_{i}'][env_n],
                                                actions[i][env_n], initial_time)
                steps += 1
                observations = next_observations
                if steps % int(self.max_steps_env/20) == 0:
                    print(f'{steps}/{self.max_steps_env}')

            for i in range(self.n_agents):
                rl_knowledge = self.algos[i].return_RL_knowledge()
                for env_n in range(self.n_environments):
                    self.dict_metrics[f'agent_{i}']['rl_knowledge'][episode][env_n] = rl_knowledge
                self.algos[i].reset_RL_knowledge()

    def trainer_step(self, actions: List, observations: Tensor, episode: int):
        next_observations, rewards, done, info = self.env.step(actions)

        if self.observability == 'mdp':
            return next_observations, rewards, done, info
        else:  # pomdp
            if self.dict_for_pomdp is not None:
                if self.dict_for_pomdp['state'] is None:  # initialization
                    self.dict_for_pomdp['state'] = observations
                    self.dict_for_pomdp['reward'] = rewards
                    self.dict_for_pomdp['next_state'] = next_observations

            observations_relative = {agent: observations[agent] - self.dict_for_pomdp['state'][agent] for agent in
                                     observations}
            next_observations_relative = {agent: next_observations[agent] - self.dict_for_pomdp['next_state'][agent] for
                                          agent in next_observations}
            rewards_relative = {agent: rewards[agent] - self.dict_for_pomdp['reward'][agent] for agent in rewards}

            self.dict_for_pomdp['state'] = observations_relative
            self.dict_for_pomdp['reward'] = rewards_relative
            self.dict_for_pomdp['next_state'] = next_observations_relative

            return next_observations_relative, rewards_relative, done, info

    def update_metrics(self, agent_key: str, episode_idx: int, step_idx: int, env_idx: int, reward_value: float = None,
                       action_value: float = None, initial_time_value: float = None):
        if reward_value is not None:
            self.dict_metrics[agent_key]['rewards'][episode_idx][step_idx][env_idx].append(float(reward_value))
        if action_value is not None:
            self.dict_metrics[agent_key]['actions'][episode_idx][step_idx][env_idx].append(int(action_value))
        if initial_time_value is not None:
            self.dict_metrics[agent_key]['time'][episode_idx][step_idx][env_idx].append(
                time.time() - initial_time_value)

    def save_metrics(self):
        dir_results = f'{GLOBAL_PATH_REPO}/navigation/results'
        os.makedirs(dir_results, exist_ok=True)

        with open(f'{dir_results}/{self.algos[0].name}_{self.observability}.json', 'w') as file:
            json.dump(self.dict_metrics, file)


def run_simulations(algorithm):
    n_episodes = 1
    n_agents = 4
    max_steps_env = 100000
    n_environments = 10
    observabilities = ['mdp', 'pomdp']
    for observability in observabilities:
        print(f'*** {algorithm} - {observability} ***')
        trainer = VMASTrainer(env_wrapper=None, n_training_episodes=n_episodes, rendering=False, n_agents=n_agents,
                              n_environments=n_environments,
                              algo_name=algorithm, max_steps_env=max_steps_env, observability=observability)
        trainer.train()


if __name__ == '__main__':
    models = ['random']
    for model in models:
        run_simulations(model)
