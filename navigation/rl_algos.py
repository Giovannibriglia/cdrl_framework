from typing import Dict
import random
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque, defaultdict
from torch import tensor, optim, Tensor
from vmas.simulator.environment import Environment
from navigation.causality_algos import CausalInferenceForRL, CausalDiscovery
from navigation.utils import detach_dict, exploration_action, _state_to_tuple, get_rl_knowledge
from path_repo import GLOBAL_PATH_REPO

EXPLORATION_GAME_PERCENT = 0.7


class RandomAgentVMAS:
    def __init__(self, env, device, seed: int = 42, agent_id: int = 0, algo_config: Dict = None):
        self.env = env
        self.name = 'random'
        self.device = device
        self.agent_id = agent_id
        self.save_df = algo_config.get('save_df', False)
        self.dict_for_storing = None
        self.continuous_actions = False

        np.random.seed(seed)
        random.seed(42)

    def choose_action(self, obs: Dict = None):
        action = torch.randint(
            low=0,
            high=9,
            size=(1,),
            device=self.device,
        )
        return action

    def update(self, state: Tensor, action: Tensor, reward: Tensor, next_state: Tensor):
        if self.save_df:
            if self.dict_for_storing is None:
                self._initialize_dict(state)

            action = action if self.continuous_actions else int(action)
            reward = reward.item() if isinstance(reward, torch.Tensor) else reward
            self._update_dict(state, reward, action, next_state)

    def reset_RL_knowledge(self):
        pass

    def _initialize_dict(self, observation):
        self.features = []
        num_sensors = len(observation) - 6  # Subtracting 6 for PX, PY, VX, VY, DX, DY

        self.obs_features = [f"agent_{self.agent_id}_{feature}" for feature in
                             ['PX', 'PY', 'VX', 'VY', 'DX', 'DY'] + [f'sensor{N}' for N in range(num_sensors)]]
        self.features += self.obs_features

        self.features.append(f"agent_{self.agent_id}_reward")
        self.features.append(f"agent_{self.agent_id}_action")
        self.next_obs_features = [f"agent_{self.agent_id}_next_{feature}" for feature in
                                  ['PX', 'PY', 'VX', 'VY', 'DX', 'DY'] + [f'sensor{N}' for N in range(num_sensors)]]
        self.features += self.next_obs_features
        self.dict_for_storing = {column: [] for column in self.features}

    def _update_dict(self, observation, reward, action, next_observation):
        agent_obs = observation.cpu().numpy()
        agent_reward = reward
        agent_action = action
        agent_next_obs = next_observation.cpu().numpy()

        if self.obs_features is not None:
            for i, feature in enumerate(self.obs_features):
                self.dict_for_storing[feature].append(agent_obs[i])
        self.dict_for_storing[f"agent_{self.agent_id}_reward"].append(agent_reward)
        self.dict_for_storing[f"agent_{self.agent_id}_action"].append(agent_action)
        if self.next_obs_features is not None:
            for i, feature in enumerate(self.next_obs_features):
                self.dict_for_storing[feature].append(agent_next_obs[i])

        # print(len(self.dict_for_causality[next(iter(self.dict_for_causality))]))

    def return_df(self):
        dict_detached = detach_dict(self.dict_for_storing)
        df = pd.DataFrame(dict_detached)
        return df

    def return_RL_knowledge(self):
        return None


class Causality:
    def __init__(self, env: Environment, causality_config: Dict = None, agent_id: int = 0,
                 continuous_actions: bool = False, seed: int = 42):

        self.env = env
        self.action_space_size = env.action_space[f'agent_{agent_id}'].n
        self.agent_id = agent_id
        self.scenario = 'navigation' + f'_agent_{self.agent_id}'
        self.continuous_actions = continuous_actions
        self._setup(causality_config)

        np.random.seed(seed)
        random.seed(42)

    def _setup(self, causality_config):
        self.if_next_obs_causality = False

        self.steps_for_causality_update = int(causality_config.get('batch_timestep_update', 1000))
        self.online_cd = causality_config.get('online_cd', True)
        self.online_ci = causality_config.get('online_ci', True)
        df_causality = causality_config.get('df_for_causality', None)
        causal_graph = causality_config.get('causal_graph', None)
        causal_table = causality_config.get('causal_table', None)

        self.features = None
        self.obs_features = None
        self.next_obs_features = None
        self.dict_for_causality = None

        if self.online_cd and self.online_ci:  # online
            self.cd = None
            self.ci = None
        else:  # offline
            if causal_table is None:
                if df_causality is None and causal_graph is None:
                    raise ValueError(
                        'if causal offline settings, you must provide df and causal graph, otherwise the causal table')
                else:
                    self.cd = None
                    self.ci = CausalInferenceForRL(df_causality, causal_graph)
            else:
                self.ci = CausalInferenceForRL(causal_table=causal_table)

    def update(self, obs: Tensor = None, action: float = None, reward: float = None,
               next_obs: Tensor = None):

        if self.online_cd:  # online
            if self.dict_for_causality is not None:
                if obs is not None and reward is not None and action is not None and next_obs is not None:
                    action = action if self.continuous_actions else int(action)
                    reward = reward.item() if isinstance(reward, torch.Tensor) else reward
                    self._update_dict(obs, reward, action, next_obs)
            else:
                self._initialize_dict(obs)

            if len(self.dict_for_causality[next(iter(self.dict_for_causality))]) > self.steps_for_causality_update:
                print(f'cd agent {self.agent_id}')
                dict_detached = detach_dict(self.dict_for_causality)
                df_causality = pd.DataFrame(dict_detached)

                if self.cd is None:
                    self.cd = CausalDiscovery(df=df_causality,
                                              dir_name=f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge',
                                              env_name=self.scenario)
                else:
                    self.cd.add_data(df_causality)

                self.cd.training(cd_algo='pc')

                causal_graph = self.cd.return_causal_graph()
                df_for_ci = self.cd.return_df()

                if self.ci is None:
                    self.ci = CausalInferenceForRL(df_for_ci, causal_graph,
                                                   dir_name=f'{GLOBAL_PATH_REPO}/navigation/causal_knowledge',
                                                   env_name=self.scenario)
                else:
                    self.ci.add_data(df_for_ci, causal_graph)

                self._initialize_dict(obs)

    def _initialize_dict(self, observation):
        self.features = []
        num_sensors = len(observation) - 6  # Subtracting 6 for PX, PY, VX, VY, DX, DY

        self.obs_features = [f"agent_{self.agent_id}_{feature}" for feature in
                             ['PX', 'PY', 'VX', 'VY', 'DX', 'DY'] + [f'sensor{N}' for N in range(num_sensors)]]
        self.features += self.obs_features
        self.features.append(f"agent_{self.agent_id}_reward")
        self.features.append(f"agent_{self.agent_id}_action")

        if self.if_next_obs_causality:
            self.next_obs_features = [f"agent_{self.agent_id}_next_{feature}" for feature in
                                      ['PX', 'PY', 'VX', 'VY', 'DX', 'DY'] + [f'sensor{N}' for N in range(num_sensors)]]
            self.features += self.next_obs_features

        self.dict_for_causality = {column: [] for column in self.features}

    def _update_dict(self, observation, reward, action, next_observation):
        agent_obs = observation.cpu().numpy()
        agent_reward = reward
        agent_action = action
        agent_next_obs = next_observation.cpu().numpy()

        if self.obs_features is not None:
            for i, feature in enumerate(self.obs_features):
                self.dict_for_causality[feature].append(agent_obs[i])

        self.dict_for_causality[f"agent_{self.agent_id}_reward"].append(agent_reward)
        self.dict_for_causality[f"agent_{self.agent_id}_action"].append(agent_action)

        if self.if_next_obs_causality and self.next_obs_features is not None:
            for i, feature in enumerate(self.next_obs_features):
                self.dict_for_causality[feature].append(agent_next_obs[i])

    def action_filter(self, obs: Tensor) -> Dict:
        if self.ci is None:
            return {key: 1 / self.action_space_size for key in range(self.action_space_size)}
        else:
            obs_dict = {key: obs[n] for n, key in enumerate(self.obs_features)}
            action_reward_values = self.ci.get_actions_rewards_values(obs_dict, self.online_ci)
            # print('Reward-action-values causal inference: ', action_reward_values)
            return action_reward_values


class QLearningAgent:
    def __init__(self, env: Environment, device: str = 'cpu', n_steps: int = 100000, agent_id: int = 0,
                 algo_config: Dict = None, causality_config: Dict = None, seed: int = 42, scenario: str = 'navigation'):

        name = 'qlearning'
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.env = env
        self.device = device
        self.agent_id = agent_id
        self.action_space_size = 9  # env.action_space[f'agent_{self.agent_id}'].n_bins
        self.n_steps = n_steps

        self.continuous_actions = False  # self.env.continuous_actions

        self._setup_algo(algo_config)

        if causality_config is None:
            self.name = f'{name}'
            self.if_causality = False
        else:
            self.name = f'causal_{name}'
            self.online_causality = causality_config.get('online_ci', True)
            self.name += '_online' if self.online_causality else '_offline'
            self.if_causality = True
            self._setup_causality(causality_config)

        self.scenario = scenario

    def _setup_algo(self, algo_config):

        self.start_epsilon = float(algo_config.get('epsilon_start', 1.0))
        self.learning_rate = float(algo_config.get('learning_rate', 0.0001))
        self.discount_factor = float(algo_config.get('discount_factor', 0.98))
        self.epsilon = self.start_epsilon
        self.min_epsilon = float(algo_config.get('min_epsilon', 0.05))
        self.epsilon_decay = 1 - (-np.log(self.min_epsilon) / (EXPLORATION_GAME_PERCENT * self.n_steps))
        filepath_transfer_learning = algo_config.get('transfer_learning', None)

        self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))

        if filepath_transfer_learning is not None:
            self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))

            rl_knowledge = get_rl_knowledge(filepath_transfer_learning, self.agent_id)

            for state_action_dict in rl_knowledge[0]:
                for state_action, q_values in state_action_dict.items():
                    # Convert the string representation of the tuple back to an actual tuple
                    state_action_tuple = eval(state_action)
                    self.q_table[state_action_tuple] = q_values

    def _setup_causality(self, causality_config):
        self.causality_obj = Causality(self.env, causality_config, self.agent_id)

    def choose_action(self, state: Tensor):
        if self.if_causality:
            action_reward_values = self.causality_obj.action_filter(state)
        else:
            action_reward_values = {key: 1 / self.action_space_size for key in range(self.action_space_size)}

        if random.uniform(0.0, 1.0) < self.epsilon:
            random_action = exploration_action(action_reward_values)
            # print('exploration', reward_action_values, random_action)
            return torch.tensor([random_action], device=self.device)
        else:
            # TODO: define "best" actions
            """if len(best_actions) > 0:
                mask = np.zeros_like(state_action_values, dtype=bool)
                mask[best_actions] = True
                masked_state_action_values = np.where(mask, state_action_values, -np.inf)

                chosen_action = np.argmax(masked_state_action_values)
            else:"""

            state_tuple = _state_to_tuple(state)
            state_action_values = self.q_table[state_tuple]
            chosen_action = np.argmax(state_action_values)
            # print('exploitation', reward_action_values, chosen_action)
            return torch.tensor([chosen_action], device=self.device)

    def update(self, obs: Tensor = None, action: float = None, reward: float = None, next_obs: Tensor = None):
        if self.if_causality:
            self.causality_obj.update(obs, action, reward, next_obs)

        self._update_rl_algo(obs, action, reward, next_obs)

    def _update_rl_algo(self, obs: Tensor = None, action: float = None, reward: float = None, next_obs: Tensor = None):
        action = int(action)
        state_tuple = _state_to_tuple(obs)
        next_state_tuple = _state_to_tuple(next_obs)

        best_next_action = np.argmax(self.q_table[next_state_tuple])
        td_target = reward + self.discount_factor * self.q_table[next_state_tuple][best_next_action]
        td_error = td_target - self.q_table[state_tuple][action]
        self.q_table[state_tuple][action] += self.learning_rate * td_error

        self._decay_epsilon()

    def _decay_epsilon(self):
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, round(self.epsilon * self.epsilon_decay, 5))

    def reset_RL_knowledge(self):
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))
        self.epsilon = self.start_epsilon

    def return_RL_knowledge(self):
        # Convert the Q-table to a JSON serializable format
        q_table_json_serializable = {
            str(k): v.tolist() for k, v in self.q_table.items()
        }
        return q_table_json_serializable


# Define a namedtuple for Experience Replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

HIDDEN_LAYERS_DQN = 128


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_LAYERS_DQN)  # Adjust input_size to match state dimension
        self.fc2 = nn.Linear(HIDDEN_LAYERS_DQN, HIDDEN_LAYERS_DQN)
        self.fc3 = nn.Linear(HIDDEN_LAYERS_DQN, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # Input x should be (batch_size, input_size)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, env: Environment, device: str = 'cpu', n_steps: int = 100000, agent_id: int = 0,
                 algo_config: Dict = None, causality_config: Dict = None, seed: int = 42, scenario: str = 'navigation'):
        name = 'dqn'
        self.state_space_size = 18

        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.env = env
        self.device = device
        self.agent_id = agent_id
        self.action_space_size = 9  # env.action_space[f'agent_{self.agent_id}']
        self.n_steps = n_steps

        self.continuous_actions = False  # self.env.continuous_actions

        self._setup_algo(algo_config)

        if causality_config is None:
            self.name = f'{name}'
            self.if_causality = False
        else:
            self.name = f'causal_{name}'
            self.online_causality = causality_config.get('online_ci', True)
            self.name += '_online' if self.online_causality else '_offline'
            self.if_causality = True
            self._setup_causality(causality_config)

        self.scenario = scenario

    def _setup_algo(self, algo_config):
        self.learning_rate = float(algo_config.get('learning_rate', 0.0001))
        self.discount_factor = float(algo_config.get('discount_factor', 0.98))
        self.start_epsilon = float(algo_config.get('start_epsilon', 1.0))
        self.epsilon = self.start_epsilon
        self.min_epsilon = float(algo_config.get('min_epsilon', 0.01))
        self.epsilon_decay = 1 - (-np.log(self.min_epsilon) / (EXPLORATION_GAME_PERCENT * self.n_steps))

        filepath_transfer_learning = algo_config.get('transfer_learning', None)
        self.q_network = QNetwork(self.state_space_size, self.action_space_size).to(self.device)
        if filepath_transfer_learning is not None:
            rl_knowledge = get_rl_knowledge(filepath_transfer_learning, self.agent_id)
            state_dict = self._rl_knowledge_to_state_dict(rl_knowledge[0][0])
            # print(state_dict)
            self.q_network.load_state_dict(state_dict)
            # self.q_network.load_state_dict(self._convert_serialized_state_dict(rl_knowledge))
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # Experience Replay
        self.batch_size = int(algo_config.get('batch_size', 64))
        self.len_replay_memory_size = int(algo_config.get('replay_memory_size', 10000))
        self.replay_memory = deque(maxlen=self.len_replay_memory_size)

    def _rl_knowledge_to_state_dict(self, rl_knowledge):
        # Initialize an empty state dictionary
        state_dict = {k: torch.tensor(v) for k, v in rl_knowledge.items()}

        # Assuming rl_knowledge is a list of lists of tensors
        """state_dict['fc1.weight'] = rl_knowledge['fc1.weight']
        state_dict['fc1.bias'] = torch.tensor(rl_knowledge[1])
        state_dict['fc2.weight'] = torch.tensor(rl_knowledge[2])
        state_dict['fc2.bias'] = torch.tensor(rl_knowledge[3])"""

        return state_dict

    def _setup_causality(self, causality_config):
        self.causality_obj = Causality(self.env, causality_config, self.agent_id)

    def choose_action(self, state: Tensor):
        if self.if_causality:
            reward_action_values = self.causality_obj.action_filter(state)
        else:
            reward_action_values = {key: 1 / self.action_space_size for key in range(self.action_space_size)}

        if random.uniform(0.0, 1.0) < self.epsilon:
            random_action = exploration_action(reward_action_values)
            return torch.tensor([random_action], device=self.device)
        else:
            with torch.no_grad():
                state = state.to(torch.float)
                q_values = self.q_network(state)
                action = torch.argmax(q_values).item()

        action = torch.tensor([action], device=self.device)

        return action

    def update(self, obs: Tensor = None, action: float = None, reward: float = None, next_obs: Tensor = None):

        if self.if_causality:
            self.causality_obj.update(obs, action, reward, next_obs)

        self._update_rl_algo(obs, action, reward, next_obs)

    def _update_rl_algo(self, obs, action, reward, next_obs):
        # Store transition in replay memory
        self.replay_memory.append(Transition(obs, action, next_obs, reward))

        # Sample a random batch from replay memory
        if len(self.replay_memory) < self.batch_size:
            return

        transitions = random.sample(self.replay_memory, self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        non_final_next_states = torch.stack(
            [s.clone().to(self.device).detach() for s in batch.next_state if s is not None])
        non_final_next_states = non_final_next_states.to(torch.float)

        state_batch = torch.stack([s.clone().to(self.device).detach() for s in batch.state])
        state_batch = state_batch.to(torch.float)

        action_batch = torch.tensor(batch.action, dtype=torch.int64, device=self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)

        # Compute Q-values for current and next states
        state_action_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.q_network(non_final_next_states).max(1)[0].detach()

        # Compute expected Q-values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self._decay_epsilon()

    def _decay_epsilon(self):
        # Decay epsilon
        self.epsilon = max(self.min_epsilon, round(self.epsilon * self.epsilon_decay, 5))

    def return_RL_knowledge(self):
        serialized_state_dict = {k: v.tolist() for k, v in self.q_network.state_dict().items()}
        return serialized_state_dict

    def load(self, serialized_state_dict):
        # Convert lists back to tensors
        state_dict = {k: torch.tensor(v) for k, v in serialized_state_dict.items()}
        self.q_network.load_state_dict(state_dict)

    def reset_RL_knowledge(self):
        # Q-Network
        self.q_network = QNetwork(self.state_space_size, self.action_space_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.replay_memory = deque(maxlen=self.len_replay_memory_size)
