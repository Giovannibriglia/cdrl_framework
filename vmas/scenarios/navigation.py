#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import Callable, Dict, List

import numpy as np
import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Entity, Landmark, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = True
        self.n_agents = kwargs.get("n_agents", 4)
        self.collisions = kwargs.get("collisions", True)
        "*************************************************************************************************************"
        self.x_semidim = kwargs.get("x_semidim", None)
        self.y_semidim = kwargs.get("y_semidim", None)
        self.n_sensors = kwargs.get("n_sensors", 12)

        self.n_bins_discretization = kwargs.get('n_bins_discretization', None)
        self.n_sensors_to_consider = kwargs.get('n_sensors_on', None)
        "*************************************************************************************************************"
        self.agents_with_same_goal = kwargs.get("agents_with_same_goal", 1)
        self.split_goals = kwargs.get("split_goals", False)
        self.observe_all_goals = kwargs.get("observe_all_goals", False)

        self.lidar_range = kwargs.get("lidar_range", 0.35)
        self.agent_radius = kwargs.get("agent_radius", 0.1)
        self.comms_range = kwargs.get("comms_range", 0)

        self.shared_rew = kwargs.get("shared_rew", True)
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)
        self.final_reward = kwargs.get("final_reward", 0.01)

        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", -1)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        "*************************************************************************************************************"
        if self.x_semidim is None and self.y_semidim is None:
            self.world_semidim = 1.0
        else:
            self.world_semidim = min(self.x_semidim, self.y_semidim)
        "*************************************************************************************************************"
        self.min_collision_distance = 0.005

        assert 1 <= self.agents_with_same_goal <= self.n_agents
        if self.agents_with_same_goal > 1:
            assert (
                not self.collisions
            ), "If agents share goals they cannot be collidables"
        # agents_with_same_goal == n_agents: all agent same goal
        # agents_with_same_goal = x: the first x agents share the goal
        # agents_with_same_goal = 1: all independent goals
        if self.split_goals:
            assert (
                    self.n_agents % 2 == 0
                    and self.agents_with_same_goal == self.n_agents // 2
            ), "Splitting the goals is allowed when the agents are even and half the team has the same goal"

        "*************************************************************************************************************"
        # Make world
        world = World(batch_dim, device, substeps=2, x_semidim=self.x_semidim, y_semidim=self.y_semidim)
        "*************************************************************************************************************"

        known_colors = [
            (0.22, 0.49, 0.72),
            (1.00, 0.50, 0),
            (0.30, 0.69, 0.29),
            (0.97, 0.51, 0.75),
            (0.60, 0.31, 0.64),
            (0.89, 0.10, 0.11),
            (0.87, 0.87, 0),
        ]
        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )
        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, Agent)

        # Add agents
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )

            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=self.collisions,
                color=color,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                sensors=(
                    [
                        Lidar(
                            world,
                            n_rays=self.n_sensors,
                            max_range=self.lidar_range,
                            entity_filter=entity_filter_agents,
                        ),
                    ]
                    if self.collisions
                    else None
                ),
            )
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)

            # Add goals
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                color=color,
            )
            world.add_landmark(goal)
            agent.goal = goal

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (-self.world_semidim, self.world_semidim),
            (-self.world_semidim, self.world_semidim),
        )

        occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)

        goal_poses = []
        for _ in self.world.agents:
            position = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=self.world,
                min_dist_between_entities=self.min_distance_between_entities,
                x_bounds=(-self.world_semidim, self.world_semidim),
                y_bounds=(-self.world_semidim, self.world_semidim),
            )
            goal_poses.append(position.squeeze(1))
            occupied_positions = torch.cat([occupied_positions, position], dim=1)

        for i, agent in enumerate(self.world.agents):
            if self.split_goals:
                goal_index = int(i // self.agents_with_same_goal)
            else:
                goal_index = 0 if i < self.agents_with_same_goal else i

            agent.goal.set_pos(goal_poses[goal_index], batch_index=env_index)

            if env_index is None:
                agent.pos_shaping = (
                        torch.linalg.vector_norm(
                            agent.state.pos - agent.goal.state.pos,
                            dim=1,
                        )
                        * self.pos_shaping_factor
                )
            else:
                agent.pos_shaping[env_index] = (
                        torch.linalg.vector_norm(
                            agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                        )
                        * self.pos_shaping_factor
                )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for a in self.world.agents:
                self.pos_rew += self.agent_reward(a)
                a.agent_collision_rew[:] = 0

            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1),
                dim=-1,
            )

            self.final_rew[self.all_goal_reached] = self.final_reward

            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        a.agent_collision_rew[
                            distance <= self.min_collision_distance
                            ] += self.agent_collision_penalty
                        b.agent_collision_rew[
                            distance <= self.min_collision_distance
                            ] += self.agent_collision_penalty

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        tot_rewards = pos_reward + self.final_rew + agent.agent_collision_rew

        if self.n_bins_discretization is None:
            print('reward not rescaled')
            return tot_rewards
        else:
            return self._rescale_vector(tot_rewards, 'reward')

    def agent_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        return agent.pos_rew

    def _rescale_value(self, kind: str, value: float | int, n_bins: int, **kwargs):
        def discretize_value(val, all_intervals):
            for i in range(len(all_intervals) - 1):
                if all_intervals[i] <= val < all_intervals[i + 1]:
                    return (all_intervals[i] + all_intervals[i + 1]) / 2
            # Handle the edge cases
            if val < all_intervals[0]:
                return all_intervals[0]
            elif val >= all_intervals[-1]:
                return all_intervals[-1]

        def create_intervals(min_val, max_val, n_intervals, scale='linear'):
            if scale == 'exponential':
                # Generate n_intervals points using exponential scaling
                intervals = np.logspace(0, 1, n_intervals, base=10) - 1
                intervals = intervals / (10 - 1)  # Normalize to range 0-1
                intervals = min_val + (max_val - min_val) * intervals
            elif scale == 'linear':
                intervals = np.linspace(min_val, max_val, n_intervals)
            else:
                raise ValueError("Unsupported scale type. Use 'exponential' or 'linear'.")
            return intervals

        if kind == 'reward':
            max_value = 0.1
            min_value = self.agent_collision_penalty * self.n_agents - max(self.x_semidim, self.y_semidim)
        elif kind == 'distX' or kind == 'distY':
            max_value = self.x_semidim * 2 if kind == 'DX' else self.y_semidim * 2
            min_value = -self.x_semidim * 2 if kind == 'DX' else -self.y_semidim * 2
        elif kind == 'velX' or kind == 'velY':
            max_value = 0.5
            min_value = -0.5
        elif kind == 'posX' or kind == 'posY':
            max_value = self.x_semidim if kind == 'posX' else self.y_semidim
            min_value = -self.x_semidim if kind == 'posX' else -self.y_semidim
        elif kind == 'sensor' and self.n_sensors_to_consider is None:
            max_value = kwargs.get('max_value', 0.35)
            min_value = 0
        else:
            max_value = None
            min_value = None

        if max_value is not None and min_value is not None:
            intervals = create_intervals(min_value, max_value, n_bins, scale='linear')
            rescaled_value = discretize_value(value, intervals)
            return rescaled_value
        else:
            return value

    def _rescale_vector(self, tensor_input: torch.Tensor, kind: str, **kwargs) -> torch.Tensor:
        n_bins = self.n_bins_discretization
        def inside_rescale_value(value, kind, n_bins, **kwargs):
            return self._rescale_value(kind, value.item(), n_bins, **kwargs)

        new_vector = []
        tensor_cpu = tensor_input.cpu()

        if kind == 'reward':
            rescaled = [inside_rescale_value(v, kind, n_bins, **kwargs) for v in tensor_cpu]
            return torch.tensor(rescaled, device='cuda:0')

        if tensor_cpu.ndim == 1:
            # If it's a 1D array (vector)
            old_X = tensor_cpu[0].item()
            old_Y = tensor_cpu[1].item()
            new_X = self._rescale_value(kind + 'X', old_X, n_bins, **kwargs)
            new_Y = self._rescale_value(kind + 'Y', old_Y, n_bins, **kwargs)
            new_vector.append(torch.tensor([new_X, new_Y], device='cuda:0'))
        elif tensor_cpu.ndim == 2:
            # If it's a 2D array (matrix)
            new_matrix = []
            for row in tensor_cpu:
                old_X = row[0].item()
                old_Y = row[1].item()
                new_X = self._rescale_value(kind + 'X', old_X, n_bins, **kwargs)
                new_Y = self._rescale_value(kind + 'Y', old_Y, n_bins, **kwargs)
                new_matrix.append([new_X, new_Y])
            new_vector.append(torch.tensor(new_matrix, device='cuda:0'))
        else:
            raise ValueError(f"Unsupported tensor shape: {tensor_cpu.shape}")

        if len(new_vector) > 1:
            result_tensor = torch.stack(new_vector)
        else:
            result_tensor = new_vector[0]

        return result_tensor.to('cuda:0')

    def _get_top_indices(self, past_sensors_infos: Tensor, n_sensors: int):
        # Check if all values in each row are zero
        all_zero_indices = torch.all(past_sensors_infos == 0, dim=1)

        # Find indices of maximum values in each row
        max_indices = torch.argmax(past_sensors_infos, dim=1)

        # Sort indices based on values in descending order
        sorted_indices = torch.argsort(past_sensors_infos, descending=True, dim=1)

        # Create a mask to handle cases where n_sensors > number of sensors
        mask = torch.arange(past_sensors_infos.size(1), device=past_sensors_infos.device) < n_sensors

        # Select top n_sensors indices
        max_indices = torch.where(
            ~all_zero_indices[:, None],
            sorted_indices[:, :n_sensors],
            torch.tensor([[-1]], device=past_sensors_infos.device)
        )

        return max_indices

    def observation(self, agent: Agent):
        goal_poses = []
        if self.observe_all_goals:
            for a in self.world.agents:
                goal_pose = agent.state.pos - a.goal.state.pos
                goal_poses.append(goal_pose)
        else:
            goal_pose = agent.state.pos - agent.goal.state.pos
            goal_poses.append(goal_pose)

        # Stack the list of goal poses into a single tensor and move to CUDA device
        goal_poses = torch.stack(goal_poses).to('cuda:0')

        # Ensure the tensor has the correct shape
        if self.observe_all_goals:
            goal_poses = goal_poses.view(-1, 2)  # Shape will be [N, 2] where N is the number of agents
        else:
            goal_poses = goal_poses.squeeze(0)  # Remove the single dimension at index 0

        # Get agent's position and velocity tensors
        agent_pos = agent.state.pos
        agent_vel = agent.state.vel

        # Measure past sensor information
        past_sensors_infos = agent.sensors[0]._max_range - agent.sensors[0].measure()

        if self.n_sensors_to_consider is None:
            if self.n_bins_discretization is None:
                sensors_info = past_sensors_infos
            else:
                sensors_info = self._rescale_vector(past_sensors_infos, 'sensor', max_value=agent.sensors[0]._max_range)
        else:
            sensors_info = self._get_top_indices(past_sensors_infos, self.n_sensors_to_consider)

        if self.n_bins_discretization is None:
            tensor_to_return = torch.cat(
                [agent_pos, agent_vel] + [goal_poses] + ([sensors_info] if self.collisions else []),
                dim=-1,
            )
        else:
            tensor_to_return = torch.cat(
                [
                    self._rescale_vector(agent_pos, 'pos'),
                    self._rescale_vector(agent_vel, 'vel'),
                    self._rescale_vector(goal_poses, 'dist')
                ] + ([sensors_info] if self.collisions else []),
                dim=-1,
            )

        return tensor_to_return

    def done(self):
        return torch.stack(
            [
                torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos,
                    dim=-1,
                )
                < agent.shape.radius
                for agent in self.world.agents
            ],
            dim=-1,
        ).all(-1)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collisions": agent.agent_collision_rew,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []

        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self.comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        return geoms


class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, clf_epsilon=0.2, clf_slack=100.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clf_epsilon = clf_epsilon  # Exponential CLF convergence rate
        self.clf_slack = clf_slack  # weights on CLF-QP slack variable

    def compute_action(self, observation: Tensor, u_range: Tensor) -> Tensor:
        """
        QP inputs:
        These values need to computed apriri based on observation before passing into QP

        V: Lyapunov function value
        lfV: Lie derivative of Lyapunov function
        lgV: Lie derivative of Lyapunov function
        CLF_slack: CLF constraint slack variable

        QP outputs:
        u: action
        CLF_slack: CLF constraint slack variable, 0 if CLF constraint is satisfied
        """
        # Install it with: pip install cvxpylayers
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer

        self.n_env = observation.shape[0]
        self.device = observation.device
        agent_pos = observation[:, :2]
        agent_vel = observation[:, 2:4]
        goal_pos = (-1.0) * (observation[:, 4:6] - agent_pos)

        # Pre-compute tensors for the CLF and CBF constraints,
        # Lyapunov Function from: https://arxiv.org/pdf/1903.03692.pdf

        # Laypunov function
        V_value = (
                (agent_pos[:, X] - goal_pos[:, X]) ** 2
                + 0.5 * (agent_pos[:, X] - goal_pos[:, X]) * agent_vel[:, X]
                + agent_vel[:, X] ** 2
                + (agent_pos[:, Y] - goal_pos[:, Y]) ** 2
                + 0.5 * (agent_pos[:, Y] - goal_pos[:, Y]) * agent_vel[:, Y]
                + agent_vel[:, Y] ** 2
        )

        LfV_val = (2 * (agent_pos[:, X] - goal_pos[:, X]) + agent_vel[:, X]) * (
            agent_vel[:, X]
        ) + (2 * (agent_pos[:, Y] - goal_pos[:, Y]) + agent_vel[:, Y]) * (
                      agent_vel[:, Y]
                  )
        LgV_vals = torch.stack(
            [
                0.5 * (agent_pos[:, X] - goal_pos[:, X]) + 2 * agent_vel[:, X],
                0.5 * (agent_pos[:, Y] - goal_pos[:, Y]) + 2 * agent_vel[:, Y],
            ],
            dim=1,
        )
        # Define Quadratic Program (QP) based controller
        u = cp.Variable(2)
        V_param = cp.Parameter(1)  # Lyapunov Function: V(x): x -> R, dim: (1,1)
        lfV_param = cp.Parameter(1)
        lgV_params = cp.Parameter(
            2
        )  # Lie derivative of Lyapunov Function, dim: (1, action_dim)
        clf_slack = cp.Variable(1)  # CLF constraint slack variable, dim: (1,1)

        constraints = []

        # QP Cost F = u^T @ u + clf_slack**2
        qp_objective = cp.Minimize(cp.sum_squares(u) + self.clf_slack * clf_slack ** 2)

        # control bounds between u_range
        constraints += [u <= u_range]
        constraints += [u >= -u_range]
        # CLF constraint
        constraints += [
            lfV_param + lgV_params @ u + self.clf_epsilon * V_param + clf_slack <= 0
        ]

        QP_problem = cp.Problem(qp_objective, constraints)

        # Initialize CVXPY layers
        QP_controller = CvxpyLayer(
            QP_problem,
            parameters=[V_param, lfV_param, lgV_params],
            variables=[u],
        )

        # Solve QP
        CVXpylayer_parameters = [
            V_value.unsqueeze(1),
            LfV_val.unsqueeze(1),
            LgV_vals,
        ]
        action = QP_controller(*CVXpylayer_parameters, solver_args={"max_iters": 500})[
            0
        ]

        return action


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
