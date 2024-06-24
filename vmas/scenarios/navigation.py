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
                            n_rays=12,
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
        " ********************************************************************************************************* "
        rescaled_rewards_list = []
        for reward in tot_rewards:
            reward_scaled = self._rescale_value('reward', reward)
            rescaled_rewards_list.append(reward_scaled)
        rescaled_rewards = torch.tensor(rescaled_rewards_list, device='cuda:0')
        return rescaled_rewards
        " ********************************************************************************************************* "

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

    " ********************************************************************************************************** "
    def _rescale_value(self, kind: str, value: float|int):

        def discretize_value(value, intervals):
            # Find the interval where the value fits
            for i in range(len(intervals) - 1):
                if intervals[i] <= value < intervals[i + 1]:
                    return (intervals[i] + intervals[i + 1]) / 2
            # Handle the edge cases
            if value < intervals[0]:
                return intervals[0]
            elif value >= intervals[-1]:
                return intervals[-1]

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
            min_value = self.agent_collision_penalty - max(self.x_semidim, self.y_semidim)*2
            n = 20
            intervals = create_intervals(min_value, max_value, n, scale='exponential')
            # print('reward bins: ', n)
        elif kind == 'DX' or kind == 'DY':
            max_value = 0
            min_value = -self.x_semidim*2 if kind == 'DX' else -self.y_semidim*2
            n = 20 # int((self.x_semidim/0.05)**2 * self.x_semidim*2) if kind == 'DX' else int((self.y_semidim/0.05)**2 * self.y_semidim*2)
            intervals = create_intervals(min_value, max_value, n, scale='exponential')
            # print('DX-DY bins: ', n)
        elif kind == 'VX' or kind == 'VY':
            max_value = 1
            min_value = -1
            n = 10 # int((self.x_semidim/0.05)**2 * self.x_semidim*2) if kind == 'DX' else int((self.y_semidim/0.05)**2 * self.y_semidim*2)
            intervals = create_intervals(min_value, max_value, n, scale='exponential')
            # print('VX-VY bins: ', n)
        elif kind == 'sensor':
            max_value = 1
            min_value = 0
            n = 20
            intervals = create_intervals(min_value, max_value, n, scale='exponential')
            # print('sensor bins: ', n)
        elif kind == 'posX' or kind == 'posY':
            max_value = self.x_semidim*2 if kind == 'posX' else self.y_semidim*2
            min_value = -self.x_semidim*2 if kind == 'posX' else -self.y_semidim*2
            n = 20 # int((self.x_semidim/0.05)**2 * self.x_semidim*2) if kind == 'posX' else int((self.y_semidim/0.05)**2 * self.y_semidim*2)
            intervals = create_intervals(min_value, max_value, n, scale='exponential')
            # print('posX-posY bins: ', n)

        new_value = discretize_value(value, intervals)
        return new_value
    " ********************************************************************************************************** "

    def observation(self, agent: Agent):
        self.BINS = int(1/self.min_collision_distance) * max(self.world.x_semidim, self.world.y_semidim) * 2

        goal_poses = []
        if self.observe_all_goals:
            for a in self.world.agents:
                goal_poses.append(agent.state.pos - a.goal.state.pos)
        else:
            goal_poses.append(agent.state.pos - agent.goal.state.pos)

        " ****************************************************************************************************** "
        new_goal_poses, new_poses = [], []
        for tensor_cuda in goal_poses:
            numpy_array = tensor_cuda.cpu().numpy()
            for single_array in numpy_array:
                pX = single_array[0]
                pY = single_array[1]
                new_X = self._rescale_value('DX', pX)
                new_Y = self._rescale_value('DY', pY)
                new_poses.append([new_X, new_Y])
        new_goal_poses.append(torch.tensor(new_poses, device='cuda:0'))
        goal_poses = new_goal_poses

        list_agent_pose = []
        for agent_pose in agent.state.pos:
            agent_pose = agent.state.pos
            tensor_cpu = agent_pose.cpu()
            numpy_array = tensor_cpu.numpy()
            pX = numpy_array[0, 0]
            pY = numpy_array[0, 1]
            new_X = self._rescale_value('posX', pX)
            new_Y = self._rescale_value('posY', pY)
            list_agent_pose.append([new_X, new_Y])
        new_agent_pose = torch.tensor(list_agent_pose, device='cuda:0')

        list_agent_vel = []
        for agent_vel in agent.state.vel:
            agent_vel = agent.state.vel
            tensor_cpu = agent_vel.cpu()
            numpy_array = tensor_cpu.numpy()
            vX = numpy_array[0, 0]
            vY = numpy_array[0, 1]
            new_vX = self._rescale_value('VX', vX)
            new_vY = self._rescale_value('VY', vY)
            list_agent_vel.append([new_vX, new_vY])
        new_agent_vel = torch.tensor(list_agent_vel, device='cuda:0')

        past_sensors_infos = agent.sensors[0]._max_range - agent.sensors[0].measure()
        all_sensor_values = []
        for past_sensors_info in past_sensors_infos:
            tensor_cpu = past_sensors_info.cpu().numpy()
            new_values = []
            for i in range(len(tensor_cpu)):
                past_value = tensor_cpu[i]
                new_value = self._rescale_value('sensor', past_value)
                new_values.append(new_value)
            all_sensor_values.append(new_values)
        new_sensors_info = torch.tensor(all_sensor_values, device='cuda:0')

        " ****************************************************************************************************** "
        # TODO: valid only for one env
        return torch.cat(
            [
                new_agent_pose, #agent.state.pos,
                new_agent_vel, #agent.state.vel
            ]
            + goal_poses
            + (
                [new_sensors_info] # past_sensors_info
                if self.collisions
                else []
            ),
            dim=-1,
        )

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
        qp_objective = cp.Minimize(cp.sum_squares(u) + self.clf_slack * clf_slack**2)

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
