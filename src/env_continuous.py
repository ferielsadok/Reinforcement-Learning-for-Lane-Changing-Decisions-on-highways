import os
import traci
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random


class SumoContinuousEnv(gym.Env):

    def __init__(self, sumo_cfg="data/obstacles.sumocfg", max_steps=200, gui=False):
        super().__init__()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.sumo_cfg = os.path.join(base_dir, sumo_cfg)
        self.max_steps = max_steps
        self.gui = gui

        self.ego_id = "vehAgent"
        self.step_count = 0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1, 100.0, 100.0], dtype=np.float32),
            dtype=np.float32
        )

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    def _safe_close_traci(self):
        try:
            if traci.isLoaded():
                traci.close()
        except Exception:
            pass

    def _start_sumo(self):
        self._safe_close_traci()
        traci.start([
            "sumo-gui" if self.gui else "sumo",
            "-c", self.sumo_cfg,
            "--start", "true",
            "--collision.action", "remove",
            "--xml-validation", "never",
            "--quit-on-end", "true"
        ])

    def get_state(self):
        lane = traci.vehicle.getLaneIndex(self.ego_id)

        leader = traci.vehicle.getLeader(self.ego_id, dist=100)
        dist_current = leader[1] if leader else 100.0

        ego_pos = traci.vehicle.getLanePosition(self.ego_id)
        other_lane = 1 - lane
        min_dist = 100.0

        for veh in traci.vehicle.getIDList():
            if veh == self.ego_id:
                continue
            if traci.vehicle.getLaneIndex(veh) == other_lane:
                d = traci.vehicle.getLanePosition(veh) - ego_pos
                if 0 < d < min_dist:
                    min_dist = d

        return np.array([lane, dist_current, min_dist], dtype=np.float32)

    def compute_reward(self, action, lane_valid, dist_current):
        speed = traci.vehicle.getSpeed(self.ego_id)
        max_speed = traci.vehicle.getMaxSpeed(self.ego_id)

        r_speed = speed / max_speed if max_speed > 0 else 0.0
        r_lane = -0.1 if action in [1, 2] else 0.0
        r_collision = -10.0 if dist_current < 2.0 else 0.0

        if not lane_valid:
            r_collision = -5.0

        return r_speed + r_lane + r_collision

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seed(seed)

        self.step_count = 0
        self._start_sumo()

        traci.vehicle.add(self.ego_id, "r_0", typeID="obstacle", depart=0)
        traci.simulationStep()

        for veh in traci.vehicle.getIDList():
            if veh != self.ego_id:
                traci.vehicle.setSpeed(veh, 0)
                traci.vehicle.setLaneChangeMode(veh, 0)

        traci.vehicle.setLaneChangeMode(self.ego_id, 0)

        return self.get_state(), {}

    def step(self, action):

        if self.ego_id not in traci.vehicle.getIDList():
            self._safe_close_traci()
            state = np.zeros(self.observation_space.shape, dtype=np.float32)
            return state, -20.0, True, False, {}

        lane = traci.vehicle.getLaneIndex(self.ego_id)
        lane_valid = True

        if action == 1 and lane > 0:
            traci.vehicle.changeLane(self.ego_id, lane - 1, 50)
        elif action == 2 and lane < 1:
            traci.vehicle.changeLane(self.ego_id, lane + 1, 50)
        elif action in [1, 2]:
            lane_valid = False

        traci.simulationStep()
        self.step_count += 1

        if self.ego_id not in traci.vehicle.getIDList():
            self._safe_close_traci()
            state = np.zeros(self.observation_space.shape, dtype=np.float32)
            return state, -20.0, True, False, {}

        state = self.get_state()
        reward = self.compute_reward(action, lane_valid, state[1])

        terminated = self.step_count >= self.max_steps
        truncated = False

        if terminated:
            self._safe_close_traci()

        return state, reward, terminated, truncated, {}

    def close(self):
        self._safe_close_traci()


