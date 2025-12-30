import os
import time
import traci
import numpy as np
import gym
from gym import spaces


class SumoContinuousEnv(gym.Env):
    def __init__(self, sumo_cfg="data/obstacles.sumocfg", max_steps=200):
        super().__init__()

        # Configuration SUMO
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.sumo_cfg = os.path.join(base_dir, sumo_cfg)
        self.max_steps = max_steps
        self.ego_id = "vehAgent"
        self.step_count = 0

        # Actions: 0=keep, 1=left, 2=right
        self.action_space = spaces.Discrete(3)

        # State: [lane, dist_current, dist_other]
        self.observation_space = spaces.Box(
            low=np.array([0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1, 100.0, 100.0], dtype=np.float32),
            dtype=np.float32
        )

    def get_state(self):
        # ⚠ Vérification que le véhicule existe
        if self.ego_id not in traci.vehicle.getIDList():
            raise RuntimeError(f"Vehicle '{self.ego_id}' not known in SUMO")

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
                veh_pos = traci.vehicle.getLanePosition(veh)
                d = veh_pos - ego_pos
                if 0 < d < min_dist:
                    min_dist = d

        return np.array([lane, dist_current, min_dist], dtype=np.float32)

    def compute_reward(self, action, lane_valid, dist_current):
        speed = traci.vehicle.getSpeed(self.ego_id)
        max_speed = traci.vehicle.getMaxSpeed(self.ego_id)
        r_speed = speed / max_speed if max_speed > 0 else 0.0

        r_lane_change = -0.1 if action in [1, 2] else 0.0
        r_collision = -10.0 if dist_current < 2.0 else 0.0

        if not lane_valid:
            r_collision = -5.0

        return r_speed + r_lane_change + r_collision

    def reset(self):
        # Lancer SUMO
        traci.start([
            "sumo-gui",
            "-c", self.sumo_cfg,
            "--start", "true",
            "--collision.action", "warn",
            "--xml-validation", "never",
            "--quit-on-end", "false"
        ])
        self.step_count = 0

        # Ajouter le véhicule ego
        traci.vehicle.add(
            self.ego_id,
            "r_0",
            typeID="obstacle",
            depart=0
        )

        # ⚡ Attendre que le véhicule soit bien enregistré
        for _ in range(10):
            traci.simulationStep()
            if self.ego_id in traci.vehicle.getIDList():
                break
        else:
            raise RuntimeError(f"Vehicle '{self.ego_id}' was not added in SUMO")

        # Freeze les autres véhicules
        for veh in traci.vehicle.getIDList():
            if veh != self.ego_id:
                traci.vehicle.setLaneChangeMode(veh, 0)
                traci.vehicle.setSpeed(veh, 0)

        # Bloquer le lane change du véhicule ego
        traci.vehicle.setLaneChangeMode(self.ego_id, 0)

        return self.get_state()

    def step(self, action):
        if self.ego_id not in traci.vehicle.getIDList():
            raise RuntimeError(f"Vehicle '{self.ego_id}' disappeared in SUMO")

        lane = traci.vehicle.getLaneIndex(self.ego_id)
        lane_valid = True

        if action == 1:
            if lane > 0:
                traci.vehicle.changeLane(self.ego_id, lane - 1, 50)
            else:
                lane_valid = False
        elif action == 2:
            if lane < 1:
                traci.vehicle.changeLane(self.ego_id, lane + 1, 50)
            else:
                lane_valid = False

        traci.simulationStep()
        self.step_count += 1

        state = self.get_state()
        reward = self.compute_reward(action, lane_valid, state[1])

        done = self.step_count >= self.max_steps

        return state, reward, done, {}

    def close(self):
        traci.close()

