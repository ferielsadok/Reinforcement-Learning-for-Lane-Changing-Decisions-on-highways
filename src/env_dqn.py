import os
import traci
import time
import numpy as np

class SumoEnvDQN:
    def __init__(self, sumo_cfg="data/obstacles.sumocfg", max_steps=200):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.sumo_cfg = os.path.join(base_dir, "data", "obstacles.sumocfg")

        self.max_steps = max_steps
        self.ego_id = "vehAgent"
        self.step_count = 0

        # Observation limits (for normalization)
        self.max_dist = 100.0
        self.max_speed = 30.0

        # Action space: 0 = stay, 1 = left, 2 = right
        self.n_actions = 3
        self.state_dim = 5

    # --------------------
    # STATE (continuous)
    # --------------------
    def get_state(self):
        if self.ego_id not in traci.vehicle.getIDList():
            return np.zeros(self.state_dim, dtype=np.float32)

        lane = traci.vehicle.getLaneIndex(self.ego_id)

        # Speed
        speed = traci.vehicle.getSpeed(self.ego_id)

        # Distance in current lane
        leader = traci.vehicle.getLeader(self.ego_id, dist=self.max_dist)
        dist_current = leader[1] if leader else self.max_dist

        # Distance in other lane
        ego_pos = traci.vehicle.getLanePosition(self.ego_id)
        other_lane = 1 - lane
        min_dist = self.max_dist

        for veh in traci.vehicle.getIDList():
            if veh != self.ego_id:
                try:
                    if traci.vehicle.getLaneIndex(veh) == other_lane:
                        veh_pos = traci.vehicle.getLanePosition(veh)
                        d = veh_pos - ego_pos
                        if 0 < d < min_dist:
                            min_dist = d
                except:
                    continue

        # Normalization (VERY IMPORTANT FOR DQN)
        state = np.array([
            lane / 1.0,                     # lane ∈ {0,1}
            speed / self.max_speed,
            dist_current / self.max_dist,
            min_dist / self.max_dist,
            self.step_count / self.max_steps
        ], dtype=np.float32)

        return state

    # --------------------
    # REWARD
    # --------------------
    def compute_reward(self, action, lane_valid, dist_current):
        if not lane_valid:
            return -5.0

        speed = traci.vehicle.getSpeed(self.ego_id)
        r_speed = speed / self.max_speed

        r_lane_change = -0.1 if action in [1, 2] else 0.0
        r_collision = -10.0 if dist_current < 5.0 else 0.0

        return r_speed + r_lane_change + r_collision

    # --------------------
    # RESET
    # --------------------
    def reset(self):
        try:
            traci.close()
            time.sleep(2)
        except:
            pass

        traci.start([
            "sumo", "-c", self.sumo_cfg,
            "--start", "true",
            "--quit-on-end", "true",
            "--no-warnings", "true"
        ])

        self.step_count = 0
        traci.simulationStep()

        try:
            traci.vehicle.add(self.ego_id, routeID="r_0", typeID="obstacle")
            traci.simulationStep()
            traci.vehicle.setLaneChangeMode(self.ego_id, 0)
        except:
            pass

        for veh in traci.vehicle.getIDList():
            if veh != self.ego_id:
                traci.vehicle.setSpeed(veh, 0)
                traci.vehicle.setLaneChangeMode(veh, 0)

        return self.get_state()

    # --------------------
    # STEP
    # --------------------
    def step(self, action):
        lane_valid = True

        try:
            lane = traci.vehicle.getLaneIndex(self.ego_id)
            edge_id = traci.vehicle.getRoadID(self.ego_id)
            num_lanes = traci.edge.getLaneNumber(edge_id)

            if action == 1 and lane < num_lanes - 1:
                traci.vehicle.changeLane(self.ego_id, lane + 1, 1)
            elif action == 2 and lane > 0:
                traci.vehicle.changeLane(self.ego_id, lane - 1, 1)
            elif action != 0:
                lane_valid = False

        except:
            lane_valid = False

        traci.simulationStep()
        self.step_count += 1

        next_state = self.get_state()
        reward = self.compute_reward(
            action,
            lane_valid,
            next_state[2] * self.max_dist
        )

        done = self.step_count >= self.max_steps

        return next_state, reward, done

    def close(self):
        try:
            traci.close()
        except:
            pass
