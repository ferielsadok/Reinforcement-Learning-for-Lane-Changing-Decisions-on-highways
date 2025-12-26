import traci
import random

class SumoEnv:
    def __init__(self, sumo_cfg="data/obstacles.sumocfg", max_steps=200):
        self.sumo_cfg = sumo_cfg
        self.max_steps = max_steps
        self.ego_id = "vehAgent"
        self.step_count = 0

        # Discretization bins for distances
        self.dist_bins = [5, 15, 30]  # Close, Medium, Far

    # -----------------------
    # Helper: discretize distance
    # -----------------------
    def discretize_distance(self, dist):
        if dist < self.dist_bins[0]:
            return 0  # Close
        elif dist < self.dist_bins[1]:
            return 1  # Medium
        else:
            return 2  # Far

    # -----------------------
    # Helper: get state
    # -----------------------
    def get_state(self):
        lane = traci.vehicle.getLaneIndex(self.ego_id)

        # Distance to vehicle ahead in current lane
        leader = traci.vehicle.getLeader(self.ego_id, dist=100)
        dist_current = leader[1] if leader else 100

        # Distance to vehicle ahead in other lane
        ego_pos = traci.vehicle.getLanePosition(self.ego_id)
        other_lane = 1 - lane
        min_dist = 100

        for veh in traci.vehicle.getIDList():
            if veh == self.ego_id:
                continue
            if traci.vehicle.getLaneIndex(veh) == other_lane:
                veh_pos = traci.vehicle.getLanePosition(veh)
                dist = veh_pos - ego_pos
                if dist > 0 and dist < min_dist:
                    min_dist = dist
        dist_other = min_dist
        # Discretize distances
        return [
        lane,
        self.discretize_distance(dist_current),
        self.discretize_distance(dist_other)
        ]

    # -----------------------
    # Reward function
    # -----------------------
    def compute_reward(self, action, lane_valid, dist_current):
        # Step reward: normalized speed
        speed = traci.vehicle.getSpeed(self.ego_id)
        max_speed = traci.vehicle.getMaxSpeed(self.ego_id)
        r_step = speed / max_speed if max_speed > 0 else 0

        # Lane-change penalty
        r_lane_change = -0.1 if action in [1, 2] else 0

        # Collision/too-close penalty
        r_collision = -10 if dist_current < 2 else 0

        # Invalid lane change penalty
        if not lane_valid:
            r_collision = -5

        return r_step + r_lane_change + r_collision

    # -----------------------
    # Reset simulation
    # -----------------------
    def reset(self):
        traci.start([
            "sumo-gui",
            "-c", self.sumo_cfg,
            "--start", "true",
            "--collision.action", "warn",
            "--xml-validation", "never"
        ])
        self.step_count = 0

        # Add AV
        traci.vehicle.add(self.ego_id, "r_0", typeID="obstacle", depart=0)

        # Stop all other vehicles and disable lane changes
        vehicleIDs = list(traci.vehicle.getIDList())
        for veh in vehicleIDs:
            if veh != self.ego_id:
                traci.vehicle.setLaneChangeMode(veh, 0)
                traci.vehicle.setSpeed(veh, 0)

        # Disable AV default lane-changing
        traci.vehicle.setLaneChangeMode(self.ego_id, 0)

        return self.get_state()

    # -----------------------
    # Take an action
    # -----------------------
    def step(self, action):
        lane = traci.vehicle.getLaneIndex(self.ego_id)
        lane_valid = True

        # -----------------------
        # Apply lane-changing action safely
        # -----------------------
        if action == 1:  # LEFT
            if lane > 0:
                traci.vehicle.changeLane(self.ego_id, lane - 1, 50)
            else:
                lane_valid = False
        elif action == 2:  # RIGHT
            if lane < 1:  # 2-lane highway: max index = 1
                traci.vehicle.changeLane(self.ego_id, lane + 1, 50)
            else:
                lane_valid = False
         # action 0 = keep lane → do nothing

        # -----------------------
        # Advance SUMO simulation by one step
        # -----------------------
        traci.simulationStep()
        self.step_count += 1

        # -----------------------
        # Get next state
        # -----------------------
        next_state = self.get_state()

        # -----------------------
        # Compute reward
        # -----------------------
        dist_current = next_state[1]  # discretized distance in current lane
        reward = self.compute_reward(action, lane_valid, dist_current)

        # -----------------------
        # Check if episode done
        # -----------------------
        done = not lane_valid or self.step_count >= self.max_steps

        return next_state, reward, done


    # -----------------------
    # Close SUMO
    # -----------------------
    def close(self):
        traci.close()
