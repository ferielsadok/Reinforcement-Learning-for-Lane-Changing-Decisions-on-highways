import os
import traci
import random
import time

class SumoEnv:
    def __init__(self, sumo_cfg=None, max_steps=200, use_gui=False):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Correction du chemin pour pointer vers src/data/
        self.sumo_cfg = os.path.join(base_dir, "data", "obstacles.sumocfg")
        self.max_steps = max_steps
        self.ego_id = "vehAgent"
        self.step_count = 0
        self.dist_bins = [5, 15, 30]
        self.use_gui = use_gui  # Ne pas oublier d'assigner ceci !

    def discretize_distance(self, dist):
        if dist < self.dist_bins[0]: return 0  # Close
        elif dist < self.dist_bins[1]: return 1 # Medium
        else: return 2 # Far

    def get_state(self):
        # Sécurité critique : si le véhicule a crashé/disparu
        if self.ego_id not in traci.vehicle.getIDList():
            return [0, 2, 2] 

        try:
            lane = traci.vehicle.getLaneIndex(self.ego_id)
            leader = traci.vehicle.getLeader(self.ego_id, dist=100)
            dist_current = leader[1] if leader else 100
            ego_pos = traci.vehicle.getLanePosition(self.ego_id)
            
            other_lane = 1 - lane
            min_dist = 100
            for veh in traci.vehicle.getIDList():
                if veh != self.ego_id:
                    if traci.vehicle.getLaneIndex(veh) == other_lane:
                        veh_pos = traci.vehicle.getLanePosition(veh)
                        dist = veh_pos - ego_pos
                        if 0 < dist < min_dist: min_dist = dist
            return [lane, self.discretize_distance(dist_current), self.discretize_distance(min_dist)]
        except:
            return [0, 2, 2]

    def compute_reward(self, action, lane_valid, dist_current_idx):
        if not lane_valid: return -10
        
        # Si le véhicule n'est plus là (collision détectée)
        if self.ego_id not in traci.vehicle.getIDList():
            return -100

        try:
            speed = traci.vehicle.getSpeed(self.ego_id)
            r_step = speed / 13.89 # Normalisé par rapport à 50km/h
        except:
            r_step = 0
        
        r_lane_change = -0.5 if action != 0 else 0
        r_collision = -50 if dist_current_idx == 0 else 0 

        return r_step + r_lane_change + r_collision

    def reset(self):
        try:
            traci.close()
        except:
            pass
        
        # Attente très courte (Mac est sensible aux ports)
        time.sleep(0.2)

        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        
        traci.start([
            sumo_binary, "-c", self.sumo_cfg,
            "--start", "true",
            "--quit-on-end", "true",
            "--no-warnings", "true"
        ])
        
        self.step_count = 0
        traci.simulationStep()

        try:
            # On utilise un type de véhicule standard si 'obstacle' n'est pas défini
            traci.vehicle.add(self.ego_id, routeID="r_0")
            traci.simulationStep()
            traci.vehicle.setLaneChangeMode(self.ego_id, 0)
        except Exception as e:
            print(f"Erreur Reset: {e}")

        return self.get_state()

    def step(self, action):
        # 1. Vérifier existence
        if self.ego_id not in traci.vehicle.getIDList():
            return [0, 2, 2], -100, True

        lane_valid = True
        try:
            lane = traci.vehicle.getLaneIndex(self.ego_id)
            if action == 1: # LEFT
                if lane < 1: traci.vehicle.changeLane(self.ego_id, lane + 1, 1)
                else: lane_valid = False
            elif action == 2: # RIGHT
                if lane > 0: traci.vehicle.changeLane(self.ego_id, lane - 1, 1)
                else: lane_valid = False
        except:
            lane_valid = False

        traci.simulationStep()
        self.step_count += 1
        
        # 2. Vérifier si encore vivant après le step
        if self.ego_id not in traci.vehicle.getIDList():
            return [0, 2, 2], -100, True

        next_state = self.get_state()
        reward = self.compute_reward(action, lane_valid, next_state[1])
        done = self.step_count >= self.max_steps
        
        return next_state, reward, done

    def close(self):
        try:
            traci.close()
        except:
            pass