import os
import traci
import random
import time

class SumoEnv:
    def __init__(self, sumo_cfg="data/obstacles.sumocfg", max_steps=200, use_gui=False):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.sumo_cfg = os.path.join(base_dir, "data", "obstacles.sumocfg")
        self.max_steps = max_steps
        self.ego_id = "vehAgent"
        self.step_count = 0
        self.dist_bins = [5, 15, 30]

    def discretize_distance(self, dist):
        if dist < self.dist_bins[0]: return 0  # Close
        elif dist < self.dist_bins[1]: return 1 # Medium
        else: return 2 # Far

    def get_state(self):
        # Sécurité : Si le véhicule n'est pas encore ou plus dans la simulation
        if self.ego_id not in traci.vehicle.getIDList():
            return [0, 2, 2] 

        lane = traci.vehicle.getLaneIndex(self.ego_id)
        
        # Distance au leader sur la voie actuelle
        leader = traci.vehicle.getLeader(self.ego_id, dist=100)
        dist_current = leader[1] if leader else 100

        # Distance au leader sur l'autre voie
        ego_pos = traci.vehicle.getLanePosition(self.ego_id)
        other_lane = 1 - lane
        min_dist = 100
        
        for veh in traci.vehicle.getIDList():
            if veh != self.ego_id:
                try:
                    if traci.vehicle.getLaneIndex(veh) == other_lane:
                        veh_pos = traci.vehicle.getLanePosition(veh)
                        dist = veh_pos - ego_pos
                        if 0 < dist < min_dist:
                            min_dist = dist
                except:
                    continue
        
        return [lane, self.discretize_distance(dist_current), self.discretize_distance(min_dist)]

    def compute_reward(self, action, lane_valid, dist_current_idx):
        # Pénalité si l'action est impossible (ex: changer de voie là où il n'y en a pas)
        if not lane_valid:
            return -5 
        
        # Récompense basée sur la vitesse
        try:
            speed = traci.vehicle.getSpeed(self.ego_id)
            max_speed = traci.vehicle.getMaxSpeed(self.ego_id)
            r_step = speed / max_speed if max_speed > 0 else 0
        except:
            r_step = 0
        
        # Petite pénalité pour chaque changement de voie (pour éviter les zig-zags inutiles)
        r_lane_change = -0.2 if action in [1, 2] else 0
        
        # Grosse pénalité si on est trop proche du véhicule de devant
        r_collision = -15 if dist_current_idx == 0 else 0 

        return r_step + r_lane_change + r_collision

    def reset(self):
        # Fermeture propre de l'ancienne instance
        try:
            traci.close()
            time.sleep(5) # Petit délai pour laisser le port se libérer
        except:
            pass
            

        # Lancement de SUMO
        # Astuce : remplacez "sumo-gui" par "sumo" pour un entraînement beaucoup plus rapide
        traci.start([
            "sumo-gui", "-c", self.sumo_cfg,
            "--start", "true",
            "--quit-on-end", "true",
            "--no-warnings", "true"
        ], label="default")
        
        self.step_count = 0
        traci.simulationStep()

        # Ajout de l'Agent
        try:
            traci.vehicle.add(self.ego_id, routeID="r_0", typeID="obstacle")
            traci.simulationStep()
            # Désactiver le contrôle automatique de SUMO sur l'agent
            traci.vehicle.setLaneChangeMode(self.ego_id, 0)
        except traci.TraCIException as e:
            print(f"Erreur init véhicule: {e}")

        # Immobiliser les obstacles (les autres voitures)
        for veh in traci.vehicle.getIDList():
            if veh != self.ego_id:
                traci.vehicle.setSpeed(veh, 0)
                traci.vehicle.setLaneChangeMode(veh, 0)

        return self.get_state()

    def step(self, action):
        lane_valid = True
        try:
            lane = traci.vehicle.getLaneIndex(self.ego_id)
            edge_id = traci.vehicle.getRoadID(self.ego_id)
            num_lanes = traci.edge.getLaneNumber(edge_id) if edge_id else 2
            
            if action == 1: # LEFT
                if lane < num_lanes - 1:
                    traci.vehicle.changeLane(self.ego_id, lane + 1, 1)
                else:
                    lane_valid = False
            elif action == 2: # RIGHT
                if lane > 0:
                    traci.vehicle.changeLane(self.ego_id, lane - 1, 1)
                else:
                    lane_valid = False
        except:
            lane_valid = False

        traci.simulationStep()
        self.step_count += 1
        
        next_state = self.get_state()
        reward = self.compute_reward(action, lane_valid, next_state[1])
        
        # L'épisode ne s'arrête QUE si le temps est écoulé (max_steps)
        # On ne met pas "not lane_valid" ici pour éviter les ouvertures/fermetures incessantes
        done = self.step_count >= self.max_steps
        
        return next_state, reward, done

    def close(self):
        try:
            traci.close()
        except:
            pass