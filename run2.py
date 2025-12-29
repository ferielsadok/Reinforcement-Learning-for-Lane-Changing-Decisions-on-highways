import traci
import time

# -----------------------------
# Start SUMO with GUI
# -----------------------------
traci.start([
    "sumo-gui",
    "-c", "data/obstacles.sumocfg",
    "--start", "true",
    "--collision.action", "warn",
    "--xml-validation", "never",
    "--log", "log",
    "--quit-on-end", "false"   # ✅ prevent auto close
])

ego_id = "vehAgent"

# Step once before adding vehicle
traci.simulationStep()

# -----------------------------
# Add the autonomous vehicle
# -----------------------------
traci.vehicle.add(
    vehID=ego_id,
    routeID="r_0",
    typeID="obstacle",
    depart=0,
    departLane=0
)

# Step once so vehicle actually appears
traci.simulationStep()

# Disable automatic lane change for AV
traci.vehicle.setLaneChangeMode(ego_id, 0)

# -----------------------------
# Simulation loop
# -----------------------------
for step in range(200):
    vehicleIDs = traci.vehicle.getIDList()
    
    # Stop all other vehicles
    for veh in vehicleIDs:
        if veh != ego_id:
            traci.vehicle.setLaneChangeMode(veh, 0)
            traci.vehicle.setSpeed(veh, 0)
    
    # Manual lane change
    if step == 50:
        traci.vehicle.changeLane(ego_id, 1, 10)
    if step == 100:
        traci.vehicle.changeLane(ego_id, 0, 10)
    
    traci.simulationStep()
    time.sleep(0.1)   # ✅ slow down so you can see it
    
    # Print status
    lane_index = traci.vehicle.getLaneIndex(ego_id)
    speed = traci.vehicle.getSpeed(ego_id)
    print(f"Step {step}: AV lane={lane_index}, speed={speed}")

traci.close()
