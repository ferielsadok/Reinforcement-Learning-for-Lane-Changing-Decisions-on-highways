import traci

traci.start(
    [
        "sumo-gui", # Start SUMO with the graphical interface
        "-c","data/obstacles.sumocfg",
        "--delay","200",
        "--start", "true",
        "--collision.action", "warn", 
        "--xml-validation","never",
        "--log","log",
    ]
)
i = 0
ego_id="vehAgent"
traci.vehicle.add(ego_id, "r_0", typeID="obstacle", depart=0) 
while i < 200:
    traci.vehicle.setLaneChangeMode(ego_id, 0) # Disable the default lane-changing model for the AV
    vehicleIDs=list(traci.vehicle.getIDList())
    if ego_id in vehicleIDs:
        vehicleIDs.remove(ego_id)
    for veh in vehicleIDs:
        traci.vehicle.setLaneChangeMode(veh, 0)  # Disable lane-changing for all surrounding vehicle
        traci.vehicle.setSpeed(veh, 0) # stop all surrounding vehicle 
    traci.simulationStep() 
    i += 1
traci.close()
