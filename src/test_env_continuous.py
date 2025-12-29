import time
import numpy as np
from env_continuous import SumoContinuousEnv


def main():
    # Create environment
    env = SumoContinuousEnv(
        sumo_cfg="data/obstacles.sumocfg",
        max_steps=200
    )

    # Reset environment
    state = env.reset()
    print("Initial state:", state)

    done = False
    step = 0

    # Run one episode
    while not done:
        # Take a RANDOM action (just to test)
        action = env.action_space.sample()

        # Step environment
        next_state, reward, done, _ = env.step(action)

        # Print info
        print(
            f"Step {step:03d} | "
            f"Action: {action} | "
            f"State: {next_state} | "
            f"Reward: {reward:.3f}"
        )

        state = next_state
        step += 1

        # Slow down so you can see SUMO GUI
        time.sleep(0.1)

    # Close SUMO
    env.close()
    print("Episode finished.")


if __name__ == "__main__":
    main()
