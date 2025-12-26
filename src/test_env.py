from env import SumoEnv
import random

def main():
    # Create environment
    env = SumoEnv(max_steps=50)  # shorter test for quick run
    state = env.reset()
    done = False
    step_count = 0

    print("Starting environment test...")

    while not done:
        # Choose a random action: 0=Keep, 1=Left, 2=Right
        action = random.choice([0, 1, 2])

        # Take step in environment
        next_state, reward, done = env.step(action)
        step_count += 1

        print(f"Step {step_count}: State={next_state}, Action={action}, Reward={reward}")

    env.close()
    print("Environment test finished.")

if __name__ == "__main__":
    main()
