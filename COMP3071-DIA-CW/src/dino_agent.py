from dino_environment import DinoEnvironment
import pandas as pd

env = DinoEnvironment()


def print_formatted_obs(observations):
    obs_titles = ["trex_y", "trex_jumping", "trex_ducking", "game_speed", "obst_dist",
                  "obst_type", "obst_x", "obst_y", "obst_width", "obst_height"]
    # Create a pandas DataFrame
    df = pd.DataFrame(observations, columns=obs_titles)

    # Set the pandas display options for better readability (optional)
    pd.set_option("display.width", 120)
    # pd.set_option("display.precision", 2)

    # Print the DataFrame
    print(df)


# Test loop - Play 5 game
for episode in range(3):
    obs = env.reset()
    done = False
    total_reward = 0
    all_observations = []
    # images = []

    while not done:
        action = env.action_space.sample()  # Take random actions
        obs, reward, done, info = env.step(action)
        # print(obs)
        all_observations.append(obs)  # Print obs formatted nicely in a table
        total_reward += reward

        # env.render(mode='human')
        # img = env.render(mode='rgb-array')
        # images.append(img) # Can use some image library to create a gif using collected images

    # print_formatted_obs(all_observations)
    print(
        f"Episode: {episode}, Total Reward: {total_reward}, Info: {info}")
