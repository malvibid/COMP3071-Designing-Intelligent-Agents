import base64
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Environment Components
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

# Selenium for automatically loading and play the game
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


# Create Dino Game Environment
class DinoEnvironment(Env):

    def __init__(self, screen_width=96, screen_height=96):

        # Subclass model
        super().__init__()

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.driver = self._create_driver()

        # Setup spaces
        self.observation_space = Box(low=0, high=255, shape=(
            self.screen_width, self.screen_height, 4), dtype=np.uint8)
        self.action_space = Discrete(3)  # Jump up, Duck down or Do nothing

        # Define actions
        self.actions_map = [
            Keys.ARROW_UP,  # Jump up
            Keys.ARROW_DOWN,  # Duck down
            Keys.ARROW_RIGHT  # Do nothing
        ]

    # Create and return an instance of the Chrome Driver
    def _create_driver(self):

        # Set options for the WebDriver
        options = Options()

        # Turn off logging to keep terminal clean
        options.add_experimental_option('excludeSwitches', ['enable-logging'])

        # Keep the browser running after the code finishes executing
        options.add_experimental_option("detach", True)

        # Create a Service instance for running the ChromeDriver executable
        service = Service(executable_path=ChromeDriverManager().install())

        # Create an instance of the Chrome WebDriver with the specified service and options - The driver object can be used to automate interactions with the Chrome browser
        driver = webdriver.Chrome(service=service, options=options)

        # Maximize the Chrome window
        driver.maximize_window()

        return driver

    # Helper method to encode the obstacle type as an integer
    def _encode_obstacle_type(self, obstacle_type):
        if obstacle_type == 'CACTUS_SMALL':
            return 0
        elif obstacle_type == 'CACTUS_LARGE':
            return 1
        elif obstacle_type == 'PTERODACTYL':
            return 2
        else:
            raise ValueError(f"Unknown obstacle type: {obstacle_type}")

    # Get obstacles that are currently on the screen
    def _get_obstacles(self):
        obstacles = self.driver.execute_script(
            "return Runner.instance_.horizon.obstacles")
        obstacle_info = []
        for obstacle in obstacles:
            obstacle_type = obstacle['typeConfig']['type']
            # Encode the obstacle type as an integer
            encoded_obstacle_type = self._encode_obstacle_type(obstacle_type)
            obstacle_x = obstacle['xPos']
            obstacle_y = obstacle['yPos']
            obstacle_width = obstacle['typeConfig']['width']
            obstacle_height = obstacle['typeConfig']['height']
            obstacle_info.append(
                (encoded_obstacle_type, obstacle_x, obstacle_y, obstacle_width, obstacle_height))
        return obstacle_info

    # Get Trex's state (Jumping, Ducking or Running/Do nothing)
    def _get_trex_info(self):
        trex = self.driver.execute_script("return Runner.instance_.tRex")
        # xpos remains the same throughout the game - don't need it
        trex_y = trex['yPos']
        trex_is_jumping = trex['jumping']
        trex_is_ducking = trex['ducking']
        return trex_y, trex_is_jumping, trex_is_ducking

    # Get current game speed
    def _get_game_speed(self):
        game_speed = self.driver.execute_script(
            "return Runner.instance_.currentSpeed")
        return game_speed

    # Get the distance between the Trex and the next obstacle
    def _get_distance_to_next_obstacle(self):
        trex_x = self.driver.execute_script(
            "return Runner.instance_.tRex.xPos")  # xpos of trex
        obstacles = self._get_obstacles()
        if obstacles:
            next_obstacle = obstacles[0]
            obstacle_x = next_obstacle[1]  # xpos of next obstacle
            distance_to_next_obstacle = obstacle_x - trex_x
        else:
            distance_to_next_obstacle = None
        return distance_to_next_obstacle

    # Get and return the score for the last game played
    def _get_current_score(self):
        try:
            score = int(''.join(self.driver.execute_script(
                "return Runner.instance_.distanceMeter.digits")))
        except:
            score = 0
        return score

    # Get and return the high score for all games played in current browser session
    def _get_high_score(self):
        try:
            score = int(''.join(self.driver.execute_script(
                "return Runner.instance_.distanceMeter.highScore.slice(-5)")))  # MaxScore=99999, MaxScoreUnits=5
        except:
            score = 0
        return score

    # Capture screenshot of current game state and return the image captured for rendering
    def _get_image(self):
        # Capture a screenshot of the game canvas as a data URL - string that represents the image in base64-encoded format
        data_url = self.driver.execute_script(
            "return document.querySelector('canvas.runner-canvas').toDataURL()")

        # Remove the leading text from the data URL using string slicing and decode the remaining base64-encoded data
        LEADING_TEXT = "data:image/png;base64,"
        image_data = base64.b64decode(data_url[len(LEADING_TEXT):])

        # Convert the binary data in 'image_data' to a 1D NumPy array
        image_array = np.frombuffer(image_data, dtype=np.uint8)

        # Decode the image data and create an OpenCV image object - OpenCV Image Shape format (H, W, C) ( rows, columns, and channels )
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        return image

    # Load and Reset the game environment
    def reset(self):
        try:
            # Navigate to the Chrome Dino website
            self.driver.get("chrome://dino/")

        except WebDriverException as e:
            # Ignore "ERR_INTERNET_DISCONNECTED" error thrown because this game is available offline
            if "ERR_INTERNET_DISCONNECTED" in str(e):
                pass  # Ignore the exception.
            else:
                raise e  # Handle other WebDriverExceptions

        # Avoid errors that can arise due to the 'runner-canvas' element not being present - Using WebDriverWait and EC together ensures that the code does not proceed until the required element is present
        timeout = 10
        WebDriverWait(self.driver, timeout).until(
            EC.presence_of_element_located((By.CLASS_NAME, "runner-canvas")))

        # Start game
        self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.SPACE)

        return self.get_observation()

    # Get the current state of the game and return it as the observation
    def get_observation(self):
        obstacles = self._get_obstacles()
        trex_y, trex_is_jumping, trex_is_ducking = self._get_trex_info()
        game_speed = self._get_game_speed()
        distance_to_next_obstacle = self._get_distance_to_next_obstacle()

        state = (
            trex_y,
            int(trex_is_jumping),
            int(trex_is_ducking),
            game_speed,
            distance_to_next_obstacle,
            # Unpack the tuple of the first obstacle
            *(obstacles[0] if obstacles else (None, None, None, None, None))
        )

        state = np.array(state, dtype=object)
        # Set dtype for game_speed to float32
        state[3] = np.float32(state[3])

        # Replace None values with -1
        state[np.equal(state, None)] = -1

        return state

    # Check if the game is over and return True or False
    def is_game_over(self):
        # Done if dino crashed into an obstacle
        return self.driver.execute_script("return Runner.instance_.crashed")

    # Calculate and return the reward for the current state of the game
    def get_reward(self, done):
        # Simple strategy - Dino gets a point for every step it is alive
        # For further optimisation - can make reward strategy better using scores
        return 1

    # Take a step in the game environment based on the given action
    def step(self, action):

        # Take action
        # Create a new ActionChains object
        action_chains = ActionChains(self.driver)

        # Perform the key press action
        action_chains.key_down(self.actions_map[action]).perform()

        # If the action is "Duck down", hold the key down for a short duration
        if action == 1:  # Duck down action
            time.sleep(0.2)

        # Perform the key release action
        action_chains.key_up(self.actions_map[action]).perform()

        # Get next observation
        obs = self.get_observation()

        # Check whether game is over
        done = self.is_game_over()

        # Get reward
        reward = self.get_reward(done)

        info = {
            'current_score': self._get_current_score(),
            'high_score': self._get_high_score()
        }

        return obs, reward, done, info

    # Visualise the game
    def render(self, mode: str = 'human'):
        img = cv2.cvtColor(self._get_image(), cv2.COLOR_BGR2RGB)
        if mode == 'rgb-array':
            return img
        elif mode == 'human':
            cv2.imshow('Dino Game', img)
            cv2.waitKey(1)

    # Close the game environment and the driver
    def close(self):
        self.driver.quit()


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
for episode in range(1):
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

    print_formatted_obs(all_observations)
    print(
        f"Episode: {episode}, Total Reward: {total_reward}, Info: {info}")
