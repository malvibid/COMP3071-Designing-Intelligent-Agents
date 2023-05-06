import base64
import cv2
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

    def __init__(self):

        # Subclass model
        super().__init__()

        self.driver = self._create_driver()

        # Setup spaces
        low_values = np.array(
            [0, 0, 0, 6, -1, -1, -1, -1, -1, -1], dtype=np.float32)  # Initial speed is 6, while max speed is 13
        high_values = np.array(
            [150, 1, 1, 13, 600, 3, 600, 150, 50, 50], dtype=np.float32)  # Canvas dimensions are 600x150
        self.observation_space = Box(
            low=low_values, high=high_values, shape=(10,), dtype=np.float32)

        # Start jumping, Start ducking, Stop ducking, Do nothing - Ducking has been divided into two actions because the agent should also learn the correct ducking duration
        self.action_space = Discrete(4)

        self.actions_map = [
            (Keys.ARROW_UP, "key_down"),  # Start jumping
            (Keys.ARROW_DOWN, "key_down"),  # Start ducking
            (Keys.ARROW_DOWN, "key_up"),  # Stop ducking
            (Keys.ARROW_RIGHT, "key_down")  # Do nothing
        ]

        # Keep track of number of obstacles the agent has passed
        self.passed_obstacles = 0

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

    # Encode the obstacle type as an integer
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

    # Check if the agent has passed an obstacle
    def _passed_obstacle(self):
        obstacles = self._get_obstacles()
        if obstacles:
            # next_obstacle: [encoded_obstacle_type, obstacle_x, obstacle_y, obstacle_width, obstacle_height]
            next_obstacle = obstacles[0]
            trex_x = self.driver.execute_script(
                "return Runner.instance_.tRex.xPos")
            obstacle_x = next_obstacle[1]  # Next obstacles xpos
            obstacle_width = next_obstacle[3]  # Next obstacles width
            return obstacle_x + obstacle_width < trex_x
        else:
            return False

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
            trex_is_jumping,
            trex_is_ducking,
            game_speed,
            distance_to_next_obstacle,
            # Unpack the tuple of the first obstacle
            *(obstacles[0] if obstacles else (None, None, None, None, None))
        )

        # Set dtype for state to float32 for consistency and compatibility with the RL algorithm
        state = np.array(state, dtype=np.float32)

        # Replace NaN values with -1
        state[np.isnan(state)] = -1

        return state

    # Check if the game is over and return True or False
    def is_game_over(self):
        # Done if either Trex crashed into an obstacle or reached max score which is 99999
        # Check if Trex crashed
        crashed = self.driver.execute_script("return Runner.instance_.crashed")

        # Get the maximum score from the game
        max_score = self.driver.execute_script(
            "return Runner.instance_.distanceMeter.maxScore")
        current_score = self._get_current_score()

        return crashed or (current_score >= max_score)

    # Calculate and return the reward for the current state of the game
    def get_reward(self, done):
        # Must maintain the relative importance of different rewards so that the agent can differentiate between the various outcomes and is encouraged to learn a good policy
        reward = 0
        if done:
            # Penalize for crashing into an obstacle
            reward -= 10
        else:
            if self._passed_obstacle():
                # Reward for passing an obstacle
                reward += 0.5
                self.passed_obstacles += 1
            else:
                # Small reward for staying alive
                reward += 0.1

        current_score = self._get_current_score()
        high_score = self._get_high_score()

        if current_score > high_score:
            # Bonus reward for surpassing the high score
            reward += 1

        return reward

    # Take a step in the game environment based on the given action
    def step(self, action):

        # Take action
        # Get key and action mapping
        key, action_type = self.actions_map[action]

        # Create a new ActionChains object
        action_chains = ActionChains(self.driver)

        # Perform the key press action
        if action_type == "key_down":
            action_chains.key_down(key).perform()
        # Perform the key release action
        elif action_type == "key_up":
            action_chains.key_up(key).perform()

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
