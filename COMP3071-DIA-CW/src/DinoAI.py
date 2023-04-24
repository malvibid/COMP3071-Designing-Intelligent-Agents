import base64
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Environment Components
from gym import Env
from gym.spaces import Box, Discrete

# Selenium for automatically loading and play the game
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
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

        # driver.maximize_window()
        return driver

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
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "runner-canvas")))

        # Start game
        self.driver.find_element(By.TAG_NAME, "body").send_keys(Keys.SPACE)

        return self.get_next_observation()

    # Get the screenshot of the game and return it as the observation
    def get_next_observation(self):

        # Get the image and convert it to grayscale
        image = cv2.cvtColor(self._get_image(), cv2.COLOR_BGR2GRAY)

        # Crop irrelavant image part out - from 150x600 pixels to 150x150 pixels square
        image = image[:150, :150]  # [Row (Height), Column (Width)]

        # Resize to ensure that the input size of the image is consistent across all observations - necessary for training the ML model
        image = cv2.resize(image, (self.screen_width, self.screen_height))
        return image

    # Calculate and return the reward for the current state of the game
    def get_reward(self):
        pass

    # Check if the game is over and return True or False
    def is_game_over(self):
        return self.driver.execute_script("return Runner.instance_.crashed")

    # Take a step in the game environment based on the given action
    def step(self, action):
        pass

    def close(self):
        # Close the game environment and the driver
        self.driver.quit()


env = DinoEnvironment()

env.reset()

# image = env._get_image()
# print(image.shape)

# Show image using OpenCV
# cv2.imshow('Dino Game', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Show image using MatPlotLib
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# plt.imshow(image)
# plt.show()

obs = env.get_next_observation()
print(obs.shape)

# Show image using MatPlotLib
obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
plt.imshow(obs)
plt.show()
