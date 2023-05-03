This is my coursework submission for COMP3071 Designing Intelligent Agents (Spring 2022/2023) at the School of Computer Science, University of Nottingham Malaysia.

---

# Setup Environment

1. Create a virtual environment and activate it.
    ```
    python -m venv venv 
    ```
    ```
    venv/Scripts/activate
    ```

2. Install Dependencies
    ```
    pip install selenium 
    pip install webdriver-manager 
    pip install gymnasium 
    pip install numpy 
    pip install matplotlib 
    pip install opencv-python
    pip install tianshou
    pip install torch -f https://download.pytorch.org/whl/cu118/torch_stable.html
    ```

# Custom Chrome Dino Environment for Reinforcement Learning
I have created a custom environment which includes necessary methods and attributes to facilitate Reinforcement Learning. This environment is a subclass of the `Env` class from the Gymnasium library.

- `__init__`: Initializes the environment, including the screen dimensions, observation space, and action space.
- `_create_driver`: Sets up the ChromeDriver instance for controlling the browser.
- Various helper methods for extracting game state information, such as obstacles, game speed, and T-Rex position.
- `reset`: Resets the environment and starts a new game.
- `get_observation`: Returns the current state of the game.
- `is_game_over`: Determines if the game is over.
- `get_reward`: Returns the reward for the current state of the game.
- `step`: Takes an action and returns the resulting observation, reward, whether the game is done, and additional information.
- `render`: Visualizes the game in the specified mode.
- `close`: Closes the game environment and the driver.

# Updates Log

- **20/04/2023** - Used manual approach to create the DinoEnvironment, with MSS library to capture screenshots of the game and PyTessaract to perform OCR on the captured screenshots to detect game over screen. This was not a very efficient method.
- **24/04/2023** - Used an automatic approach with Selenium to create the DinoEnvironment. Discovered the `Runner` JavaScript object of the Chrome Dino game. Used Selenium functions with the `Runner` object to capture the screenshots the game and get the game over state. This is a more efficient and compact solution than the manual approach.
- **26/04/2023** - Migrated from OpenAI [Gym](https://www.gymlibrary.dev/) library which is no longer being actively maintained to [Gymnasium](https://gymnasium.farama.org/api/env/) which is a fork of OpenAI Gym and is being actively maintained and improved by the Farama Foundation. Also planning to use the [Tianshou](https://tianshou.readthedocs.io/en/master/index.html) library instead of the originally planned [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) since Tianshou is based on pure PyTorch and seems like a cleaner framework.
- **03/05/2023** - I realised that the approach of taking screenshots as observations is adding an unneccesary layer of complexity (as with this method I'd have to define and train a Convolutional Neural Network to understand the state of the game from the screenshots captured) when I can easily define the state of game at a given step directly and compactly by accessing properties of the `Runner` object to extract information such as obstacles on the screen, Trex, game speed, etc. This direct numerical representation is more informative than the screenshots while being computationally efficient. I am hoping this will help the agent learn and perform better. Thus, I replaced the old method to get observations with the new one.