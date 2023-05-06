This is my coursework submission for COMP3071 Designing Intelligent Agents (Spring 2022/2023) at the School of Computer Science, University of Nottingham Malaysia.

---

# TODO:

- [ ] Use WANDB to log model performance 
- [ ] Save useful information to state_dict
- [ ] Adjust reward strategy to encourage Agent to Duck correctly
- [ ] Move code to a notebook to preserve output



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
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install wandb
    pip install jupyterlab
    ```

# Custom Chrome Dino Environment for Reinforcement Learning
I have created a custom environment which includes necessary methods and attributes to facilitate Reinforcement Learning. This environment is a subclass of the `Env` class from the Gymnasium library.

Key method of DinoEnvironment class include:

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

# Chrome Dino DQN Agent
The DinoDQNAgent class is an implementation of a Deep Q-Network (DQN) agent for the Chrome Dino game. The agent interacts with the DinoEnvironment and learns to play the game by approximating the optimal Q-values for each state-action pair using a deep neural network.

In Q-learning, the agent iteratively updates its Q-function based on its experiences in the environment. The agent chooses actions using an exploration-exploitation strategy, balancing the need to explore new actions and exploit the current knowledge of the Q-function. The goal of the agent is to learn an optimal policy that maximizes the cumulative discounted rewards over time. As the agent interacts with the environment and updates its Q-function, the Q-values converge to the optimal Q-function, which represents the expected cumulative reward for each state-action pair when following the optimal policy. The optimal policy can then be derived from the optimal Q-function by choosing the action with the highest Q-value in each state.

**Update rule for Q-learning (Bellman Equation)**

This rule expresses the relationship between the current state-action value and the expected value of the next state-action pair, taking into account the immediate reward received for performing the current action

$Q(s,a)$<sub>$new$</sub>  ← $Q(s,a)$<sub>$old$</sub> + $α [r + γ * max_a' Q(s',a') -  Q(s,a)$<sub>$old$</sub>

where:
- $s$ is the current state.
- $a$ is the action taken in the current state.
- $r$ is the immediate reward received after taking action a in state $s$. It's a crucial part of the update because it provides direct feedback about the action's value.
- $s'$ is the next state (resulting from taking action $a$ in state $s$).
- $a'$ is the action taken in the next state $s'$.
- $α$ (alpha) is the learning rate, which controls the step size of the update. A higher learning rate makes the agent more sensitive to recent experiences, while a lower learning rate makes the agent more conservative, relying more on its accumulated knowledge.
- $γ$ (gamma) is the discount factor, which determines the importance of future rewards relative to immediate rewards.
- $max_a' Q(s',a')$ represents the maximum Q-value for the next state $s'$ over all possible actions $a'$.
- $γ * max_a' Q(s',a')$ represents the estimated cumulative reward from the next state ($s'$) onward, assuming the agent follows the optimal policy. By taking the maximum Q-value over all possible actions ($a'$) in the next state ($s'$), we assume that the agent will choose the action that leads to the highest expected future rewards. 
- $[r + γ * max_a' Q(s',a') - Q(s,a)]$ calculates the difference between the updated Q-value (taking into account the immediate reward and the maximum future reward) and the current Q-value. This difference, also known as the temporal difference (TD) error, measures how much our current estimate is off from the new information we just received.

The update rule combines the current Q-value with the weighted TD error to produce a new Q-value for the state-action pair. By incorporating both immediate rewards and future rewards, the agent updates its estimates to better reflect the true value of each action in a given state. 

Note: This equation can be read as “Update the Q-value for state $s$ and action $a$ by assigning it the current Q-value plus the weighted temporal difference error.”

Based on the above theory, the key methods of the DinoDQNAgent class include:

- `__init__` : Initializes the agent's parameters, such as the environment, state and action sizes, discount factor (gamma), exploration rate (epsilon) and its decay, learning rate, batch size, and memory size. It also creates the DQN model, optimizer, and loss function.
- `_build_model`: Builds the DQN model architecture using PyTorch, which is a simple feed-forward neural network with two hidden layers and ReLU activation functions.
- `remember`: Appends a new experience (state, action, reward, next_state, done) to the agent's memory, which is used for training.
- `act`: Determines the action to take given a state, following the epsilon-greedy strategy. It either selects a random action (exploration) or the action with the highest predicted Q-value (exploitation of gathered knowledge) based on the current epsilon value.
- `replay`: This method is called periodically (after each episode in this case) during training to update/train the DQN model using a batch of experiences randomly sampled from the memory. It updates the model's weights to minimize the mean squared error (MSE) between the target Q-values (ground truth) and the predicted Q-values for a set of state-action pairs. The target Q-values are calculated by adding the immediate reward and the discounted maximum Q-value of the next state. The epsilon value is also decayed in this method. Experience replay helps break the correlation between consecutive samples, leading to better learning.

- `save_model` and `load_model`: Functions to save and load the trained DQN model's weights to/from a .pth file.



# Updates Log

- **20/04/2023** - Used manual approach to create the DinoEnvironment, with MSS library to capture screenshots of the game and PyTessaract to perform OCR on the captured screenshots to detect game over screen. This was not a very efficient method.
- **24/04/2023** - Used an automatic approach with Selenium to create the DinoEnvironment. Discovered the `Runner` JavaScript object of the Chrome Dino game. Used Selenium functions with the `Runner` object to capture the screenshots the game and get the game over state. This is a more efficient and compact solution than the manual approach.
- **26/04/2023** - Migrated from OpenAI [Gym](https://www.gymlibrary.dev/) library which is no longer being actively maintained to [Gymnasium](https://gymnasium.farama.org/api/env/) which is a fork of OpenAI Gym and is being actively maintained and improved by the Farama Foundation. 
- **03/05/2023** - I realised that the approach of taking screenshots as observations is adding an unneccesary layer of complexity (as with this method I'd have to define and train a Convolutional Neural Network to understand the state of the game from the screenshots captured) when I can easily define the state of game at a given step directly and compactly by accessing properties of the `Runner` object to extract information such as obstacles on the screen, Trex, game speed, etc. This direct numerical representation is more informative than the screenshots while being computationally efficient. I am hoping this will help the agent learn and perform better. Thus, I replaced the old method to get observations with the new one.
- **03/05/2023** - Updated the reward strategy such that the agent can differentiate between the various outcomes and is encouraged to learn a good policy. This is done by maintaining the relative importance of different rewards: 
    - A reward of 0.1 for staying alive at each step.
    - A reward of 0.5 for passing an obstacle.
    - A penalty of -10 for crashing into an obstacle.
    - A bonus reward of 1 for surpassing the high score at each step.
- **04/05/2023** - Modified the action space by separating the `Duck` action into two separate actions, one for `Start Ducking` and another for `Stop Ducking`. This way, the agent can learn the appropriate duration to hold the ducking action. This is a more dynamic approach to decide ducking duration than my previous approach of using time sleep, which can slow down the training process. With this modification, the agent will need to learn the appropriate sequence of actions, i.e., `Start Ducking` before `Stop Ducking`. As the agent interacts with the environment, it will learn the most effective action sequences, including the correct order of starting and stopping the duck action, as well as for how long to keep ducking. This will be achieved through the agent receiving higher rewards for the correct sequence of actions, which will encourage the agent to learn and perform the correct action sequence in the future.
- **06/05/2023** - Implemented the `DinoDQNAgent` class from scratch using pure PyTorch and trained the model for a 100 episodes to test its performance. The model performance was relatively good. Tested the trained agent for 5 episodes and achieved a high score of `1114`. In such a short time, I observed the agent has learnt how to avoid unnecessarily jumping (i.e., do nothing) when there are no obstacles on the horizon. And it has also learnt well to jump over the Cacti. However, it also tries to jump over the Pterodactyl even if it is higher up towards the sky, consequently crashing. This suggests that the agent hasn't learnt to duck properly yet. 
    ```
    # Test Log
        Total episode reward: 667.2000000000077, Final score: 388, Highest score achieved: 388
        Total episode reward: 56.200000000000415, Final score: 346, Highest score achieved: 388
        Total episode reward: 58.50000000000027, Final score: 368, Highest score achieved: 388
        Total episode reward: 165.99999999999972, Final score: 467, Highest score achieved: 467
        Total episode reward: 954.700000000013, Final score: 1114, Highest score achieved: 1114
    ```

