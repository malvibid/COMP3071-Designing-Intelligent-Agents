import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from dino_environment import DinoEnvironment


class DinoDQNAgent():
    def __init__(self, env,
                 gamma=0.95,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 learning_rate=0.001,
                 batch_size=32,
                 memory_size=100000):
        self.env = env
        self.state_size = env.observation_space.shape[0]  # 10
        self.action_size = env.action_space.n  # 4
        self.hidden_sizes = [64, 128]  # number of hidden neurons for the model
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # discounting factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min  # min exploration rate
        self.epsilon_decay = epsilon_decay  # exploration decay per step
        self.batch_size = batch_size
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    # Define the DQN model architecture - This model will be used to approximate the Q-values of the agent's actions given a state.
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_sizes[1], self.action_size)
        )

        return model

    # Store agents experiences as a tuple
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Determine which action to take given a state
    def act(self, state):
        # Explore randomly or exploit given the current epsilon value
        if random.uniform(0, 1) <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state)
            action = torch.argmax(q_values).item()
            return action

    # Update the DQN model using a batch of experiences sampled from the memory
    def replay(self):
        # Check if the number of experiences (state, action, reward, next_state, done) in the memory is less than the batch size
        if len(self.memory) < self.batch_size:
            # Don't do anything since there's not enough data to create a minibatch for training
            return

        # Create minibatch from a random sample of experiences from the memory
        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            # Calculate the expected Q-value for the current state-action pair (q_target)
            # If done, - Game has ended, don't need to make predictions about future rewards
            q_target = reward
            if not done:
                # Calculate the Q-values for the next state using the DQN model, i.e., estimate future reward
                next_state = torch.tensor(next_state, dtype=torch.float32)
                q_values_next = self.model(next_state)
                # Update the target value by adding the discounted maximum Q-value of the next state to the current reward
                q_target = reward + self.gamma * \
                    torch.max(q_values_next).item()

            # Calculate the Q-values for the current state using the DQN model
            state = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state)

            # Update/Map the expected Q-value of the chosen action with the calculated target value
            q_values_expected = q_values.clone().detach()

            q_values_expected[action] = q_target

            # Note: q_values_expected is the ground truth for the action that the agent took in the current state vs q_values is the models prediction of what should happen

            # Reset the gradients of the optimizer before performing backpropagation
            self.optimizer.zero_grad()

            # Calculate the loss using the Mean Squared Error (MSE) between the current Q-values and the expected Q-values
            loss = self.loss_fn(q_values, q_values_expected)

            # Perform backpropagation to calculate the gradients of the model's parameters with respect to the loss
            loss.backward()

            # Update the model's parameters using the calculated gradients and the optimizer's learning rate
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Save model
    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    # Load a saved model
    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path))


if __name__ == "__main__":

    env = DinoEnvironment()
    agent = DinoDQNAgent(env)

    EPISODE_NUMS = 1000
    OUTPUT_DIR = "model_output/dino"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # for episode in range(EPISODE_NUMS):
    #     state = env.reset()
    #     done = False
    #     episode_reward = 0

    #     while not done:
    #         action = agent.act(state)
    #         next_state, reward, done, info = env.step(action)
    #         agent.remember(state, action, reward, next_state, done)
    #         state = next_state
    #         episode_reward += reward

    #     agent.replay()
    #     print(
    #         f"Episode {episode + 1}/{EPISODE_NUMS}, Reward: {episode_reward}, Current Score: {info['current_score']}, High Score: {info['high_score']}")

    #     if (episode + 1) % 50 == 0:
    #         model_file = os.path.join(OUTPUT_DIR, f"episode_{episode + 1}.pth")
    #         agent.save_model(model_file)
    #         print(f"Model saved after episode {episode + 1}")

    # Test model
    # agent = DinoDQNAgent(env)
    agent.load_model("model_output\dino\episode_100.pth")
    # Set agent's exploration rate (epsilon) to zero, so that it only chooses actions based on the model's predictions
    agent.epsilon = 0

    # Test loop - Play 5 game
    for episode in range(5):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward

        print(
            f"Total episode reward: {episode_reward}, Final score: {info['current_score']}, Highest score achieved: {info['high_score']}")
