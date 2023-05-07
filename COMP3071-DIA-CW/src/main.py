import os
import wandb

from dino_environment import DinoEnvironment
from dino_agent import DinoDQNAgent


def train(agent, env, episodes, model_output_dir, save_interval=10, log_to_wandb=False, render=False):

    if log_to_wandb:
        wandb.init(project='chrome_dino_rl_agent', name='train_run')

    total_rewards = []
    total_scores = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_loss = []

        while not done:
            if render:
                env.render(mode='human')

            # Use agent to predict action
            action = agent.act(state)

            # Take a step in the environment
            next_state, reward, done, info = env.step(action)

            # Remember agents experience after every step
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

        # Train/Update the model every episode
        loss = agent.replay()
        episode_loss.append(loss)

        total_rewards.append(episode_reward)
        total_scores.append(info["current_score"])

        # Calculate overall training metrics
        mean_episode_loss = sum(episode_loss) / len(episode_loss)
        mean_reward = sum(total_rewards) / len(total_rewards)
        mean_score = sum(total_scores) / len(total_scores)

        # Log metrics
        print(
            f"Episode {episode + 1}/{episodes}, Highest Score: {info['high_score']}, Episode Score: {info['current_score']}, Episode Reward: {episode_reward:.4f}, Episode Epsilon: {agent.epsilon:.4f}, Episode Loss: {loss:.4f}, Mean Score: {mean_score:.4f}, Mean Reward {mean_reward:.4f}")

        if log_to_wandb:
            wandb.log({
                "episode": (episode + 1)/episodes,
                "highest_score": info["high_score"],
                "episode_score": info["current_score"],
                "episode_reward": episode_reward,
                "episode_epsilon": agent.epsilon,
                "episode_loss": loss,
                "mean_loss": mean_episode_loss,
                "mean_reward": mean_reward,
                "mean_current_score": mean_score
            })

        # Save the model every save_interval episodes
        if (episode + 1) % save_interval == 0:
            model_name = f"dino_dqn_episode_{episode + 1}.pth"
            agent.save_model(model_name, model_output_dir, log_to_wandb)
            print(f"Model saved after episode {episode + 1}")

    # Finish wandb logging
    if log_to_wandb:
        wandb.finish()


def test(agent, env, episodes, model_path, log_to_wandb=False, older_model=False, render=False):

    if log_to_wandb:
        wandb.init(project='chrome_dino_rl_agent', name='test_run')

    total_rewards = []
    total_scores = []

    agent.load_model(model_path, older_model, for_training=False)

    # Set exploration rate (epsilon) to 0 to only choose actions based on the model's predictions (exploit its knowledge)
    agent.epsilon = 0

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            if render:
                env.render(mode='human')

            # Use agent to predict action
            action = agent.act(state)

            # Take a step in the environment
            next_state, reward, done, info = env.step(action)

            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)
        total_scores.append(info["current_score"])

        # Calculate overall training metrics
        mean_reward = sum(total_rewards) / len(total_rewards)
        mean_score = sum(total_scores) / len(total_scores)

        # Log metrics
        print(
            f"Episode {episode + 1}/{episodes}, Highest Score: {info['high_score']}, Episode Score: {info['current_score']}, Episode Reward: {episode_reward:.4f}, Episode Epsilon: {agent.epsilon:.4f}, Mean Score: {mean_score:.4f}, Mean Reward {mean_reward:.4f}")

        if log_to_wandb:
            wandb.log({
                "episode": (episode + 1)/episodes,
                "highest_score": info["high_score"],
                "episode_score": info["current_score"],
                "episode_reward": episode_reward,
                "episode_epsilon": agent.epsilon,
                "mean_reward": mean_reward,
                "mean_current_score": mean_score
            })

    if log_to_wandb:
        # Finish wandb logging
        wandb.finish()


if __name__ == "__main__":

    env = DinoEnvironment()
    agent = DinoDQNAgent(env)

    TRAIN_EPISODES = 500
    TEST_EPISODES = 50
    OUTPUT_DIR = "trained_models/"
    MODEL_LOAD_PATH = "best_trained_models\episode_100.pth"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Train Model
    # train(agent, env, TRAIN_EPISODES, OUTPUT_DIR, log_to_wandb=True)

    # Test Model
    # test(agent, env, TEST_EPISODES, MODEL_LOAD_PATH, log_to_wandb=True)

    # Test Best Model
    # test(agent, env, TEST_EPISODES, MODEL_LOAD_PATH, log_to_wandb=False, older_model=True)
