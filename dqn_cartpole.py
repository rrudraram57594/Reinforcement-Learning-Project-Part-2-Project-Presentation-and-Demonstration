# A deep Q-network was created to learn the problem of balancing a pole using reinforcement learning.
# The trained agent's performance was compared to the random agent using all the three measures of performance, reward comparison and training visualisations.

# importing required libraries
import os
import random
from collections import deque

# importing reinforcement learning environment
import gymnasium as gym

# importing numerical and visualization libraries
import numpy as np
import matplotlib.pyplot as plt

# importing deep learning libraries
import torch
import torch.nn as nn
import torch.optim as optim


# setting random seeds and environment
SEED = 42
ENV_NAME = "CartPole-v1"

# defining training limits
NUM_EPISODES = 1000
MAX_STEPS = 500

# defining learning parameters
GAMMA = 0.99
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MEMORY_SIZE = 100000

# defining exploration parameters
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# defining target network update settings
TARGET_UPDATE_FREQUENCY = 10
WARMUP_STEPS = 1000

# defining evaluation settings
EVALUATION_FREQUENCY = 100
EVALUATION_EPISODES = 10
FINAL_EVALUATION_EPISODES = 100

# defining success and stopping conditions
SOLVED_REWARD = 475
EARLY_STOPPING_CHECKPOINTS = 3

# setting the model checkpoint location
CHECKPOINT_PATH = "best_dqn_cartpole.pth"

# selecting the available computing device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)


# initializing random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# initializing GPU random seeds
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# creating the CartPole environment
env = gym.make(ENV_NAME)
env.action_space.seed(SEED)

# retrieving state and action dimensions
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print("State size:", state_size)
print("Action size:", action_size)


# defining the random agent evaluation process
def evaluate_random_agent(environment, episodes=20):
    # storing rewards from random actions
    rewards = []

    # running evaluation episodes
    for episode in range(episodes):
        state, info = environment.reset(seed=SEED + episode)
        total_reward = 0

        # selecting random actions until episode completion
        for step in range(MAX_STEPS):
            action = environment.action_space.sample()

            state, reward, terminated, truncated, info = (
                environment.step(action)
            )

            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)

    return rewards


# evaluating the random baseline agent
print("\nEvaluating random agent...")

random_rewards = evaluate_random_agent(env, episodes=20)

# calculating baseline performance statistics
random_mean = np.mean(random_rewards)
random_std = np.std(random_rewards)

print(f"Random Agent Mean Reward: {random_mean:.2f}")
print(f"Random Agent Standard Deviation: {random_std:.2f}")


# defining the deep Q-network architecture
class DQN(nn.Module):

    # initializing neural network layers
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    # generating action value predictions
    def forward(self, state):
        return self.network(state)


# defining the experience replay buffer
class ReplayBuffer:

    # initializing memory storage
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    # storing an experience
    def push(self, state, action, reward, next_state, terminated):
        self.memory.append(
            (state, action, reward, next_state, terminated)
        )

    # sampling a random training batch
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)

        states, actions, rewards, next_states, terminated = zip(*batch)

        # converting states into tensors
        states = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device=DEVICE
        )

        # converting actions into tensors
        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=DEVICE
        ).unsqueeze(1)

        # converting rewards into tensors
        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=DEVICE
        )

        # converting next states into tensors
        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=DEVICE
        )

        # converting termination flags into tensors
        terminated = torch.tensor(
            terminated,
            dtype=torch.float32,
            device=DEVICE
        )

        return states, actions, rewards, next_states, terminated

    # retrieving the current memory size
    def __len__(self):
        return len(self.memory)


# creating policy and target networks
policy_net = DQN(state_size, action_size).to(DEVICE)
target_net = DQN(state_size, action_size).to(DEVICE)

# synchronizing initial network parameters
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# configuring the optimizer
optimizer = optim.Adam(
    policy_net.parameters(),
    lr=LEARNING_RATE
)

# defining the loss function
loss_function = nn.SmoothL1Loss()

# initializing experience replay memory
replay_buffer = ReplayBuffer(MEMORY_SIZE)


# defining the action selection process
def select_action(state, epsilon):

    # exploring through random actions
    if random.random() < epsilon:
        return env.action_space.sample()

    # converting the current state into a tensor
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=DEVICE
    ).unsqueeze(0)

    # selecting the action with the highest predicted value
    with torch.no_grad():
        q_values = policy_net(state_tensor)

    return q_values.argmax(dim=1).item()


# defining one training update
def train_step():

    # checking whether sufficient experiences are available
    if len(replay_buffer) < BATCH_SIZE:
        return None

    # sampling experiences from replay memory
    states, actions, rewards, next_states, terminated = (
        replay_buffer.sample(BATCH_SIZE)
    )

    # calculating current action values
    current_q_values = policy_net(states).gather(1, actions).squeeze(1)

    # calculating target action values
    with torch.no_grad():

        next_actions = policy_net(next_states).argmax(
            dim=1,
            keepdim=True
        )

        next_q_values = target_net(next_states).gather(
            1,
            next_actions
        ).squeeze(1)

        target_q_values = rewards + GAMMA * next_q_values * (
            1 - terminated
        )

    # calculating the training loss
    loss = loss_function(current_q_values, target_q_values)

    # updating network parameters
    optimizer.zero_grad()
    loss.backward()

    # limiting gradient values
    torch.nn.utils.clip_grad_norm_(
        policy_net.parameters(),
        max_norm=1.0
    )

    optimizer.step()

    return loss.item()


# defining the trained agent evaluation process
def evaluate_agent(network, episodes=10):

    # storing evaluation rewards
    evaluation_rewards = []

    # running evaluation episodes
    for episode in range(episodes):

        state, info = env.reset(seed=SEED + 10000 + episode)
        total_reward = 0

        # selecting actions using the trained network
        for step in range(MAX_STEPS):

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=DEVICE
            ).unsqueeze(0)

            with torch.no_grad():
                action = network(state_tensor).argmax(dim=1).item()

            state, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward

            if terminated or truncated:
                break

        evaluation_rewards.append(total_reward)

    return evaluation_rewards


# initializing training history lists
episode_rewards = []
episode_losses = []
epsilon_history = []

# initializing evaluation history lists
evaluation_episodes = []
evaluation_means = []

# setting initial exploration rate
epsilon = EPSILON_START
global_step = 0

# initializing best performance tracking
best_evaluation_reward = -float("inf")
solved_checkpoint_count = 0

print("\nStarting DQN training...\n")


# training the DQN agent across episodes
for episode in range(1, NUM_EPISODES + 1):

    # resetting the environment
    state, info = env.reset(seed=SEED + episode)

    # initializing episode statistics
    total_reward = 0
    losses = []

    # running actions within the episode
    for step in range(MAX_STEPS):

        # selecting an action using epsilon-greedy exploration
        action = select_action(state, epsilon)

        # executing the selected action
        next_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        # storing the experience in replay memory
        replay_buffer.push(
            state,
            action,
            reward,
            next_state,
            terminated
        )

        # updating the current state and reward
        state = next_state
        total_reward += reward
        global_step += 1

        # training the network after the warmup period
        if global_step >= WARMUP_STEPS:
            loss = train_step()

            if loss is not None:
                losses.append(loss)

        # ending the episode when required
        if terminated or truncated:
            break

    # recording episode rewards
    episode_rewards.append(total_reward)

    # recording average training loss
    if losses:
        episode_losses.append(np.mean(losses))
    else:
        episode_losses.append(np.nan)

    # reducing the exploration rate
    epsilon = max(
        EPSILON_MIN,
        epsilon * EPSILON_DECAY
    )

    epsilon_history.append(epsilon)

    # updating the target network periodically
    if episode % TARGET_UPDATE_FREQUENCY == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # displaying progress every ten episodes
    if episode % 10 == 0:

        recent_mean = np.mean(episode_rewards[-10:])

        print(
            f"Episode {episode:4d} | "
            f"Reward: {total_reward:6.1f} | "
            f"Mean Reward (10): {recent_mean:6.2f} | "
            f"Epsilon: {epsilon:.3f}"
        )

    # evaluating the agent at regular intervals
    if episode % EVALUATION_FREQUENCY == 0:

        evaluation_rewards = evaluate_agent(
            policy_net,
            episodes=EVALUATION_EPISODES
        )

        evaluation_mean = np.mean(evaluation_rewards)

        evaluation_episodes.append(episode)
        evaluation_means.append(evaluation_mean)

        print(
            f"\nEvaluation at Episode {episode}: "
            f"Mean Reward = {evaluation_mean:.2f}\n"
        )

        # saving the best-performing model
        if evaluation_mean > best_evaluation_reward:

            best_evaluation_reward = evaluation_mean

            torch.save(
                policy_net.state_dict(),
                CHECKPOINT_PATH
            )

            print(
                f"Best model saved with reward "
                f"{best_evaluation_reward:.2f}"
            )

        # tracking repeated successful evaluations
        if evaluation_mean >= SOLVED_REWARD:
            solved_checkpoint_count += 1
        else:
            solved_checkpoint_count = 0

        # stopping training after repeated success
        if solved_checkpoint_count >= EARLY_STOPPING_CHECKPOINTS:

            print(
                "\nEarly stopping triggered after "
                "repeated successful evaluations."
            )

            break


# loading the best saved model
if os.path.exists(CHECKPOINT_PATH):

    policy_net.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    policy_net.eval()

    print("\nBest saved model loaded successfully.")


# evaluating the trained DQN agent
print("\nEvaluating trained DQN agent...")

final_rewards = evaluate_agent(
    policy_net,
    episodes=FINAL_EVALUATION_EPISODES
)

# calculating final performance statistics
dqn_mean = np.mean(final_rewards)
dqn_std = np.std(final_rewards)

# measuring improvement over the random baseline
absolute_improvement = dqn_mean - random_mean

# calculating percentage improvement
if random_mean != 0:
    percentage_improvement = (
        absolute_improvement / random_mean
    ) * 100
else:
    percentage_improvement = float("nan")


# displaying final performance results
print("\nFINAL PERFORMANCE SUMMARY")

print(f"Random Agent Mean Reward: {random_mean:.2f}")
print(f"Random Agent Std. Deviation: {random_std:.2f}")

print(f"DQN Agent Mean Reward: {dqn_mean:.2f}")
print(f"DQN Agent Std. Deviation: {dqn_std:.2f}")

print(f"Absolute Improvement: {absolute_improvement:.2f}")
print(f"Percentage Improvement: {percentage_improvement:.2f}%")

print(f"Best Checkpoint Reward: {best_evaluation_reward:.2f}")

# checking whether the target reward was reached
if dqn_mean >= SOLVED_REWARD:
    print("Result: The trained agent reached the CartPole target.")
else:
    print("Result: The trained agent did not reach the target mean reward.")


# visualizing episode rewards and moving averages
plt.figure(figsize=(12, 5))

plt.plot(
    episode_rewards,
    alpha=0.4,
    label="Episode Reward"
)

# calculating and displaying the moving average
if len(episode_rewards) >= 20:

    moving_average = np.convolve(
        episode_rewards,
        np.ones(20) / 20,
        mode="valid"
    )

    plt.plot(
        range(19, len(episode_rewards)),
        moving_average,
        label="20-Episode Moving Average",
        linewidth=2
    )

# displaying the target reward threshold
plt.axhline(
    y=SOLVED_REWARD,
    color="red",
    linestyle="--",
    label="Target Reward"
)

# formatting the reward chart
plt.title("DQN Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# visualizing training loss
plt.figure(figsize=(12, 5))

plt.plot(
    episode_losses,
    label="Average Training Loss"
)

# formatting the loss chart
plt.title("DQN Training Loss")
plt.xlabel("Episode")
plt.ylabel("Smooth L1 Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# visualizing exploration rate decay
plt.figure(figsize=(12, 5))

plt.plot(
    epsilon_history,
    label="Epsilon"
)

# formatting the exploration chart
plt.title("Exploration Rate Decay")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# visualizing evaluation performance
if evaluation_episodes:

    plt.figure(figsize=(12, 5))

    plt.plot(
        evaluation_episodes,
        evaluation_means,
        marker="o",
        label="Evaluation Mean Reward"
    )

    # displaying the target reward threshold
    plt.axhline(
        y=SOLVED_REWARD,
        color="red",
        linestyle="--",
        label="Target Reward"
    )

    # formatting the evaluation chart
    plt.title("DQN Evaluation Performance")
    plt.xlabel("Training Episode")
    plt.ylabel("Mean Evaluation Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# comparing random and trained agent performance
plt.figure(figsize=(8, 5))

agent_names = ["Random Agent", "Trained DQN"]
agent_means = [random_mean, dqn_mean]
agent_stds = [random_std, dqn_std]

# displaying average rewards with standard deviations
plt.bar(
    agent_names,
    agent_means,
    yerr=agent_stds,
    capsize=8
)

# displaying the target reward threshold
plt.axhline(
    y=SOLVED_REWARD,
    color="red",
    linestyle="--",
    label="Target Reward"
)

# formatting the comparison chart
plt.title("Random Agent vs. Trained DQN")
plt.ylabel("Mean Episode Reward")
plt.legend()
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# closing the environment
env.close()

print("\nTraining and evaluation completed.")