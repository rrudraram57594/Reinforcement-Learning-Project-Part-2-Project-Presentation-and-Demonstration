## Double Deep Q-Network for CartPole Balancing

### Project Overview

The Double Deep Q-Network (DDQN) project demonstrates the application of deep reinforcement learning to the CartPole-v1 environment. The goal is to design an intelligent agent that learns the control actions required to balance an upright pole on a cart that slides across the floor with repeated interactions with the environment. The trained agent acquires an efficient balancing strategy without using control rules manually programmed by human agents.

### Objectives

The main goal is to apply and test a DDQN agent to the CartPole balancing task. Other goals include reward analysis for training, tracking training loss, examining changes in exploration rate, assessing the learned policy, and comparing the trained agent with a random-action agent.

### Technologies Used

- Programming Language: Python
- Reinforcement Learning Environment: Gymnasium
- Deep Learning Framework: PyTorch
- Numerical Computation: NumPy
- Data Visualization: Matplotlib

### Algorithm and Methodology

The algorithm employed is Double Deep Q-Network that is used to estimate action values and enhance decision making. A policy network is used to choose actions, and a target network is used to compute less volatile target values. Experience replay saves past transitions and provides mini-batches randomly sampled from the saved experiences for training. The epsilon-greedy approach is a trade-off between exploration and exploitation, and the Smooth L1 loss function quantifies the distance between forecasted and real Q values.

### Environment Description

The CartPole-v1 environment offers the following four states: cart position, cart velocity, pole angle, and pole angular velocity. The agent chooses one of two actions: move left, or move right. A reward of one point is provided for every successful time step, with a maximum episode reward of 500.

### Training Configuration

The neural network has 2 hidden layers with 128 neurons in each layer and ReLU activation functions. The Adam optimizer has a learning rate of 0.001, and a discount factor of 0.99. The replay buffer can store up to 100,000 experiences and the mini-batch size is 64. Training episodes up to 1000, epsilon is decremented from 1.0 to at least 0.01. The target network is updated after 10 training episodes.

### Training and Evaluation

Initialization of environment and network setup is the initial step of the training process. The agent gathers experience by taking an epsilon-greedy action and adding transitions to the replay buffer. After sufficient experience collection, randomly sampled mini-batches are used to update the policy network through the Double DQN learning method. Evaluation is performed periodically but not randomly exploring the learned policy and the best checkpoint is kept for final evaluation.

### Results

Agent shows enhanced ability to balance during training. The average evaluation reward reaches 305.70 at episode 100, 402.60 at episode 300, and 500.00 at episode 800. The optimal checkpoint is recorded at episode 800 to be reviewed at the end. The average reward of the random-action agent is 31.05, and the standard deviation is 16.25. The trained DDQN produces an average reward of 500.00 with a standard deviation of 0.00 over 100 evaluation episodes, compared to the trained DDQN. The results show successful learning of the CartPole balancing task under the evaluation conditions tested.

### Key Features

Double Deep Q-Network implementation
Policy and target neural networks
Experiential Replay for training stability.
Epsilon-greedy exploration strategy
Training reward and moving-average visualization
Training loss analysis
Exploration rate monitoring
Periodic policy evaluation
Best-model checkpoint selection
Random agent and trained agent performance comparison

### Limitations and Future Improvements

The current implementation only uses the simulated CartPole-v1 environment. In reality, control systems face other complexities such as sensor noise, external disturbances, and safety requirements. The future plans involve comparisons with standard DQN and Proximal Policy Optimization (PPO), further testing with varying random seeds, tuning hyperparameters, and testing in more complex control environments.

### Conclusion

The project illustrates the learning of the control policies with environmental interaction by using Double Deep Q-Networks. The learning process is enhanced by experience replay, separate policy and target networks, and epsilon-greedy exploration. The final results of the evaluation demonstrate that the trained agent fares significantly better than the baseline that attempts random actions, and achieves the highest possible average episode reward of 500.
