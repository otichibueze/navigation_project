# Train an Agent to Collect Bananas

_ For this project, we have a trained agent that navigates an enviroment and collect bananas! in a large, square world.

_ Below is trained agent

- GIF


### Enviroment State  Action Reward
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.


### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.
    
    

### Learning Algorithm
**Goal**
The goal of the agent is to decide which actions to take within the environment to maximize reward. Since the effects of possible actions aren't known in advance, the optimal policy must be discovered by interacting with the environment and recording observations. Therefore, the agent "learns" the policy through a process of trial-and-error that iteratively maps various environment states to the actions that yield the highest reward. This type of algorithm is called Q-Learning.


**Epsilon Greedy Algorithm**
Epsilon greedy policy uses stochastic probability when picking state action reward that would have the highest reward probability but sometimes a policy that does not produce the highest reward also has a probability of being picked. This is known as the exploration vs. exploitation dilemma. This algorithm allows the agent to systematically manage the exploration vs. exploitation trade-off. [More on Epsilon greedy policy](https://medium.com/analytics-vidhya/the-epsilon-greedy-algorithm-for-reinforcement-learning-5fe6f96dc870)


**Experience Replay**
Here we store experience of agent gameplay. The previous gameplay experience is stored in a tuple as it interacts with the environment {S, A, R, S}. SO we can learn from previous experience replay and also sample from this experience replay in random

This tuple contains the state of the environment , the action  taken from state , the reward  given to the agent at time  as a result of the previous state-action pair , and the next state of the environment . This tuple indeed gives us a summary of the agent’s experience at time.

- This can be thought of as a supervised learning 

When the agent interacts with the environment, the sequence of experience tuples can be highly correlated.The naive Q-learning algorithm that learns from each of these experience tuples in sequential order runs the risk of getting swayed by the effects of this correlation. By instead keeping track of a replay buffer and using experience replay to sample from the buffer at random, we can prevent action values from 
oscillating or diverging catastrophically.



**Double Deep Q-Network (DDQN)**
Q-learning is a model-free reinforcement learning algorithm to learn a policy telling an agent what action to take under what circumstances. The neural network architecture used for this project can be found [here](link) in the model.py file. here is one major issue with Q-learning that we need to deal with: over-estimation bias, which means that the Q-values learned are actually higher than they should be. Mathematically, maxaQ(st+1, a) converges to E(maxaQ(st+1, a)), which is higher than maxa(E(Q(st+1, a)), the true Q-value. To get more accurate Q-values, we use something called double Q-learning. In double Q-learning we have two Q-tables: one which we use for taking actions, and another specifically for use in the Q-update equation. The double Q-learning update equation is:

Q*(st, at)←Q*(st, at) + α(rt+1 + γmaxaQT(st+1, a) - Q*(st, at))

where Q* is the Q table that gets updated, and QT is the target table. QT copies the values of Q* every n steps.
[More](https://arxiv.org/abs/1509.06461)


**Dueling Network Architectures**
Dueling networks utilize two streams: one that estimates the state value function V(s), and another that estimates the advantage for each action A(s,a). These two values are then combined to obtain the desired Q-values. The neural network architecture used for this project can be found [here](link) in the dueling_model.py file. The dueling architecture shares the same input-output interface with the standard DQN architecture, the training process is identical. We define the loss of the model as the mean squared error:


[Image]()

[More](https://arxiv.org/abs/1511.06581)



**Prioritized Experience Replay**
Prioritized Experience Replay is a type of experience replay in reinforcement learning where we In more frequently replay transitions with high expected learning progress, as measured by the magnitude of their temporal-difference (TD) error. This prioritization can lead to a loss of diversity, which is alleviated with stochastic prioritization, and introduce bias, which can be corrected with importance sampling.

The stochastic sampling method interpolates between pure greedy prioritization and uniform random sampling. The probability of being sampled is ensured to be monotonic in a transition's priority, while guaranteeing a non-zero probability even for the lowest-priority transition. 

[More](https://arxiv.org/abs/1511.05952)




### Run Experiments
The best performing agents were able to solve the environment in 300-500 episodes. While this set of agents included ones that utilized Double DQN, Dueling DQN and Double DQN with prioritized replay, the top performing agent was Double DQN with replay buffer.

**View Agents Performance**

-image

[Notebook]()


### Future Improvements
Implement Rainbow: Combining Improvements in DQN — This approach is explained here in this [research paper](). 
Implement A Distributional Perspective on Reinforcement Learning  — This approach is explained here in this [research paper](). 

Train agent and compare performance.


#### Instructions
Follow the instructions in Navigation.ipynb to get started with training your own agent!