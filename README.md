We have developed a Python script, `CVor_grad_mean_var.py`, specifically designed for the selection of the F function in computational processes.

Here are some illustrative examples demonstrating the usage of the F function:

```python
seed = 42

# ... [Additional relevant code]

# Calculation of the F value
F_value = self.F(policy_loss_bak) + 0.01 * policy_loss_bak      # This function resembles the residual learning concept used in ResNet architecture.
```

**Results Visualization:**

![Resulting Graph](https://github.com/Anonymous034/SAC-CVOR/assets/110434246/57826c25-ba58-4130-8403-b554722d1f26)

```python
seed = 0

# ... [Additional relevant code]

# Calculation of the F value
F_value = 0.01 * policy_loss_bak
```

**Results Visualization:**

![image](https://github.com/Anonymous034/SAC-CVOR/assets/110434246/c0778cc9-4cd1-4667-b652-ca8d0ba8436c)


```python
seed = 123

# ... [Additional relevant code]

# Calculation of the F value
F_value = self.F(policy_loss_bak) + 0.15 * policy_loss_bak
```

**Results Visualization:**

![image](https://github.com/Anonymous034/SAC-CVOR/assets/110434246/312165fe-495d-4dd2-b368-1ce2fc672ae4)

### Correctness of CVor

The formulation of the CVor (Control Variate Operator) can be articulated as follows:

![image](https://github.com/Anonymous034/SAC-CVOR/assets/110434246/b08f4a1b-50e2-4670-a89d-1b232e2de8c1)

<span style="color: red; font-weight: bold;">Note that we need to use $\bot(\mathcal{L}_{\theta})$ to eliminate the gradient information of \mathcal{L}_{\theta}.</span>

![image](https://github.com/Anonymous034/SAC-CVOR/assets/110434246/b60068c6-a1a5-4533-92ec-5e2104e59e3a)

### Test Results

Humanoid-v2 (random seed=123456, alpha=0.05)
![Image text](https://github.com/Anonymous034/SAC-CVOR/blob/main/figures/2.png)

Humanoid-v2 (random seed=1, alpha=0.05)
![Image text](https://github.com/Anonymous034/SAC-CVOR/blob/main/figures/Humanoid_v2_1.png)

Humanoid-v2 (random seed=12, alpha=0.05)
![Image text](https://github.com/Anonymous034/SAC-CVOR/blob/main/figures/Humanoid_v2_12.png)

Humanoid-v2 (random seed=123, alpha=0.05)
![Image text](https://github.com/Anonymous034/SAC-CVOR/blob/main/figures/Humanoid_v2_123.png)

Humanoid-v2 (random seed=1234, alpha=0.05)
![Image text](https://github.com/Anonymous034/SAC-CVOR/blob/main/figures/Humanoid_v2_1234.jpg)

Humanoid-v2 (random seed=999, alpha=0.05)
![Image text](https://github.com/Anonymous034/SAC-CVOR/blob/main/figures/SAC+CVOR_999.png)

Humanoid-v2 (random seed=888, alpha=0.05)
![Image text](https://github.com/Anonymous034/SAC-CVOR/blob/main/figures/888.png)

Humanoid-v2 (random seed=777, alpha=0.05)
![Image text](https://github.com/Anonymous034/SAC-CVOR/blob/main/figures/777.png)

Humanoid-v2 (random seed=666, alpha=0.05)
![Image text](https://github.com/Anonymous034/SAC-CVOR/blob/main/figures/666.png)

Walker2d-v2 (random seed=1234, alpha=0.2)
![Image text](https://github.com/Anonymous034/SAC-CVOR/blob/main/figures/Walker2d-v2.png)

Ant-v2 (random seed=1234, alpha=0.2)
![Image text](https://github.com/Anonymous034/SAC-CVOR/blob/main/figures/Ant-v2.png)

### Usage OF CVOR

#### RUN SAC BASELINE

```
python main.py --env-name Humanoid-v2 --alpha 0.05
```

#### RUN SAC WITH CVOR

```
python main.py --env-name Humanoid-v2 --alpha 0.05 --cvor True
```

### Implementation OF CVOR (IN sac.py line 83-110)
```python
        policy_loss_bak = ((self.alpha * log_pi) - min_qf_pi)
        policy_loss = policy_loss_bak.mean()
        if not self.cvor:                                      # baseline 
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()
        else:                                                  # with cvor
            policy_loss_bak = policy_loss_bak.squeeze()

            '''
            #########################################
            main part for CVor with NN control variate
            #########################################
            '''
            self.F = nn.Sequential(
                nn.Linear(256, 512),
                nn.Tanh(),
                nn.Linear(512, 256),
            ).to('cuda')

            F_value = self.F(policy_loss_bak)
            F_value = (F_value - F_value.mean()) / (F_value.std() + 1e-5)
            tilde_F_value = torch.exp(F_value - F_value.detach()).mean()
            CVor = torch.exp((torch.exp(tilde_F_value - tilde_F_value.detach()) - torch.exp(F_value - F_value.detach())))
            CVor_loss = CVor * policy_loss_bak
            '''
            #########################################
            '''

            self.policy_optim.zero_grad()
            CVor_loss.mean().backward()
            self.policy_optim.step()
```





------------

### Description
------------
Reimplementation of [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf) and a deterministic variant of SAC from [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf).

Added another branch for [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf) -> [SAC_V](https://github.com/pranz24/pytorch-soft-actor-critic/tree/SAC_V).

### Requirements
------------
*   [mujoco-py](https://github.com/openai/mujoco-py)
*   [PyTorch](http://pytorch.org/)

### Default Arguments and Usage
------------
### Usage

```
usage: main.py [-h] [--env-name ENV_NAME] [--policy POLICY] [--eval EVAL]
               [--gamma G] [--tau G] [--lr G] [--alpha G]
               [--automatic_entropy_tuning G] [--seed N] [--batch_size N]
               [--num_steps N] [--hidden_size N] [--updates_per_step N]
               [--start_steps N] [--target_update_interval N]
               [--replay_size N] [--cuda]
```

(Note: There is no need for setting Temperature(`--alpha`) if `--automatic_entropy_tuning` is True.)

#### For SAC

```
python main.py --env-name Humanoid-v2 --alpha 0.05
```

#### For SAC (Hard Update)

```
python main.py --env-name Humanoid-v2 --alpha 0.05 --tau 1 --target_update_interval 1000
```

#### For SAC (Deterministic, Hard Update)

```
python main.py --env-name Humanoid-v2 --policy Deterministic --tau 1 --target_update_interval 1000
```

### Arguments
------------
```
PyTorch Soft Actor-Critic Args

optional arguments:
  -h, --help            show this help message and exit
  --env-name ENV_NAME   Mujoco Gym environment (default: HalfCheetah-v2)
  --policy POLICY       Policy Type: Gaussian | Deterministic (default:
                        Gaussian)
  --eval EVAL           Evaluates a policy a policy every 10 episode (default:
                        True)
  --gamma G             discount factor for reward (default: 0.99)
  --tau G               target smoothing coefficient(τ) (default: 5e-3)
  --lr G                learning rate (default: 3e-4)
  --alpha G             Temperature parameter α determines the relative
                        importance of the entropy term against the reward
                        (default: 0.2)
  --automatic_entropy_tuning G
                        Automaically adjust α (default: False)
  --seed N              random seed (default: 123456)
  --batch_size N        batch size (default: 256)
  --num_steps N         maximum number of steps (default: 1e6)
  --hidden_size N       hidden size (default: 256)
  --updates_per_step N  model updates per simulator step (default: 1)
  --start_steps N       Steps sampling random actions (default: 1e4)
  --target_update_interval N
                        Value target update per no. of updates per step
                        (default: 1)
  --replay_size N       size of replay buffer (default: 1e6)
  --cuda                run on CUDA (default: False)
```

| Environment **(`--env-name`)**| Temperature **(`--alpha`)**|
| ---------------| -------------|
| HalfCheetah-v2| 0.2|
| Hopper-v2| 0.2|
| Walker2d-v2| 0.2|
| Ant-v2| 0.2|
| Humanoid-v2| 0.05|

