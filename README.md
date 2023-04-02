# Reinforcement-Learning-Materials-with-OpenAI-Gym
Store some templates for neural network modeling

## 價值梯度

## 策略梯度
策略梯度是一種直接以神經網路產出**動作**(action)的建模邏輯，就像是我們人類在操作電腦一樣。在這樣的情境下，我們會將所謂的策略分成**行為策略(behavior policy)**和**目標策略(target policy)**來看待：行為策略是被用於與環境互動以產生經驗、探索不同可能性的策略，而目標策略則是一個從這些經驗學習優化決策規則的策略，這兩個策略可以是同一個神經網路，也可以是不同的神經網路。

### REINFORCE
on-policy Monte-Carlo

### Proximal Policy Optimization, PPO
- **資源**:
    - **程式碼**: [PPO-tensorflow1.13.1.py](#code)
    - **參考文獻**: https://arxiv.org/abs/1707.06347
