# 強化學習Tensorflow/Pytorch模版

就如同標題所示，這裡主要就是搭配OpenAI Gym的釋例，提供以Python建模強化學習模型的模板，也對研究強化學習的思路做一些統整。

## 目錄

- [背景](#background)
- [相關的研究進展](#install)
- [程式碼與內容說明](#install)


## 背景

在強化學習中，價值梯度和策略梯度是兩個重要的概念。**價值梯度**是指通過神經網路(neuron networks, NNs)計算每個動作的價值，使我們能夠自由地選擇適合的動作；而**策略梯度**則是指直接由NNs選擇要執行的動作。一般來說，價值梯度適合用於離散的動作空間(如↑/↓/←/→)，並且決策的後果是可視的；而策略梯度則更適用於連續的動作空間(車輛速率、馬力輸出等)，並且能減少人力的參與，但也因為如此經常使得開發者難以得知其決策規則。

諸如**Q-learning**和**SARSA**的價值梯度方法已經在許多實務應用中取得成功，包括機器人、遊戲和推薦系統。這些方法也也已經演進到能夠處理複雜和連續動作空間的**深度Q網路(deep Q-networks, DQNs)** 和**深度確定性策略梯度(deep deterministic policy gradients, DDPGs)**，而經典的REINFORCE策略梯度算法近年也取得重大的進展，例如使用**Actor-Critic**方法或**信任域策略優化 (trust region policy optimization, TRPO)** 來處理更高維的狀態空間(如遊戲畫面、仿生機器人的控制)。

為了能幫助大家快速地選擇合適的模型，以及避免一些工程上的機制被誤用，下個章節將會說明它們之間的演進。

## 相關的研究進展
### 價值梯度
(尚未整理)

### 策略梯度
由於策略梯度的NNs在更新其網路權重 $\pi$ 上會面臨到一個關鍵的瓶頸，就是在策略梯度在用與價值梯度相同的*on-policy*方法迭代時， $\pi_{iter=t}$ 會使用經驗 $E(S_{t}, A_{t}, R_{t+1}$ $|\pi_{iter=t} )$ 中的 $R$ 進行反向傳播。而在 $\pi_{iter=t}$ 更新後，過去所累積的經驗樣本 $E$ 將與新的 $\pi_{iter=t+1}$ 再也無關，因為 $\pi_{iter=t+1}$ 在 $S_{t}$ 時不會採取與 $\pi_{iter=t}$ 相同的 $A_{t}$ ，也不會得到相同的 $R_{t}$ 。因此過去所累積的經驗樣本將被丟棄，使得 $\pi$ 在訓練更新上非常沒有效率。這裡的 $\pi$ 不像一般在訓練ML模型或深度模型一樣，可以重複利用舊的樣本更新，而是在反向傳播一次過後就需要重新累積新的經驗。

#### 1. REINFORCE[[1]](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
最早由 Ronald J. William於1992年提出的Monte-Carlo REINFORCE演算法就是最典型的on-policy算法，由於Monte-Carlo需要在NNs在完成一整個系列的互動後才能計算價值回饋，儘管能有效讓 $\pi$ 進一步考量到長期價值，但也因為如此訓練過程變得極其緩慢，而且也容易陷入局部最佳解(只要找到一個可行的方法就不再進步了，即缺乏第一性原理)。

#### 2. Trust Region Policy Optimization, TRPO[[2]](https://arxiv.org/abs/1502.05477)

為了解決上述的問題，近年的策略梯度模型多採用*off-policy*的方法。其將策略梯度分為**行為策略(behavior policy, BP)** 和 **目標策略(target policy, TP)** 兩個 $NN$ ， $NN_{BP}$ 專門用於與環境互動累積經驗而不做更新，而 $NN_{TP}$ 則只負責利用這些經驗更新策略。藉由將兩者區分開來，使我們得以利用舊策略的經驗 $E(R|\pi_{iter=t})$ 來更新 $\pi_{iter=t+n}$ 。在這裡用了一個很重要的機制：**「隨機策略梯度」**，也是後續所有策略梯度的基礎。隨機策略梯度即是將 $NN$ 的動作 $A$ 區分為期望值 $\mu_{A}$ 與標準差 $\sigma_{A}$ ，從而組成行為的信任區域(Trust Region, $\theta$ )，其為一個常態機率分配。在 $NN_{BP}$ 與環境互動的過程中， $NN_{TP}$ 也會同步對每個狀態估計 $\theta_{TP}$ ，並與實際執行操作的 $\theta_{BP}$ 進行比較，以此衡量 $E_{BP}(R)$ 的重要度來更新 $NN_{TP}$ ，可以參考 $NN_{TP}$ 的surrogate梯度上升函數。當 $NN_{TP}$ 獲得的重要度都很低使其更新緩慢時，則可以考慮將 $NN_{TP}$ 的權重轉移給 $NN_{BP}$ 做一次經驗彙總(當然過去累積那些經驗樣本 $E$ 就需要丟棄了，但至少比on-policy每次更新都丟棄來得好)。

- surrogate函數: 
![](https://i.imgur.com/eXSKZLh.png)

這個手法稱作**重要度採樣**。直觀來講，就像我們在生活上不會經常改變自己的生活習慣(BP)，但是會將一些新的知識記在頭腦裡(TP)。當某天我們好好靜下來思考時，才會將一些頭腦裡的經驗轉化為實際的生活習慣。

> [備註：Actor-Critic框架]
> 在這裡也經常會導入另一種訓練機制，那就是Actor-Critic框架，以此再進一步提升樣本的使用效率。一般我們在衡量策略優勢 $A_{\pi_{\theta_{k}}}$ 時，需要以Monte-Carlo將未來的獎勵$R_{t+1,t+2...,t+n}$回歸到當前的狀態價值，因此 $NN_{TP}$ 也*無法對較新尚未結束Episode的樣本進行更新*。然而，我們可以藉由一個專門預測這個狀態價值的Critic網路來解決這個問題，使我們不必等到事實發生後才學到經驗，也能夠幫助我們在實際應用中衡量未來的潛力(如預知圍棋棋局當下的勝率)，為策略梯度帶來價值梯度的好處。

#### 3. Proximal Policy Optimization, PPO[[2]](https://arxiv.org/abs/1707.063477)
PPO是TRPO的改進版本，兩者皆可以透過surrogate函數防止策略出現太大的更動，使學習的過程有更好的穩定性與品質。而PPO的原理則是在原有surrogate函數的基礎上加入截斷機制來防止策略overfitting，如。這通常在訓練上會比TRPO更快速、樣本效率更高，但在某些情況可能會阻止策略找到最佳解。

- clipped surrogate函數: 
![](https://i.imgur.com/34hiku1.png)

## 程式碼與內容說明

|        模型     | `DQN`          |`DDPG`            |`TRPO`                |`PPO`           |
| :---:           | :---:            | :---:            | :---:            | :---:            |
| 原則         | Value-based    | Value-based      | Policy-based         | Policy-based   |
| 策略         | Deterministic    | Deterministic      | Stochastic         | Stochastic   |
| 訓練框架         | Q-table     | Actor-Critic    | Actor-Critic         | Actor-Critic   |
| Demo釋例         | -     | -     | -        | LunarLanderContinuous-v2   |
| 檔名/版本         | -     | -     |-         | [PPO-tensorflow1.13.1.py]()   |
| 平台         | -     | -     |-         | Linux   |
