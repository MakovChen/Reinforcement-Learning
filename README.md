# Reinforcement-Learning-Materials-with-OpenAI-Gym

## 價值梯度

## 策略梯度
策略梯度是一種直接以神經網路產出**動作(action, $A$)** 的建模邏輯，就像我們人類很直觀的操作電腦一樣。在這樣的情境下，產出動作的策略將會分成**行為策略(behavior policy)** 和 **目標策略(target policy)** 來看待：行為策略是被用於與環境互動以產生**經驗(experiment, $E$)**、*探索*不同可能性的策略，而目標策略則是個*利用*這些經驗學習來優化決策規則的策略，這兩個策略可以是同一個神經網路，也可以是不同的神經網路。我們之所以將這兩種策略分開來看，主要的原因是在於策略梯度有兩種主要的訓練的方式：
1. **On-policy**:  指在策略網路與環境*互動/探索*(行為策略的功能)的過程中，不斷*利用新經驗$E_{new}$更新梯度*(目標策略的功能)的做法，但是在策略網路反向傳播之後，這些些經驗就會變成舊的經驗$E_{old}$(與更新後的策略網路不相關)，因此這些舊有的經驗皆需直接丟棄。這種經驗用完即丟的方式訓練效率很差，而且也容易受限於局部最佳解。
2. **Off-plocy**: 將策略網路分為目標策略和行為策略兩個網路，使它們同時與環境互動以取得兩者所採取的動作$A_{old}$、$A_{new}$。行為策略的$A_{old}$用於與環境互動取得經驗$E_{old}$，而目標策略則是透過$A_{old}$與$A_{new}$間的差異來決定$E_{old}$的重要度，以此更新目標策略的網路(差異越小$E_{old}$越重要)，這種方式也就是所謂的重要度採樣(**Important Sampling**)。當$A_{old}$與$A_{new}$差異太大時，則可以考慮將目標策略的網路轉移到行為策略的網路，使整個策略能夠利用舊有且與當前策略無關的經驗來更新。

### REINFORCE
> REINFORCE就是在策略梯度中最典型on-policy的範例，若要使用Monte-Carlo search來估計長期的動作價值來更新梯度，那麼訓練的過程就會變得極為緩慢，而且容易陷入局部最佳解。

### Trust Region Policy Optimization, TRPO[[1]](https://arxiv.org/abs/1502.05477)


>TRPO則是一種Off-plocy的訓練框架，若再使用A2C將會同時存在三個網路(Actor目標策略網路、Actor行為策略網路與Critic網路)。而TRPO主要是將策略網路的動作劃分為期望值$A_{mu}$和標準差$A_{sigma}$，以此組合成動作的採樣空間$\theta$，也就是隨機梯度。如此一來，在設計梯度函數時便可以透過重要度採樣的比率與行為策略的優勢來更新目標策略以此解決了REINFORCE在on-policy上的缺點。可以參考下方的 surrogate gradient function。 
>
>![](https://i.imgur.com/eXSKZLh.png)
>* 補充：A2C算法也是為了補強REINFORCE因為樣本效率差而訓練緩慢的問題，藉由加入一個能預測後續獎勵回饋的神經網路(Critic)，便不需要再等到未來發生後才能開始更新策略，加快了訓練的時程。









### Proximal Policy Optimization, PPO[[2]](https://arxiv.org/abs/1707.063477)
> PPO主要是透過clip surrogate來改進TRPO在更新網路時的缺點
> ![](https://i.imgur.com/34hiku1.png)

**程式碼**: [PPO-tensorflow1.13.1.py](#code)
