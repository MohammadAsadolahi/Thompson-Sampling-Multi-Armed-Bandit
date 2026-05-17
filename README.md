<div align="center">

# 🎰 Thompson Sampling for Multi-Armed Bandits

### A Bayesian Approach to the Exploration–Exploitation Dilemma

[![Python](https://img.shields.io/badge/Python-3.7%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=plotly&logoColor=white)](https://matplotlib.org)
[![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org)

*A from-scratch NumPy implementation of Thompson Sampling for K-action stochastic bandits with Gaussian reward distributions, including experiments reproducing the TS-Normal strategy from [Agrawal & Goyal (2012)](https://arxiv.org/abs/1209.3353).*

**By [Mohammad Asadolahi](https://github.com/MohammadAsadolahi)**
Senior Agentic AI Engineer | Agentic AI Architectures In The Wild

</div>

---

## 📖 Overview

The **Multi-Armed Bandit (MAB)** problem is one of the most elegant formulations of the *exploration vs. exploitation* trade-off in sequential decision-making. Imagine standing in front of $K$ slot machines, each with an unknown reward distribution — how do you maximize your cumulative reward over $T$ rounds?

**Thompson Sampling** (TS) offers a beautifully simple Bayesian solution: maintain a posterior belief over each arm's reward distribution, sample from those posteriors, and play the arm with the highest sample. As evidence accumulates, the posteriors concentrate around the true means, naturally shifting the balance from exploration to exploitation.

This project implements Thompson Sampling with **Gaussian (Normal) priors and likelihoods**, providing an intuitive visualization of how posterior beliefs evolve over time.

## 🧮 The Math

For each arm $k$, we model the reward as $r \sim \mathcal{N}(\mu_k, \sigma_k^2)$ and maintain a conjugate Normal posterior over $\mu_k$:

$$\mu_k \mid \text{data} \sim \mathcal{N}(\mu_0, \tau_0^{-1})$$

where $\tau_0$ is the **precision** ($\tau = 1 / \text{variance}$) of the posterior. After observing $n$ rewards, the posterior is updated as:

$$\mu_0 \leftarrow \frac{\tau_0 \cdot \mu_0 + n \cdot \bar{x}}{\tau_0 + n}, \qquad \tau_0 \leftarrow \tau_0 + 1$$

where $\bar{x}$ is the running mean of observed rewards (computed incrementally).

At each time step:
1. **Sample** $\theta_k \sim \mathcal{N}(\mu_0^{(k)}, (\tau_0^{(k)})^{-1})$ for each arm
2. **Select** arm $a_t = \arg\max_k \theta_k$
3. **Observe** reward $r_t$ from the selected arm
4. **Update** the posterior of arm $a_t$

## 🏗️ Project Structure

```
Thompson_Sampling_Multi_Armed_Bandit/
├── Thompson_Sampling_Multi_Armed_Bandit.py   # Full standalone script
├── THombson_sampling.ipynb                   # Jupyter notebook with visualizations
└── README.md
```

| File | Description |
|------|-------------|
| `Thompson_Sampling_Multi_Armed_Bandit.py` | Self-contained Python script with the `Gaussian_TS_Arm` class and all experiments |
| `THombson_sampling.ipynb` | Interactive notebook with step-by-step explanations, formulas, and plotted results |

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy matplotlib scipy seaborn tqdm pandas
```

### Run

```bash
python Thompson_Sampling_Multi_Armed_Bandit.py
```

Or open `THombson_sampling.ipynb` in Jupyter for an interactive walkthrough.

## 🧪 Experiments

### Experiment 1 — 4-Arm Bandit (T = 100)

Four arms with true distributions $\mathcal{N}(4, 1)$, $\mathcal{N}(2, 2)$, $\mathcal{N}(5, 3)$, and $\mathcal{N}(-5, 1)$.

After 100 trials, Thompson Sampling correctly identifies the best arm (μ = 5) and concentrates most pulls on it. The posterior distributions sharpen around the true means, with the best arm receiving the tightest estimate.

![4 arms bandit for horizon 100 using Thompson sampling](https://github.com/MohammadAsadolahi/Thompson_Sampling_Multi_Armed_Bandit/blob/main/Normal%20distribution%203%20arm%20bandit.png)

> *Legend shows pulls per arm / total trials. The algorithm quickly learns to favor the highest-reward arm.*

---

### Experiment 2 — TS-Normal Strategy (Agrawal & Goyal)

Reproduces the **TS-Normal** strategy from *"Further Optimal Regret Bounds for Thompson Sampling"* by Agrawal & Goyal ([arXiv:1209.3353](https://arxiv.org/abs/1209.3353)).

**Setup:** 100 instances of a 2-arm bandit, $\sigma^2 = 0.25$, $\mu_k \sim \text{Uniform}(0, 10)$, horizon $T = 50$, with varying prior mean $m_0 \in \{0, 5, 10\}$.

#### $m_0 = 0$
![TS-Normal strategy — m₀ = 0](https://github.com/MohammadAsadolahi/Thompson_Sampling_Multi_Armed_Bandit/blob/main/TS-Normal%20strategy%20for%202%20arm%20bandit%20m%20%3D0%20.png)

#### $m_0 = 5$
![TS-Normal strategy — m₀ = 5](https://github.com/MohammadAsadolahi/Thompson_Sampling_Multi_Armed_Bandit/blob/main/TS-Normal%20strategy%20for%202%20arm%20bandit%20m%20%3D5%20.png)

#### $m_0 = 10$
![TS-Normal strategy — m₀ = 10](https://github.com/MohammadAsadolahi/Thompson_Sampling_Multi_Armed_Bandit/blob/main/TS-Normal%20strategy%20for%202%20arm%20bandit%20m%20%3D10%20.png)

> *The prior mean $m_0$ has minimal impact on asymptotic performance — Thompson Sampling is robust to prior misspecification.*

---

### Experiment 3 — Long Horizon (T = 1000)

100 instances of a 2-arm bandit with $\sigma^2 = 0.25$ and $\mu_k \sim \text{Uniform}(0, 1)$. With a longer horizon, the posteriors become extremely concentrated, demonstrating the algorithm's convergence guarantees.

![2-arm bandit, T=1000](https://github.com/MohammadAsadolahi/Thompson_Sampling_Multi_Armed_Bandit/blob/main/Normal%20distribution%203%20arm%20bandit%20%CF%832%20%3D%200.25%20and%20%CE%BCk%20uniformly%20sampled%20in%20the%20interval%20%5B0.0%2C%201.0%5D.png)

> *With 1000 trials, the posterior variance shrinks dramatically, confirming near-optimal arm identification.*

## 📚 Key Concepts

| Concept | Description |
|---------|-------------|
| **Thompson Sampling** | A Bayesian algorithm that balances exploration and exploitation by sampling from posterior distributions |
| **Conjugate Prior** | Using a Normal prior with a Normal likelihood yields a closed-form Normal posterior |
| **Precision (τ)** | The inverse of variance; increases with each observation, tightening the posterior |
| **Regret** | The cumulative difference between the optimal arm's reward and the agent's actual reward |
| **TS-Normal** | The specific Thompson Sampling variant for Normal rewards, proven to achieve $O(\sqrt{KT \ln T})$ regret |

## 📄 Reference

> Shipra Agrawal and Navin Goyal. *"Further Optimal Regret Bounds for Thompson Sampling."*
> In Proceedings of the 16th International Conference on Artificial Intelligence and Statistics (AISTATS), 2013.
> [[arXiv:1209.3353]](https://arxiv.org/abs/1209.3353)

---

## Author

**Mohammad Asadolahi** — Senior Agentic AI Engineer

Focused on **Agentic AI Architectures In The Wild**.

[![GitHub](https://img.shields.io/badge/GitHub-MohammadAsadolahi-181717?logo=github)](https://github.com/MohammadAsadolahi)
