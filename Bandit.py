from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger


# ------------------------- Provided Abstract Base Classes ------------------------- #
class Bandit(ABC):
    @abstractmethod
    def __init__(self, p):
        self.p = p

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        pass


class AdOption:
    def __init__(self, true_mean, option_id, init_value=0.0):
        """
        :param true_mean: The true reward mean for this ad option.
        :param option_id: Identifier for the ad option.
        :param init_value: Initial estimated value (default is 0.0).
        """
        self.true_mean = true_mean
        self.option_id = option_id
        self.n = 0  # number of pulls
        self.value = init_value  # estimated mean reward

    def pull(self):
        """Simulate pulling the ad option (reward ~ N(true_mean, 1))."""
        return np.random.normal(self.true_mean, 1)

    def update(self, reward):
        """Update the estimated reward using an incremental average."""
        self.n += 1
        self.value += (reward - self.value) / self.n

    def __repr__(self):
        return f"Ad Option {self.option_id + 1}: true={self.true_mean:.2f}, est={self.value:.2f}, pulls={self.n}"


# --------------------------------------#
class EpsilonGreedy(Bandit):
    def __init__(self, p, trials=20000):
        """
        Initialize EpsilonGreedy.
        :param p: List of true reward means for each ad.
        :param trials: Total number of trials.
        """
        super().__init__(p)
        self.trials = trials
        self.arms = [AdOption(mu, idx, init_value=max(p)) for idx, mu in enumerate(p)]
        self.selected_arms = []  # store chosen arm ids (0-indexed)
        self.rewards = []  # store reward per trial
        self.cum_rewards = []  # cumulative rewards over trials
        self.cum_regret = []  # cumulative regret over trials
        self.estimated_history = {arm.option_id: [] for arm in self.arms}
        self.optimal_mean = max(p)

    def __repr__(self):
        return "EpsilonGreedy"

    # Dummy implementations to satisfy the abstract methods:
    def pull(self):
        pass

    def update(self):
        pass

    def experiment(self):
        total_reward = 0.0
        total_regret = 0.0
        for t in range(self.trials):
            epsilon = 1.0 / (t + 1)
            if np.random.rand() < epsilon:
                chosen_arm = np.random.choice(self.arms)
            else:
                chosen_arm = max(self.arms, key=lambda arm: arm.value)
            self.selected_arms.append(chosen_arm.option_id)
            r = chosen_arm.pull()
            chosen_arm.update(r)
            total_reward += r
            total_regret += (self.optimal_mean - chosen_arm.true_mean)
            self.rewards.append(r)
            self.cum_rewards.append(total_reward)
            self.cum_regret.append(total_regret)
            for arm in self.arms:
                self.estimated_history[arm.option_id].append(arm.value)
            if (t + 1) % 5000 == 0:
                logger.info(f"EpsilonGreedy trial {t + 1}: cumulative reward = {total_reward:.2f}")
        return total_reward, total_regret

    def report(self):
        df = pd.DataFrame({
            "Bandit": self.selected_arms,
            "Reward": self.rewards,
            "Algorithm": ["EpsilonGreedy"] * len(self.rewards)
        })
        df.to_csv("epsilon_rewards.csv", index=False)
        logger.info(f"EpsilonGreedy -> Cumulative Reward: {self.cum_rewards[-1]:.2f}")
        logger.info(f"EpsilonGreedy -> Cumulative Regret: {self.cum_regret[-1]:.2f}")


# --------------------------------------#
class ThompsonSampling(Bandit):
    def __init__(self, p, trials=20000, known_var=1.0, prior_var=1.0):
        """
        Initialize ThompsonSampling.
        :param p: List of true reward means.
        :param trials: Total number of trials.
        :param known_var: Assumed known variance of rewards.
        :param prior_var: Variance of the prior for the mean.
        """
        super().__init__(p)
        self.trials = trials
        self.known_var = known_var
        self.prior_var = prior_var
        # Standard initialization: initial estimates remain at 0.0.
        self.arms = [AdOption(mu, idx) for idx, mu in enumerate(p)]
        self.selected_arms = []
        self.rewards = []
        self.cum_rewards = []
        self.cum_regret = []
        self.estimated_history = {arm.option_id: [] for arm in self.arms}
        self.optimal_mean = max(p)
        self.counts = {arm.option_id: 0 for arm in self.arms}
        self.sum_rewards = {arm.option_id: 0.0 for arm in self.arms}

    def __repr__(self):
        return "ThompsonSampling"

    def pull(self):
        pass

    def update(self):
        pass

    def experiment(self):
        total_reward = 0.0
        total_regret = 0.0
        for t in range(self.trials):
            samples = []
            for arm in self.arms:
                if self.counts[arm.option_id] == 0:
                    sample = np.random.normal(arm.true_mean, np.sqrt(self.prior_var))
                else:
                    mean_est = self.sum_rewards[arm.option_id] / self.counts[arm.option_id]
                    sample = np.random.normal(mean_est, np.sqrt(self.known_var / self.counts[arm.option_id]))
                samples.append(sample)
            chosen_index = int(np.argmax(samples))
            chosen_arm = self.arms[chosen_index]
            self.selected_arms.append(chosen_arm.option_id)
            r = chosen_arm.pull()
            self.counts[chosen_arm.option_id] += 1
            self.sum_rewards[chosen_arm.option_id] += r
            chosen_arm.update(r)
            total_reward += r
            total_regret += (self.optimal_mean - chosen_arm.true_mean)
            self.rewards.append(r)
            self.cum_rewards.append(total_reward)
            self.cum_regret.append(total_regret)
            for arm in self.arms:
                self.estimated_history[arm.option_id].append(arm.value)
            if (t + 1) % 5000 == 0:
                logger.info(f"ThompsonSampling trial {t + 1}: cumulative reward = {total_reward:.2f}")
        return total_reward, total_regret

    def report(self):
        df = pd.DataFrame({
            "Bandit": self.selected_arms,
            "Reward": self.rewards,
            "Algorithm": ["ThompsonSampling"] * len(self.rewards)
        })
        df.to_csv("thompson_rewards.csv", index=False)
        logger.info(f"ThompsonSampling -> Cumulative Reward: {self.cum_rewards[-1]:.2f}")
        logger.info(f"ThompsonSampling -> Cumulative Regret: {self.cum_regret[-1]:.2f}")


# --------------------------------------#
class Visualization():
    def plot1(self, estimated_history, algorithm_name):
        """
        Visualize the learning process (evolution of estimated values) for one algorithm.
        Produces a single linear-scale plot.

        :param estimated_history: Dict mapping arm id to list of estimated values.
        :param algorithm_name: Name of the algorithm.
        """
        plt.figure(figsize=(10, 6))
        for arm_id, estimates in estimated_history.items():
            plt.plot(estimates, label=f"Arm {arm_id + 1}")
        plt.xlabel("Trial")
        plt.ylabel("Estimated Value")
        plt.title(f"Learning Curve: {algorithm_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{algorithm_name.replace(' ', '_').lower()}_learning_process.png")
        plt.close()

    def plot2(self, cum_rewards_eps, cum_rewards_ts):
        """
        Visualize cumulative rewards from both algorithms in a combined plot.

        :param cum_rewards_eps: List of cumulative rewards from EpsilonGreedy.
        :param cum_rewards_ts: List of cumulative rewards from ThompsonSampling.
        """
        trials = range(1, len(cum_rewards_eps) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(trials, cum_rewards_eps, label="EpsilonGreedy", color='red')
        plt.plot(trials, cum_rewards_ts, label="ThompsonSampling", color='blue')
        plt.xlabel("Trial")
        plt.ylabel("Cumulative Reward")
        plt.title("Cumulative Rewards Comparison")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("cumulative_rewards_comparison.png")
        plt.close()


# --------------------------------------#
if __name__ == '__main__':
    np.random.seed(123)
    true_means = [1, 2, 3, 4]
    trials = 20000

    # EpsilonGreedy Experiment
    eps_algo = EpsilonGreedy(true_means, trials)
    logger.info("Starting EpsilonGreedy Experiment...")
    eps_total, eps_regret = eps_algo.experiment()
    eps_algo.report()

    # ThompsonSampling Experiment
    ts_algo = ThompsonSampling(true_means, trials, known_var=1.0, prior_var=1.0)
    logger.info("Starting ThompsonSampling Experiment...")
    ts_total, ts_regret = ts_algo.experiment()
    ts_algo.report()

    # Overall summary CSV
    summary = pd.DataFrame({
        "Algorithm": ["EpsilonGreedy", "ThompsonSampling"],
        "Cumulative Reward": [eps_algo.cum_rewards[-1], ts_algo.cum_rewards[-1]],
        "Cumulative Regret": [eps_algo.cum_regret[-1], ts_algo.cum_regret[-1]]
    })
    summary.to_csv("cumulative_summary.csv", index=False)

    print(f"EpsilonGreedy -> Cumulative Reward: {eps_algo.cum_rewards[-1]:.2f}, Regret: {eps_algo.cum_regret[-1]:.2f}")
    print(f"ThompsonSampling -> Cumulative Reward: {ts_algo.cum_rewards[-1]:.2f}, Regret: {ts_algo.cum_regret[-1]:.2f}")

    # Visualization
    vis = Visualization()
    vis.plot1(eps_algo.estimated_history, "EpsilonGreedy")
    vis.plot1(ts_algo.estimated_history, "ThompsonSampling")
    vis.plot2(eps_algo.cum_rewards, ts_algo.cum_rewards)

    logger.info("Experiment complete. CSV files and visualizations have been saved.")