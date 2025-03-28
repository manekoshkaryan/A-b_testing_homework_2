# A/B Testing Homework 2

## Overview
This project implements a multi-armed bandit experiment using two popular algorithms: **EpsilonGreedy** and **ThompsonSampling**. The experiment simulates 20,000 trials using four advertisement options with true reward means of `[1, 2, 3, 4]`. We compare the algorithms based on their cumulative reward and cumulative regret, and we also visualize how each arm’s estimated value evolves over time.

## Features
- **EpsilonGreedy Algorithm:**  
  Uses an epsilon that decays as 1/t and is initialized optimistically (each arm starts with an estimated value equal to the maximum true mean). This encourages early exploitation and quickly adjusts estimates.
  
- **ThompsonSampling Algorithm:**  
  Uses a Bayesian approach (assuming a known reward variance) with default initial estimates (starting at 0). This method samples from the posterior to decide which arm to pull.

- **Visualizations:**  
  - **Learning Curves (plot1):**  
    The learning process is visualized using separate subplots—one per arm—to clearly show the evolution of each arm's estimated value for each algorithm.
  - **Cumulative Rewards (plot2):**  
    A multi-panel plot compares the cumulative rewards of EpsilonGreedy and ThompsonSampling, and also displays the difference in their performance over time.

- **Data Logging:**  
  The trial-level data (including chosen arm, reward, and algorithm) is saved into CSV files, and a summary CSV provides the final cumulative reward and regret.

## Requirements
- **Python:** 3.10 or higher  
- **Dependencies:**  
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `loguru`

All dependencies are listed in the `requirements.txt` file.
   ```bash
   pip install -r requirements.txt
```

## Installation
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/manekoshkaryan/A-b_testing_homework_2.git
   cd A-b_testing_homework_2

## Usage

To run the experiment, execute the main script:
``` bash
python Bandit.py