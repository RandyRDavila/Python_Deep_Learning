import pandas_datareader as web
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import empyrical as emp
from mlp import MultilayerPerceptron


__all__ = ["ulcer",
            "state_converstion",
            "make_sample_playground",
            "make_dpn_network",
            "epsilon_greedy_action",
            "state_action_reward",
            "collect_replay"]

def ulcer_index(prices):
    L = len(prices)
    R = [0.0]
    high = np.max(prices)
    for i in range(1, L):
        r = 100* (prices[i] - high)/high
        R.append(r)

    return np.sqrt(sum(r**2 for r in R)/L)



def state_converstion(playground, state, lags=30):
    cols = [f"lag{i}" for i in range(1, lags+1)]
    state = playground[cols].iloc[state].to_numpy()
    
    x = np.array([ulcer_index(state),
                    #emp.annual_volatility(state),
                    #emp.stability_of_timeseries(state),
                    emp.tail_ratio(state),
                    emp.conditional_value_at_risk(state)])
    x = x/np.linalg.norm(x)
    n = len(x)
    x = x.reshape(n, 1)
    return x


def make_sample_playground(ticker="AMD", source="yahoo", start="2019-02-01", end="2021-10-20", lags=30):
    df = web.DataReader(ticker, source, start=start, end=end)
    data = pd.DataFrame(df.Close)
    cols = []
    for i in range(1, lags+1):
        col = f"lag{i}"
        data[col] = data.Close.shift(i)
        cols.append(col)

    data.dropna(inplace=True)

    return data


def make_dqn_network(layers):
    return MultilayerPerceptron(layers)


def epsilon_greedy_action(playground, q_network, state, action_space, epsilon = 0.002):
    greedy_or_not = [1, 0]
    choice = np.random.choice(greedy_or_not, p = [1 - epsilon, epsilon])
    if choice == 1:
        x = state_converstion(playground, state)
        q_values = q_network.forward_pass(x, predict_vector = True)
        return action_space[np.argmax(q_values)]

    else:
        i = np.random.randint(len(action_space))
        return action_space[i]

def state_action_reward(playground, q_network, state, action_space, epsilon = 0.002):
    action = epsilon_greedy_action(playground, q_network, state, action_space, epsilon = epsilon)
    current_day_close = playground["Close"].iloc[state]
    next_day_close = playground["Close"].iloc[state+1]
    reward = (next_day_close - current_day_close)*action
    return reward

def collect_replay(playground, q_network, action_space, K=10, steps = 10):
    replay = []
    N = len(playground) - steps + 1
    for _ in range(K):
        state = np.random.randint(0, N - steps)
        episode = []
        for j in range(state, state+steps):
            action = epsilon_greedy_action(playground, q_network, j, action_space, epsilon = 0.015)
            current_day_close = playground["Close"].iloc[j]
            next_day_close = playground["Close"].iloc[j+1]
            reward = (next_day_close - current_day_close)*action
            replay.append([j, action, reward])
       
    return replay

def collect_target_q(playground, q_network, action_space, K=10, steps = 16, gamma = 0.79):
    Y = []
    X = []
    batch = collect_replay(playground, q_network, action_space)
    for experience in batch:
        x = state_converstion(playground, experience[0])
        X.append(x)

        y = experience[2] + gamma*epsilon_greedy_action(playground, q_network, experience[0], action_space)
        Y.append(y)

    return X, Y


def total_reward(playground, q_network, action_space):
    r = 0
    for i in range(len(playground)-10):
        r += state_action_reward(playground, q_network, i, action_space)

    return r

playground = make_sample_playground()

net = make_dqn_network([3, 600, 600, 3])
action_space = [1, .5,-1]
net2 = make_dqn_network([3, 10, 10, 3])
net2.W = list(net.W)
net2.B = list(net.B)
print(f"Initial Reward = {total_reward(playground, net2, action_space)}")
for _ in range(600):
    X, Y = collect_target_q(playground, net, action_space, K=1_000_000)

    net2.stochastic_gradient_descent(X, Y)
    print(f"{_}-th Reward = {total_reward(playground, net2, action_space)}")

    net.W = list(net2.W)
    net.B = list(net2.B)



for i in range(len(playground)):
    print(epsilon_greedy_action(playground, net, i, action_space))
    