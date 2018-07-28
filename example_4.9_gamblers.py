import numpy as np

def initial_value():
    v = np.random.random(101)
    v[0] = 0
    v[100] = 1
    return v


def initial_policy_random(states):
    p = [0] + [np.random.randint(1, max_stakes(s) + 1) for s in states[1:-1]] + [0]
    return np.array(p)


def initial_policy_zeros(states):
    p = [0 for s in states]
    return np.array(p)


def max_stakes(capital):
    return min(capital, 100-capital)


def states():
    return range(0, 101)


def possible_actions(state):
    return range(1, max_stakes(state))

def evaluate_policy(policy, values, states, P_WIN):
    new_values = values.copy()
    for s in states[1:-1]:
        a = policy[s]
        new_values[s] = P_WIN * (values[s + a]) + \
                        (1 - P_WIN) * values[s - a]
    return new_values


def improve_policy(policy, values, states, P_WIN):
    new_policy = policy.copy()
    for s in states[1:-1]:
        best_action_value = 0
        for a in possible_actions(s):
            action_value = P_WIN * values[s + a] + \
                           (1 - P_WIN) * values[s - a]
            if action_value > best_action_value:
                best_action_value = action_value
                new_policy[s] = a
    return new_policy


def train(P_WIN=0.4):
    S_plus = states()
    V = initial_value() 
    P = initial_policy_zeros(S_plus)
    #P = initial_policy_random(S_plus)

    for i in range(1000):
        V = evaluate_policy(P, V, S_plus, P_WIN)
        P = improve_policy(P, V, S_plus, P_WIN) 

    print V
    print P

if __name__ == "__main__":
    
    train()
