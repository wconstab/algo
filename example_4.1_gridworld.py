import numpy as np

# example 4.1
#
# [ X  1  2  3  ]
# [ 4  5  6  7  ]
# [ 8  9  10 11 ]
# [ 12 13 14 X  ]
#
# Rt = -1 on all transitions; 0 if no state change
#
#A = ['up', 'down', 'right', 'left']
A = {'up': (-1, 0),
     'down': (1, 0),
     'right': (0, 1),
     'left': (0, -1)}

# Start with a random policy and randomly initialize the value function
P = {s : A.keys() for s in range(16)}

TERMINAL = 0
S = [[TERMINAL, 1, 2, 3],
     [4, 5, 6, 7],
     [8, 9, 10, 11],
     [12, 13, 14, TERMINAL]]
S = np.array(S, dtype=np.int32)


V = np.random.random((len(S), len(S)))


def policy_eval(policy, values, LAMBDA):
    # compute the new value for a state given a policy
    new_values = np.array(values)
    delta = 0
    for row in range(S.shape[0]):
        for col in range(S.shape[1]):
            state_idx = (row, col)
            state = S[state_idx]

            if state == TERMINAL:
                delta = max(delta, abs(0 - new_values[state_idx]))
                new_values[state_idx] = 0
                continue

            actions = P[state]
            new_v_accum = 0
            for a in actions:
                new_state_idx, reward = move(state_idx, a)
                new_state_value = values[new_state_idx]
                new_v_accum += reward + LAMBDA * new_state_value

            new_v_accum /= len(actions)
            delta = max(delta, abs(new_v_accum - new_values[state_idx]))
            new_values[state_idx] = new_v_accum

    return new_values, delta


def policy_improve(policy, values):
    new_policy = dict(policy)
    for row in range(S.shape[0]):
        for col in range(S.shape[1]):
            state_idx = (row, col)
            state = S[state_idx]
            best_actions = list()
            for a in A:
                new_state_idx, reward = move(state_idx, a)
                new_state_value = values[new_state_idx]

                best_value = -99999999
                if reward == -1:
                    # hack, only way to determine valid move?
                    if new_state_value > best_value:
                        best_actions = list()

                    best_value = max(best_value, new_state_value)
                    if best_value == new_state_value:
                        best_actions.append(a)

            new_policy[state] = best_actions

    return new_policy


def validate(row, col):
    rc = row >= 0 and \
         col >= 0 and \
         row < S.shape[0] and \
         col < S.shape[1]
    return rc


def is_terminal(row, col):
    return S[row, col] == TERMINAL


def move(state, action):
    row, col = state
    assert validate(row, col)
    
    if is_terminal(*state):
        return state, 0
  
    rinc, cinc = A[action]
    new_state = (row + rinc, col + cinc)
    reward = -1
    if validate(*new_state):
        if is_terminal(*new_state):
            reward = 0
    else:
        new_state = state
        reward = 0

    return new_state, reward


def move_test():
    state = (0, 1)
    for action in A:
        for i in range(3):
            new_state, reward =  move(state, A[action])
            print state, action, new_state, reward
            state = new_state


def pretty_print_policy(policy):
    for row in range(V.shape[0]):
        line = ""
        for col in range(V.shape[1]):
            policy_actions = policy[S[(row, col)]]
            encode = {'up': '^',
                      'down': 'v',
                      'left': '<',
                      'right': '>'}
            arrows = [encode[a] for a in policy_actions]
            line += "%4s\t" % "".join(arrows)
        print line

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--convergence_epsilon', '-e', default=1e-5, type=float)
parser.add_argument('--lambda_value', default=0.9, type=float)
args = parser.parse_args()


if __name__ == "__main__":
    policy = P
    new_V = V
    delta = 100
    iter = 0
    while delta > args.convergence_epsilon:
        new_V, delta = policy_eval(policy, new_V, LAMBDA=args.lambda_value)
        policy = policy_improve(policy, new_V)
        iter += 1
    print V.astype(np.int32)
    print new_V.astype(np.int32)
    pretty_print_policy(policy)
    print "Converged in ", iter, " iterations."
