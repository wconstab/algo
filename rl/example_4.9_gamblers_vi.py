import numpy as np

def initial_value():
    v = np.random.random(101)
    v[0] = 0
    v[100] = 0
    return v


def max_stakes(capital):
    return min(capital, 100-capital)


def states():
    return range(0, 101)


def possible_actions(state):
    return range(1, max_stakes(state) + 1)


def value_iter(values, states, P_WIN):
    delta = 0
    for s in states[1:-1]:
        v = values[s]
        best_action_value = 0
        for a in possible_actions(s):
            r = 1 if s + a == 100 else 0
            action_value = P_WIN * (r + values[s + a]) + \
                           (1 - P_WIN) * values[s - a]
            if action_value > best_action_value:
                best_action_value = action_value

        values[s] = best_action_value
        delta = max(delta, abs(v - values[s]))
    return values, delta

dbg_states = [24, 25, 26]
def best_policy(values, states, P_WIN):
    p = np.zeros(len(states))
    for s in states[1:-1]:
        best_action_value = 0
        best_action = 0
        if s in dbg_states:
            print "State", s
        for a in possible_actions(s):
            r = 1 if s + a == 100 else 0
            action_value = P_WIN * (r + values[s + a]) + \
                           (1 - P_WIN) * values[s - a]
            if s in dbg_states:
                print "a=", a, " s+a, v[s+a]=", s+a, values[s+a], " s-a, v[s-a]=", s-a, values[s-a], " summed=", action_value
            if action_value > best_action_value:
                best_action_value = action_value
                best_action = a
        p[s] = best_action
    return p
             

def train(P_WIN=0.4):
    S_plus = states()
    V = initial_value() 

    theta = 1e-13
    delta = 100
    i = 0
    while delta > theta:
        V, delta = value_iter(V, S_plus, P_WIN)
        i += 1
    
    P = best_policy(V, S_plus, P_WIN) 

    print i, " iterations"
    print V
    print P

if __name__ == "__main__":
    
    train()
