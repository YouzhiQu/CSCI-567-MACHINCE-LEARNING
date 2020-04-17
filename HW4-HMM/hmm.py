from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        o1 = self.obs_dict[Osequence[0]]
        alpha[:, 0] = self.pi * self.B[:, o1]

        for t in range(1, L):
            o_t = self.obs_dict[Osequence[t]]
            new_state = np.dot(alpha[:, t-1], self.A)
            alpha[:, t] = new_state * self.B[:, o_t]


        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        beta[:, L-1] = np.ones(S)
        for t in range(L-2, -1, -1):
            o_t = self.obs_dict[Osequence[t+1]]
            p = self.B[:, o_t] * beta[:, t+1]
            beta[:, t] = np.dot(self.A, p)
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        L = len(Osequence)
        A = self.forward(Osequence)
        prob = np.sum(A[:, L-1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        A = self.forward(Osequence)
        B = self.backward(Osequence)
        P = self.sequence_prob(Osequence)
        prob = A * B / P
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here

        A = self.forward(Osequence)
        B = self.backward(Osequence)
        P = self.sequence_prob(Osequence)

        for i in range(S):
            for t in range(L-1):
                o_t = self.obs_dict[Osequence[t + 1]]
                prob[i][:, t] = A[i][t] * self.A[i] * self.B[:, o_t] * B[:, t+1] / P

        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S, L])
        temp_path = np.zeros([S, L], dtype=int)
        o1 = self.obs_dict[Osequence[0]]
        delta[:, 0] = self.pi * self.B[:, o1]
        for i in range(1, L):
            o_t = self.obs_dict[Osequence[i]]
            b_ot = self.B[:, o_t]
            state = (delta[:, i - 1].reshape(S, 1) * self.A)
            probability = state.max(axis=0)
            temp_path[:, i] = np.argmax(state, axis=0)
            delta[:, i] = b_ot * probability
        p_last = np.argmax(delta[:, L-1])
        path.insert(0, p_last)
        for i in range(L-1):
            state = temp_path[:, L-1-i]
            p_last = state[p_last]
            path.insert(0, p_last)
        for i in range(L):
            path[i] = list(self.state_dict.keys())[list(self.state_dict.values()).index(path[i])]
        ###################################################
        return path
