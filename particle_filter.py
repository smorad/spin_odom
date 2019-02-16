#!/usr/bin/env python3

import pfilter


class SpinState:
    def __init__(self):
        self.s = {}
        self.s['ang_vel'] = 0
        self.s['ang_pos'] = 0

    def __delitem__(self, key):
        self.__delattr__(key)

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def to_vec(self):
        return [v for k,v in s.items()]

    def predict_state(self, state, dt=0.01):
        '''Predict state at t+1 given state at t'''
        return [state[0], state[1] + state[0] * dt]

    def predict_obs(self, state, dt=0.01):
        return [state[1] + state[0] * dt]


def init_model():
    return pfilter.ParticleFilter(
        SpinState().to_vec(),

    )

