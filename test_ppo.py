#!/usr/bin/env python

import unittest
from functools import partial

import gym
import torch 

from spinup import ppo_pytorch as ppo


class TestPPO(unittest.TestCase):
    def test_cartpole(self):
        ''' Test training a small agent in a simple environment '''
        env_fn = partial(gym.make, 'CartPole-v1')
        ac_kwargs = dict(hidden_sizes=(64,32))
        
        ppo(env_fn, ac_kwargs=ac_kwargs)
        # TODO: ensure policy has got better at the task


if __name__ == '__main__':
    unittest.main()
