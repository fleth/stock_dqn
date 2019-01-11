# -*- coding: utf-8 -*-
import argparse
import sys
import numpy
import json
from dqn_env import Env
from dqn_agent import Agent
from keras.backend import tensorflow_backend as backend

sys.path.append("lib")
import utils

parser = argparse.ArgumentParser()
parser.add_argument("start_date", action="store")
parser.add_argument("end_date", action="store")
parser.add_argument("-m", "--memory-size", dest="memory_size", default=5000,
                    type=int, help='Numpber of memory size (default: 5000)')
args = parser.parse_args()

start_date = utils.format(args.start_date)
end_date = utils.format(args.end_date)

env = Env(start_date, end_date)
agent = Agent(env.actions, len(env.columns), env.state_size, args.memory_size)
agent.load_model()

terminal = False
total_frame = 0
max_step = 0
frame = 0
state_t, reward_t, terminal = env.observe()
while not terminal:

    action_t, is_random = agent.select_action([state_t], 0.0)
    env.execute_action(action_t)

    state_t, reward_t, terminal = env.observe()

    frame += 1
    total_frame += 1
    if max_step < env.step:
        max_step = env.step

    print("frame: %s, total_frame: %s, terminal: %s, action: %s, reward: %s" % (frame, total_frame, terminal, action_t, reward_t))

backend.clear_session()
print("max_step: %s, score: %s" % (max_step, env.score))
