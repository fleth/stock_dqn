# -*- coding: utf-8 -*-
import sys
import argparse
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
parser.add_argument("-l", "--load", dest="load", action="store_true",
                    default=False, help='load trained model (default: off)')
parser.add_argument("-e", "--epoch-num", dest="n_epochs", default=1000,
                    type=int, help='Numpber of training epochs (default: 1000)')
parser.add_argument("-m", "--memory-size", dest="memory_size", default=5000,
                    type=int, help='Numpber of memory size (default: 5000)')
args = parser.parse_args()

start_date = utils.format(args.start_date)
end_date = utils.format(args.end_date)

env = Env(start_date, end_date)
agent = Agent(env.actions, len(env.columns), env.state_size, args.memory_size)

if args.load:
    print("[Agent] load model")
    agent.load_model()

terminal = False
n_epochs = args.n_epochs
loops = -1
e = 0
total_frame = 0
do_replay_count = 0
max_step = 0
start_replay = False

optimistic = 3
optimistic_num = len(env.actions) * optimistic

def optimistic_action(env, epoch, optimistic):
    i = int(epoch / optimistic)
    return env.actions[i]

def is_optimistic_epoch(epoch, optimistic_num):
    return epoch < optimistic_num

while e < n_epochs:
    frame = 0
    loss = 0.0
    Q_max = 0.0
    env.reset()
    state_t_1, reward_t, terminal = env.observe()
    # 終了までアクションして経験を積む
    # 経験が一定以上になったら経験から学習する
    loops += 1

    while not terminal:
        state_t = state_t_1

        exploration = 0.1 if args.load and not start_replay else agent.exploration

        action_t, is_random = agent.select_action([state_t], exploration)
        env.execute_action(action_t)
        state_t_1, reward_t, terminal = env.observe()

        # 楽観的初期値法
        # 最初のK回は各アクションに対して同じ報酬を得られたことにする
        if is_optimistic_epoch(loops, optimistic_num) and not args.load:
            action_t = optimistic_action(env, loops, optimistic)
            reward_t = 1 if action_t == 0 else 1

        start_replay = False
        start_replay = agent.store_experience([state_t], action_t, reward_t, [state_t_1], terminal)

        if start_replay:
            do_replay_count += 1
            agent.update_exploration(e)
            if do_replay_count > 2:
                agent.replay()
                do_replay_count = 0

        if total_frame % 500 == 0 and start_replay:
            agent.update_target_model()

        frame += 1
        total_frame += 1
        loss += agent.current_loss
        Q_max += numpy.max(agent.Q_values([state_t]))

        if start_replay:
            agent.replay()
            print("epochs: %s/%s, loss: %s, Q_max: %s, terminal: %s, step: %s, action: %s, reward: %s, random: %s" % (e, n_epochs, loss / frame, Q_max / frame, terminal, env.step, action_t, reward_t, is_random))
            e += 1
            if max_step < env.step:
                max_step = env.step
            if e % 500 == 0:
                agent.save_model()
        else:
            print("frame: %s, total_frame: %s, terminal: %s, action: %s, reward: %s, is_random: %s" % (frame, total_frame, terminal, action_t, reward_t, is_random))


backend.clear_session()
print("max_step: %s, score: %s" % (max_step, env.score))
