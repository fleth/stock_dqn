# -*- coding: utf-8 -*-
import os
import numpy
import copy
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import Lambda, Input, Reshape, Conv2D
from keras.layers.recurrent import LSTM
from keras.models import model_from_yaml, Model
from keras.optimizers import RMSprop
from keras import backend as K
from collections import deque
from keras.models import model_from_config

losses = {'loss': lambda y_true, y_pred: y_pred,
    'main_output': lambda y_true, y_pred: K.zeros_like(y_pred)}

def loss_func(args):
    import tensorflow as tf
    y_true, y_pred = args
    error = tf.abs(y_pred - y_true)
    quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
    linear_part = error - quadratic_part
    loss = tf.reduce_sum(0.5 * tf.square(quadratic_part) + linear_part)
    tf.summary.scalar('loss', loss)
    return loss

def clone_model(model, custom_objects={}):
    config = {
        'class_name': model.__class__.__name__,
        'config': model.get_config(),
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone


class Agent:

    def __init__(self, actions, features, state_size, memory_size=5000):
        self.actions = actions
        self.replay_memory_size = memory_size
        self.minibatch_size = 32
        self.D = deque(maxlen=self.replay_memory_size)
        self.exploration = 1.0
        self.exploration_step = (1.0 - 0.1) / 500
        self.current_loss = 0.0
        self.learning_rate = 0.0025
        self.discount_factor = 0.9
        self.n_actions = len(self.actions)
        self.features = features
        self.state_size = state_size
        self.create_model()
#        self.cnn()

    def Q_values(self, states, isTarget=False):
        model = self.target_model if isTarget else self.model
        res = model.predict({'state': numpy.array([states]),
                             'action': numpy.array([0]),
                             'y_true': numpy.array([[0] * self.n_actions])
                             })
        return res[1][0]

    # 経験蓄積
    def store_experience(self, states, action, reward, states_1, terminal):
        self.D.append((states, action, reward, states_1, terminal))
        return (len(self.D) >= self.replay_memory_size)

    def update_exploration(self, num):
        if self.exploration > 0.1:
            self.exploration -= self.exploration_step * num
            if self.exploration < 0.1:
                self.exploration = 0.1

    # 行動選択
    def select_action(self, states, epsilon):
        if numpy.random.rand() <= epsilon:
            # random
            return numpy.random.choice(self.actions), True
        else:
            # max_action Q(state, action)
            return self.actions[numpy.argmax(self.Q_values(states))], False

    # 経験から学習
    def replay(self):
        state_minibatch = []
        y_minibatch = []

        minibatch_size = min(len(self.D), self.minibatch_size)
        minibatch_indexes = numpy.random.randint(0, len(self.D), minibatch_size)

        for j in minibatch_indexes:
            state_j, action_j, reward_j, state_j_1, terminal = self.D[j]
            action_j_index = self.actions.index(action_j)
            y_j = self.Q_values(state_j)
            v = numpy.max(self.Q_values(state_j_1, isTarget=True))
            y_j[action_j_index] = reward_j + self.discount_factor * v

            state_minibatch.append(state_j)
            y_minibatch.append(y_j)

        self.model.fit({'state': numpy.array(state_minibatch),
                        'y_true': numpy.array(y_minibatch)},
                       [numpy.zeros([minibatch_size]),
                        numpy.array(y_minibatch)],
                       batch_size=minibatch_size,
                       nb_epoch=1,
                       verbose=0)

        score = self.model.predict({'state': numpy.array(state_minibatch),
                                    'y_true': numpy.array(y_minibatch)})
        self.current_loss = score[0][0]

    # 評価用モデルを更新
    def update_target_model(self):
        self.target_model = clone_model(self.model)

    def create_model(self):
        state_input = Input(shape=(1, self.features, self.state_size), name='state')

        x = Flatten()(state_input)
        x = Dense(512, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        y_pred = Dense(self.n_actions, activation='linear', name='main_output')(x)

        y_true = Input(shape=(self.n_actions, ), name='y_true')
        loss_out = Lambda(loss_func, output_shape=(1, ), name='loss')([y_true, y_pred])
        self.model = Model(inputs=[state_input, y_true], outputs=[loss_out, y_pred])

        optimizer = RMSprop
        self.model.compile(loss=losses,
                           optimizer=optimizer(lr=self.learning_rate),
                           metrics=['accuracy'])

        self.target_model = copy.copy(self.model)

    def cnn(self):

        state_input = Input(shape=(1, self.features, self.state_size), name='state')
        action_input = Input(shape=[None], name='action', dtype='int32')

        x = Conv2D(128, (4, 4), padding='same', activation='relu', strides=(2, 2))(state_input)
        x = Dropout(0.25)(x)
        x = Conv2D(64, (2, 2), padding='same', activation='relu', strides=(1, 1))(x)
        x = Dropout(0.1)(x)
        x = Conv2D(64, (2, 2), padding='same', activation='relu', strides=(1, 1))(x)
        x = Dropout(0.05)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)

        y_pred = Dense(self.n_actions, activation='linear', name='main_output')(x)
        y_true = Input(shape=(self.n_actions, ), name='y_true')

        loss_out = Lambda(loss_func, output_shape=(1,), name='loss')([y_true, y_pred])
        self.model = Model(inputs=[state_input, y_true], outputs=[loss_out, y_pred])

        optimizer = RMSprop
        self.model.compile(loss=losses,
                           optimizer=optimizer(lr=self.learning_rate),
                           metrics=['accuracy'])

        self.target_model = copy.copy(self.model)

    def settings_path(self):
        return os.path.dirname(__file__)

    def load_model(self):
        yaml_string = open(os.path.join(self.settings_path(), "model", "dqn_model.yaml"), 'r').read()
        self.model = model_from_yaml(yaml_string)
        self.model.load_weights(os.path.join(self.settings_path(), "model", "dqn_model_weights.hdf"))

        optimizer = RMSprop
        self.model.compile(loss=losses,
                           optimizer=optimizer(lr=self.learning_rate),
                           metrics=['accuracy'])

    def save_model(self):
        yaml_string = self.model.to_yaml()
        savedir = os.path.join(self.settings_path(), "model")
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        open(os.path.join(savedir, "dqn_model.yaml"), 'w').write(yaml_string)
        self.model.save_weights(os.path.join(savedir, "dqn_model_weights.hdf"))
