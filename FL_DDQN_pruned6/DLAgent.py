import random
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
from tensorflow.keras import backend as k_backend
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule


class DQNAgent:
    def __init__(self, state_size, action_size, replay_memory_size,
                 mini_batch_size, replace_target_period, gamma, epsilon_min):
        self.state_size = state_size
        self.action_size = action_size
        self.replay_memory_size = replay_memory_size
        self.mini_batch_size = mini_batch_size
        self.memory = np.zeros((self.replay_memory_size, state_size * 2 + 2))
        self.gamma = gamma  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.learning_rate = 0.005  # 0.0005
        self.epsilon_decrement = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.memory_counter = 0
        self.learning_step = 0
        self.replace_target_period = replace_target_period

    # self-define loss function
    @staticmethod
    def _loss(target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        # return k_backend.mean(k_backend, axis=-1)
        return k_backend.mean(k_backend.square(error), axis=-1)

    def _build_model(self):
        pruning_params = {
            'pruning_schedule':
                pruning_schedule.PolynomialDecay(
                    initial_sparsity=0,
                    final_sparsity=0.6,
                    begin_step=200,
                    end_step=2000,
                    frequency=20)
        }

        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(
            Dense(500, input_dim=self.state_size, activation='tanh',
                  kernel_initializer=initializers.RandomNormal(mean=5, stddev=0.03)))
        # model.add(
        #     Dense(600, activation='linear')
        # )
        #
        # model.add(
        #     Dense(600, activation='linear')
        # )
        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model = prune.prune_low_magnitude(
            model, **pruning_params)

        # model.compile(loss=self._loss,
        #               optimizer=Adam(lr=self.learning_rate))
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.learning_rate))

        return model


    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    # def remember(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done))

    def remember(self, state, action, reward, next_state):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % self.replay_memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # random action, exploration
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.reshape(state, [1, self.state_size]))
        return np.argmax(act_values[0])  # returns action

    def replay(self):
        if self.memory_counter > self.replay_memory_size:
            sample_index = np.random.choice(self.replay_memory_size, size=self.mini_batch_size)
        else:
            # when the memory is not full, avoid sampling transitions of zeros
            sample_index = np.random.choice(self.memory_counter, size=self.mini_batch_size)
        batch_memory = self.memory[sample_index, :]

        # obtain the samples
        state_array = batch_memory[:, :self.state_size]
        target_array = self.model.predict(state_array)
        next_target_array = self.target_model.predict(batch_memory[:, -self.state_size:])
        # obtain the target q values for the comparison with eval q values
        for idx_mini_batch in range(self.mini_batch_size):
            target_array[idx_mini_batch, batch_memory[idx_mini_batch, self.state_size].astype(int)] = \
                batch_memory[idx_mini_batch, self.state_size + 1] + \
                self.gamma * np.amax(next_target_array[idx_mini_batch])

        callbacks = [
            pruning_callbacks.UpdatePruningStep(),
            # pruning_callbacks.PruningSummaries('../../other-dir/6/')
        ]

        # mini-batch
        cur_history = self.model.fit(state_array, target_array, epochs=1,
                                     batch_size=self.mini_batch_size, verbose=0, callbacks=callbacks)

        self.epsilon = self.epsilon - self.epsilon_decrement \
            if self.epsilon > self.epsilon_min else self.epsilon_min

        # record the learning step already been done
        self.learning_step = self.learning_step + 1
        # replace target network periodically
        if self.learning_step % self.replace_target_period == 0:
            self.update_target_model()

        return cur_history.history["loss"][0]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
