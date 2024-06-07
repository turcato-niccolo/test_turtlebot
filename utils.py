import warnings

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from bayes_opt import BayesianOptimization
from bayes_opt.util import acq_max, UtilityFunction


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class FunctionBuffer():

    def __init__(self, input_dim, output_dim, max_size=int(1e2)):
        self.max_size = max_size
        self.memory = np.zeros((max_size, input_dim + output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.position = 0
        self.is_full = False

    def push(self, input_, output_):
        self.memory[self.position, :self.input_dim] = input_
        self.memory[self.position, self.input_dim:] = output_

        if not self.is_full and self.position == self.max_size - 1:
            self.is_full = True

        self.position = (self.position + 1) % self.max_size

    def get_batch(self):
        if self.is_full:
            return self.memory
        else:
            return self.memory[:self.position, :]

    def get_batch_X_Y(self):
        if self.is_full:
            return self.memory[:, :self.input_dim], self.memory[:, self.input_dim:]
        else:
            return self.memory[:self.position, :self.input_dim], self.memory[:self.position, self.input_dim:]


class BayesianOpt(BayesianOptimization):

    def __init__(self, f, pbounds, constraint=None, random_state=None, verbose=2, bounds_transformer=None,
                 allow_duplicate_points=False, buff_max_size=10 ** 2):
        super().__init__(f, pbounds, constraint, random_state, verbose, bounds_transformer, allow_duplicate_points)

        self.buff = FunctionBuffer(input_dim=len(list(pbounds.keys())), output_dim=1, max_size=buff_max_size)

        # Internal GP regressor rewritten
        self._gp = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0, length_scale_bounds=(0.01, 2.0)),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5)

    def register(self, input_, target):
        """Register an observation with known target.

        Parameters
        ----------
        params: dict or list
            The parameters associated with the observation.

        target: float
            Value of the target function at the observation.

        constraint_value: float or None
            Value of the constraint function at the observation, if any.
        """
        self.buff.push(input_, target)

    def suggest(self, utility_function):
        """Suggest a promising point to probe next.

        Parameters
        ----------
        utility_function:
            Surrogate function which suggests parameters to probe the target
            function at.
        """
        # Sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X, Y = self.buff.get_batch_X_Y()
            self._gp.fit(X, Y)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(ac=utility_function.utility,
                             gp=self._gp,
                             constraint=self.constraint,
                             # y_max=self._space._target_max(),
                             y_max=self._space.max(),
                             bounds=self._space.bounds,
                             random_state=self._random_state)
        # y_max_params=self._space.params_to_array(self._space.max()['params']))

        return self._space.array_to_params(suggestion)


class BiasSelectorBayesianOpt():
    def __init__(self, bounds=(0.1, 0.9), alpha=0.25):
        # Bounded region of parameter space
        pbounds = {'x': bounds}

        self.optimizer = BayesianOpt(
            f=None,
            pbounds=pbounds,
            random_state=1,
            buff_max_size=50
        )

        self.utility = UtilityFunction(kind="ucb", kappa=3.0, xi=0.0, kappa_decay=0.99)
        self.moving_avg = 0.0
        self.alpha = alpha

    def suggest(self):
        next_point_to_probe = self.optimizer.suggest(self.utility)
        print("Next point to probe is:", next_point_to_probe)
        self.utility.update_params()
        return next_point_to_probe['x']

    def register_point(self, point, value):
        residual = (value - self.moving_avg)
        self.moving_avg += self.alpha * residual
        self.optimizer.register(point, residual)  # / np.abs(self.moving_avg))
    # relative displacement from mov avg
