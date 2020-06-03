# Copyright 2020 The Simons Foundation, Inc. - All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools as _itertools
import json

from mpi4py import MPI

import numpy as _np
import scipy.integrate as _scint
import scipy
from tqdm import tqdm

import netket as _nk
from netket.logging import JsonLog as _JsonLog
from netket.vmc_common import tree_map, trees2_map

from netket._vmc import Vmc
from netket._steadystate import SteadyState
from netket.machine.density_matrix import AbstractDensityMatrix
from netket.abstract_variational_driver import AbstractVariationalDriver

from .operator import (
    local_values as _local_values,
    der_local_values as _der_local_values,
)
from netket.stats import (
    statistics as _statistics,
    mean as _mean,
    sum_inplace as _sum_inplace,
)

from functools import singledispatch

_NaN = float("NaN")


# VMC
@singledispatch
def _time_evo(self):
    raise ErrorException("unknown type?")


@_time_evo.register(Vmc)
def _time_evo_vmc(self):
    """
    Performs a number of VMC optimization steps.

    Args:
        n_steps (int): Number of steps to perform.
    """

    self._sampler.reset()
    self._sampler.generate_samples(self._n_discard)
    self._samples = self._sampler.generate_samples(
        self._n_samples_node, samples=self._samples
    )
    eloc, self._loss_stats = self._get_mc_stats(self._ham)
    eloc -= _mean(eloc)
    samples_r = self._samples.reshape((-1, self._samples.shape[-1]))
    eloc_r = eloc.reshape(-1, 1)
    self._grads, self._jac = self._machine.vector_jacobian_prod(
        samples_r, eloc_r / self._n_samples, self._grads, return_jacobian=True
    )
    self._grads = tree_map(_sum_inplace, self._grads)
    self._grads = tree_map(lambda x: -1.0j * x, self._grads)
    self._dp = self._sr.compute_update(self._jac, self._grads, self._dp)
    return self._dp


@_time_evo.register(SteadyState)
def _time_evo_vmc(self):
    self._obs_samples_valid = False
    self._sampler.reset()
    self._sampler.generate_samples(self._n_discard)
    self._samples = self._sampler.generate_samples(
        self._n_samples_node, samples=self._samples
    )
    self._lloc, self._loss_stats = self._get_mc_superop_stats(self._lind)
    self._loss1_stats = _statistics(self._lloc.T)
    lloc = self._lloc - _mean(self._lloc)
    samples_r = self._samples.reshape((-1, self._samples.shape[-1]))
    lloc_r = lloc.reshape(-1, 1)
    self._grads, self._jac = self._machine.vector_jacobian_prod(
        samples_r, lloc_r / self._n_samples, self._grads, return_jacobian=True
    )
    self._grads = tree_map(_sum_inplace, self._grads)
    self._dp = self._sr.compute_update(self._jac, self._grads, self._dp)
    return self._dp


def create_timevo(op, sampler, *args, julia=False, **kwargs):
    machine = sampler._machine
    if not isinstance(machine, AbstractDensityMatrix):
        driver = Vmc(op, sampler, *args, optimizer=None, **kwargs)
    else:
        driver = SteadyState(op, sampler, *args, optimizer=None, **kwargs)

    flatten = machine.numpy_flatten
    unflatten = lambda w: machine.numpy_unflatten(w, machine.parameters)

    def _fun(t, w):
        machine.parameters = unflatten(w)
        dp = _time_evo(driver)
        return flatten(dp)

    if julia:

        def fun(u, p, t):
            _fun(t, w)

    else:
        fun = _fun

    return driver, fun


class TimeEvolution(AbstractVariationalDriver):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    def __init__(
        self, operator, sampler, sr, *args, **kwargs,
    ):

        super(TimeEvolution, self).__init__(
            sampler.machine, None,
        )
        driver, fun = create_timevo(operator, sampler, *args, sr=sr, **kwargs)
        self._driver = driver
        self._fun = fun

        self._t = None
        self._solver = None
        self._state = None

    def setup(self, solver, tspan, dt=None, adaptive=False, **kwargs):
        if isinstance(tspan, tuple):
            t0 = tspan[0]
            tend = tspan[-1]
        else:
            t0 = tspan

        y0 = self._machine.numpy_flatten(self._machine.parameters)

        if solver == "ode45" or solver == "dopri5" or solver == "rk45":
            if adaptive is False:
                self._integrator = _scint.RK45(
                    self._fun,
                    t0=t0,
                    y0=y0,
                    t_bound=tend,
                    max_step=dt,
                    first_step=dt,
                    rtol=_np.inf,
                    atol=_np.inf,
                )
            else:
                self._integrator = _scint.RK45(
                    self._fun, t0=t0, y0=y0, t_bound=tend, **kwargs
                )

    def advance(self, t_end=None, n_steps=None):
        """
        Advance the time propagation by `n_steps` simulation steps
        of duration `self.dt`.

           Args:
               :n_steps (int): No. of steps to advance.
        """
        if (t_end is None and n_steps is None) or (
            t_end is not None and n_steps is not None
        ):
            raise ValueError("Both specified")

        if n_steps is not None:
            for i in range(n_steps):
                self._integrator.step()
        elif t_end is not None:
            while self._integrator.t < t_end:
                self._integrator.step()

        if self._integrator.status == "failed":
            raise ...

    def iter(self, t_end, t_interval=1e-10):
        """
        Returns a generator which advances the time evolution in
        steps of `step` for a total of `n_iter` times.

        Args:
            :n_iter (int): The total number of steps.
            :step (int=1): The size of each step.

        Yields:
            :(int): The current step.
        """
        while self.t < t_end:
            _step_end = self.t + t_interval
            t0 = self.t
            while self.t < _step_end:
                if self.t == t0:
                    yield self.t

                self._integrator.step()

    def run(
        self,
        t_end=None,
        out=None,
        obs=None,
        show_progress=True,
        save_params_every=50,  # for default logger
        write_every=50,  # for default logger
        step_size=1e-10,  # for default logger
    ):
        if obs is None:
            obs = {}

        if out is None:
            print(
                "No output specified (out=[apath|nk.logging.JsonLogger(...)])."
                "Running the optimization but not saving the output."
            )

        t_b = self._integrator.t_bound
        if t_b is None or t_b is _np.inf:
            if t_end is None:
                raise ValueError("must specify t_end")

        if t_end is not None:
            self._integrator.t_bound = t_end

        # Log only non-root nodes
        if self._mynode == 0:
            # if out is a path, create an overwriting Json Log for output
            if isinstance(out, str):
                logger = _JsonLog(out, "w", save_params_every, write_every)
            else:
                logger = out
        else:
            logger = None
            show_progress = False

        with tqdm(total=self.t_end, initial=self.t, disable=not show_progress) as pbar:
            told = self.t
            for step, t in enumerate(self.iter(self.t_end, step_size)):
                # if the cost-function is defined then report it in the progress bar
                if self._loss_stats is not None:
                    itr.set_postfix_str(self._loss_name + "=" + str(self._loss_stats))

                obs_data = self.estimate(obs)

                if self._loss_stats is not None:
                    obs_data[self._loss_name] = self._loss_stats

                obs_data["t"] = t

                if logger is not None:
                    logger(step, obs_data, self.machine)

                pbar.update(t - told)
                told = t

        # flush at the end of the evolution so that final values are saved to
        # file
        if logger is not None:
            logger.flush(self.machine)

    def _forward_and_backward(self):
        integrator = self._integrator
        integrator._step()
        return integrator.f

    @property
    def dt(self):
        return self._integrator.step_size

    @property
    def t(self):
        return self._integrator.t

    @t.setter
    def t(self, t):
        self._integrator.t = t

    @property
    def t_end(self):
        return self._integrator.t_bound

    @property
    def state(self):
        return self._integrator.y

    @state.setter
    def state(self, y):
        if isinstance(y, AbstractMachine):
            y = y.numpy_flatten(y.parameters)

        self._integrator.y = y

    def _estimate_stats(self, obs):
        return self._driver.estimate(obs)

    def reset(self):
        self._driver.reset()

    def update_parameters(self, dp):
        print("using this")
        self._machine.numpy_unflatten(dp, self._machine.parameters)

    def __repr__(self):
        return "TimeEvo"

    def info(self, depth=0):
        return "stuff"
