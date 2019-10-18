import sys

import numpy as _np

import netket as _nk
from netket._core import deprecated
from netket.operator import local_values as _local_values
from netket.stats import statistics as _statistics, covariance_sv as _covariance_sv
from netket.utils import subtract_mean as _subtract_mean

import json


def info(obj, depth=None):
    if hasattr(obj, "info"):
        return obj.info(depth)
    else:
        return str(obj)


def make_optimizer_fn(arg, ma):
    """
    Utility function to create the optimizer step function for VMC drivers.

    It currently supports three kinds of inputs:

    1. A NetKet optimizer, i.e., a subclass of `netket.optimizer.Optimizer`.

    2. A 3-tuple (init, update, get) of optimizer functions as used by the JAX
       optimizer submodule (jax.experimental.optimizers).

       The update step p0 -> p1 with bare step dp is computed as
            x0 = init(p0)
            x1 = update(i, dp, x1)
            p1 = get(x1)

    3. A single function update with signature p1 = update(i, dp, p0) returning the
       updated parameter value.
    """
    if isinstance(arg, tuple) and len(arg) == 3:
        init, update, get = arg

        def optimize_fn(i, grad, p):
            x0 = init(p)
            x1 = update(i, grad, x0)
            return get(x1)

        desc = "JAX-like optimizer"
        return optimize_fn, desc

    elif issubclass(type(arg), _nk.optimizer.Optimizer):

        arg.init(ma.n_par, ma.is_holomorphic)

        def optimize_fn(_, grad, p):
            arg.update(grad, p)
            return p

        desc = info(arg)
        return optimize_fn, desc

    elif callable(arg):
        import inspect

        sig = inspect.signature(arg)
        if not len(sig.parameters) == 3:
            raise ValueError(
                "Expected netket.optimizer.Optimizer subclass, JAX optimizer, "
                + " or callable f(i, grad, p); got callable with signature {}".format(
                    sig
                )
            )
        desc = "{}{}".format(arg.__name__, sig)
        return arg, desc
    else:
        raise ValueError(
            "Expected netket.optimizer.Optimizer subclass, JAX optimizer, "
            + " or callable f(i, grad, p); got {}".format(arg)
        )


class VmcDriver(object):
    """
    Driver class for Energy minimization using Variational Monte Carlo (VMC).
    """

    def __init__(
        self, hamiltonian, sampler, optimizer, n_samples, n_discard=None, sr=None
    ):
        """
        Initializes the driver class.

        Args:
            hamiltonian: The Hamiltonian of the system.
            sampler: The Monte Carlo sampler.
            optimizer: Determines how optimization steps are performed given the
                bare energy gradient. This parameter supports three different kinds of inputs,
                which are described in the docs of `make_optimizer_fn`.
            n_samples: Number of Markov Chain Monte Carlo sweeps to be
                performed at each step of the optimization.
            n_discard (int, optional): Number of sweeps to be discarded at the
                beginning of the sampling, at each step of the optimization.
                Defaults to 10% of n_samples.
            sr (SR, optional): Determines whether and how stochastic reconfiguration
                is applied to the bare energy gradient before performing applying
                the optimizer. If this parameter is not passed or None, SR is not used.

        Example:
            Optimizing a 1D wavefunction with Variational Monte Carlo.

            ```python
            >>> import netket as nk
            >>> SEED = 3141592
            >>> g = nk.graph.Hypercube(length=8, n_dim=1)
            >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
            >>> ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
            >>> ma.init_random_parameters(seed=SEED, sigma=0.01)
            >>> ha = nk.operator.Ising(hi, h=1.0)
            >>> sa = nk.sampler.MetropolisLocal(machine=ma)
            >>> sa.seed(SEED)
            >>> op = nk.optimizer.Sgd(learning_rate=0.1)
            >>> vmc = nk.Vmc(ha, sa, op, 200)
            ```
        """
        self._ham = hamiltonian
        self._machine = sampler.machine
        self._sampler = sampler
        self._sr = sr
        self._stats = None

        self._optimizer_step, self._optimizer_desc = make_optimizer_fn(
            optimizer, self._machine
        )

        self._npar = self._machine.n_par

        if n_samples <= 0:
            raise ValueError(
                "Invalid number of samples: n_samples={}".format(n_samples)
            )
        if n_discard is not None and n_discard < 0:
            raise ValueError(
                "Invalid number of discarded samples: n_discard={}".format(
                    n_discard)
            )

        self._n_chains = sampler.n_chains

        self._n_samples = int(_np.ceil((n_samples / self._n_chains)))
        self._n_samples_node = int(_np.ceil(self._n_samples / _nk.MPI.size()))

        self.n_discard = n_discard if n_discard else n_samples // 10

        self._obs = {}

        self.step_count = 0
        self._samples = _np.ndarray(
            (self._n_samples_node, self._n_chains, hamiltonian.hilbert.size)
        )
        self._logvals = _np.ndarray(
            (self._n_samples_node, self._n_chains), dtype=_np.complex128
        )
        self._der_logs = _np.ndarray(
            (self._n_samples_node, self._n_chains, self._npar), dtype=_np.complex128
        )

    def advance(self, n_steps=1):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        for _ in range(n_steps):

            self._sampler.reset()

            # Burnout phase
            for _ in range(self.n_discard):
                self._sampler.sweep()

            # Generate samples
            for i in range(self._samples.shape[0]):
                self._sampler.sweep()
                self._samples[i], self._logvals[i] = self._sampler.current_state

                # Compute Log derivatives
                self._der_logs[i] = self._machine.der_log(self._samples[i])

            # Center the log derivatives
            _subtract_mean(self._der_logs)

            # Estimate energy
            eloc, self._stats = self._get_mc_stats(self._ham)

            # Estimate energy gradient
            grad = _covariance_sv(eloc, self._der_logs, center_s=False)

            # Perform update
            if self._sr:
                dp = _np.empty(self._npar, dtype=_np.complex128)
                # flatten MC chain dimensions:
                derlogs = self._der_logs.reshape(-1, self._der_logs.shape[-1])
                self._sr.compute_update(derlogs, grad, dp)
                derlogs = self._der_logs.reshape(
                    self._n_samples_node, self._n_chains, self._npar
                )
            else:
                dp = grad

            self._machine.parameters = self._optimizer_step(
                self.step_count, dp, self._machine.parameters
            )

            self.step_count += 1

    def iter(self, n_steps, step=1):
        """
        Returns a generator which advances the VMC optimization, yielding
        after every `step_size` steps.

        Args:
            n_iter (int=None): The total number of steps to perform.
            step_size (int=1): The number of internal steps the simulation
                is advanced every turn.

        Yields:
            int: The current step.
        """
        for _ in range(0, n_steps, step):
            self.advance(step)
            yield self.step_count

    def add_observable(self, name, obs):
        """
        Add an observables to the set of observables that will be computed by default
        in get_obervable_stats.
        """
        self._obs[name] = obs

    def get_observable_stats(self, observables=None, include_energy=True):
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.

        Args:
            observables: A dictionary of the form {name: observable} or a list
                of tuples (name, observable) for which statistics should be computed.
                If observables is None or not passed, results for those observables
                added to the driver by add_observables are computed.
            include_energy: Whether to include the energy estimate (which is already
                computed as part of the VMC step) in the result.

        Returns:
            A dictionary of the form {name: stats} mapping the observable names in
            the input to corresponding Stats objects.

            If `include_energy` is true, then the result will further contain the
            energy statistics with key "Energy".
        """
        if not observables:
            observables = self._obs
        r = {"Energy": self._stats} if include_energy else {}
        r.update({name: self._get_mc_stats(obs) for name, obs in observables})
        return r

    def _get_mc_stats(self, op):
        loc = _local_values(op, self._machine, self._samples, self._logvals)
        return loc, _statistics(loc)

    def __repr__(self):
        return "Vmc(step_count={}, n_samples={}, n_discard={})".format(
            self.step_count, self.n_samples, self.n_discard
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Hamiltonian", self._ham),
                ("Machine", self._machine),
                ("Optimizer", self._optimizer_desc),
                ("SR solver", self._sr),
            ]
        ]
        return "\n  ".join([str(self)] + lines)

    # The following functions are for backward compatibility with
    # json output and will be removed at some point

    def _add_to_json_log(self):

        stats = self.get_observable_stats()
        self._json_out["Output"].append({})
        self._json_out["Output"][-1] = {}
        json_iter = self._json_out["Output"][-1]
        json_iter["Iteration"] = self.step_count
        for key, value in stats.items():
            st = value.asdict()
            st["Mean"] = st["Mean"].real
            json_iter[key] = st

    def _init_json_log(self):

        self._json_out = {}
        self._json_out["Output"] = []

    def run(
        self, output_prefix, n_iter, step_size=1, save_params_every=50, write_every=50
    ):
        self._init_json_log()

        for k in range(n_iter):
            self.advance(step_size)

            self._add_to_json_log()
            if k % write_every == 0 or k == n_iter - 1:
                if _nk.MPI.rank() == 0:
                    with open(output_prefix + ".log", "w") as outfile:
                        json.dump(self._json_out, outfile)
