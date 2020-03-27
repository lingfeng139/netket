import abc
import numpy as _np
from netket import random as _random


class AbstractHilbert(abc.ABC):
    """Abstract class for NetKet hilbert objects"""

    @property
    @abc.abstractmethod
    def size(self):
        r"""int: The total number number of spins."""
        return NotImplementedError

    @property
    @abc.abstractmethod
    def is_discrete(self):
        r"""bool: Whether the hilbert space is discrete."""
        return NotImplementedError

    @property
    @abc.abstractmethod
    def local_size(self):
        r"""int: Size of the local degrees of freedom that make the total hilbert space."""
        return NotImplementedError

    @property
    @abc.abstractmethod
    def local_states(self):
        r"""list[float]: A list of discreet local quantum numbers."""
        return NotImplementedError

    def random_vals(self, out=None, rgen=None):
        r"""Member function generating uniformely distributed local random states.

        Args:
            out: If provided, the random quantum numbers will be inserted into this array.
                 It should be of the appropriate shape and dtype.
            rgen: The random number generator. If None, the global
                  NetKet random number generator is used.

        Examples:
           Test that a new random state is a possible state for the hilbert
           space.

           >>> import netket as nk
           >>> import numpy as np
           >>> hi = nk.hilbert.Boson(n_max=3, graph=nk.graph.Hypercube(length=5, n_dim=1))
           >>> rstate = np.zeros(hi.size)
           >>> rg = nk.utils.RandomEngine(seed=1234)
           >>> hi.random_vals(rstate, rg)
           >>> local_states = hi.local_states
           >>> print(rstate[0] in local_states)
           True
           """

        # Default version for discrete hilbert spaces without constraints
        # More specialized initializations can be defined in the derived classes
        if(self.is_discrete):
            if out is None:
                out = _np.empty(self._size)

            if(rgen is None):
                rgen = _random

            for i in range(self._size):
                rs = rgen.randint(0, self._local_size)
                out[i] = self.local_states[rs]

            return out

        return NotImplementedError

    def numbers_to_states(self, numbers, out=None):
        r"""Returns the quantum numbers corresponding to the n-th basis state
        for input n. n is an array of integer indices such that numbers[k]=Index(states[k]).
        Throws an exception iff the space is not indexable.
        Args:
            numbers: Batch of input numbers to be converted into arrays of quantum numbers.
            out: Array of quantum numbers corresponding to numbers.
                 If None, memory is allocated.
        """
        return NotImplementedError

    def number_to_state(self, number):
        return self.numbers_to_states(_np.atleast_1d(number))

    def states_to_numbers(self, states, out=None):
        r"""Returns the basis state number corresponding to given quantum states.
        The states are given in a batch, such that states[k] has shape (hilbert.size).
        Throws an exception iff the space is not indexable.
        Args:
            states: Batch of states to be converted into the corresponding integers.
            out: Array of integers such that out[k]=Index(states[k]).
                 If None, memory is allocated.
        """
        return NotImplementedError

    @property
    def n_states(self):
        r"""int: The total dimension of the many-body Hilbert space.
        Throws an exception iff the space is not indexable."""
        return NotImplementedError

    def states(self):
        r"""Returns an iterator over all valid configurations of the Hilbert space.
        Throws an exception iff the space is not indexable.
        Iterating over all states with this method is typically inefficient,
        and ```all_states``` should be prefered.

        """
        for i in range(self.n_states):
            yield self.number_to_state(i).reshape(-1)

    def all_states(self, out=None):
        r"""Returns all valid states of the Hilbert space.
        Throws an exception iff the space is not indexable.
        Args:
            batch_size: If 'all' or None, all valid states in the Hilbert space are returned,
                        otherwise an iterator yielding batch_size states at the time.
                        If batch_size is not an integer multiple of the total number of states,
                        an error is returned.
            out: Optionally pre-allocated output.
        """
        numbers = _np.arange(0, self.n_states, dtype=_np.int64)
        return self.numbers_to_states(numbers, out)

    def is_indexable(self):
        if(not self.is_discrete):
            return False

        max_states = (_np.iinfo(_np.intp).max)
        log_max = _np.log(max_states)

        return self.size * _np.log(self.local_size) <= log_max
