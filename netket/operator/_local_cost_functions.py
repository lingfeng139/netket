import jax
import numpy as _np
from functools import partial

from inspect import signature

# This is a dict that maintains a list of the batch_axes for every function to be used as a local
# cost function.
# As this is used within jitted functions where the function itself is a constant, this is a
# zero cost abstraction 🧙.
_batch_axes = {}

# Store unjitted functions in a dictionary, where keys are jitted functions, so that all
# functions only jit once (the external level). This improves compilation time.
_unjitted_cost_fun = {}


def local_cost_function(fun, static_argnums=0, batch_axes=None):
    """
    @local_cost_function(fun, static_argnums=0, batch_axes=automatic)

    A decorator to be used to define a local cost function and it's gradient. The function
    to be decorated must be a jax-compatible function, that takes the following positional
    arguments:
     - The neural-network function evaluating a single input and returning a scalar output.
     - A pytree of parameters for the neural network function. Gradients will be computed with
     respect to this argument
     - N additional positional arguments (non static) containing any additional data.

    In order to support batching, one must also define the batch_axes variable according
    to jax vmap documentation. By default `batch_axesw=(None, None, 0...)`, meaning that
    no batching is performed for the first two arguments (the network and the parameters)
    and batching is performed along the 0-th dimension of all arguments.

    An example is provided below:
    ```python
    @partial(local_cost_function, static_argnums=0, batch_axes=(None, None, 0, 0, 0))
    def local_energy_kernel(logpsi, pars, vp, mel, v):
        return jax.numpy.sum(mel * jax.numpy.exp(logpsi(pars, vp) - logpsi(pars, v)))
    ```
    """
    jitted_fun = jax.jit(fun, static_argnums=static_argnums)

    ig = signature(jitted_fun)
    npars = len(ig.parameters)
    if npars < 2:
        raise ValueError("Local cost functions should have at least 2 parameters.")

    # If batch_axes is not specified, than assume that all parameters except the first two
    # are to be batched upon. The first two would be the function itself and its weights.
    if batch_axes is None:
        batch_axes = (None, None) + tuple([None for _ in range(npars - 2)])

    _batch_axes[jitted_fun] = batch_axes
    _unjitted_cost_fun[jitted_fun] = fun

    return jitted_fun


# In the following, all functions assume that the arguments are passed in that order:
# 0 - (static) local_cost_fun (for example local_energy kernel).
# 1 - (static) the nn function
# 2 - weights for the nn function in pytree format. Those will be the directions of the gradient
# 3 - various parameters
# Also assumes that args 1..N are the args (in that order) of local_cost_fun

# Unjitted-version. Should not be exported.
# It's defined so that we dont double-jit some functions leading to a slight
# speedup in compilation time
def _der_local_cost_function(local_cost_fun, logpsi, pars, *args):
    der_local_cost_fun = jax.grad(
        _unjitted_cost_fun[local_cost_fun], argnums=1, holomorphic=True
    )
    return der_local_cost_fun(logpsi, pars, *args)


der_local_cost_function = jax.jit(_der_local_cost_function, static_argnums=(0, 1))


@partial(jax.jit, static_argnums=(0, 1))
def ders_local_cost_function(local_cost_fun, logpsi, pars, *args):
    """
    ders_local_cost_function(local_cost_fun, logpsi, pars, *args)

    Function to compute the gradient of the `local_cost_fun` function with respect
    to it's parameters `pars`, vmapped along `*args` 0-th dimension.
    This function is fully jitted upon first call.

    Args:
        local_cost_fun: the cost function
        logpsi: the parametric function encoding the quantum state
        pars: the variational parameters representing the state
        *args: additional arguments

    Returns:
        the gradient with respect to `pars`
    """
    der_local_cost_funs = (
        jax.vmap(
            _der_local_cost_function, in_axes=_batch_axes[local_cost_fun], out_axes=0
        ),
    )
    return der_local_cost_funs(local_cost_fun, logpsi, pars, *args)


###
# same assumptions as above
def _local_cost_and_grad_function(local_cost_fun, logpsi, pars, *args):
    der_local_cost_fun = jax.value_and_grad(
        _unjitted_cost_fun[local_cost_fun], argnums=1, holomorphic=True
    )
    return der_local_cost_fun(logpsi, pars, *args)


local_cost_and_grad_function = jax.jit(_local_cost_and_grad_function, static_argnums=0)


@partial(jax.jit, static_argnums=(0, 1))
def local_costs_and_grads_function(local_cost_fun, logpsi, pars, *args):
    """
    local_costs_and_grads_function(local_cost_fun, logpsi, pars, *args)

    Function to compute the value and the gradient of the `local_cost_fun` function
    with respect to it's parameters `pars`, vmapped along `*args` 0-th dimension.
    This function is fully jitted upon first call.

    Args:
        local_cost_fun: the cost function
        logpsi: the parametric function encoding the quantum state
        pars: the variational parameters representing the state
        *args: additional arguments

    Returns:
        the value of the local_cost_function for every `*args` input
        the gradient with respect to `pars`
    """

    local_costs_and_grads_fun = jax.vmap(
        _local_cost_and_grad_function,
        in_axes=_batch_axes[local_cost_fun],
        out_axes=(0, 0),
    )
    return local_costs_and_grads_fun(local_cost_fun, logpsi, pars, *args)


####### Defined functions
@partial(local_cost_function, static_argnums=0, batch_axes=(None, None, 0, 0, 0))
def local_energy_kernel(logpsi, pars, vp, mel, v):
    return jax.numpy.sum(mel * jax.numpy.exp(logpsi(pars, vp) - logpsi(pars, v)))
