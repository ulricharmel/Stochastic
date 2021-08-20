# adapted from https://github.com/google/jax/issues/3296
# trying to implement an adam optimisation with different learning rates per parameters

from functools import partial
from jax import grad, ops
import jax.numpy as jnp
from jax.util import unzip2
from jax.tree_util import tree_flatten, tree_unflatten
from jax.experimental.optimizers import OptimizerState

def optimizer(opt_maker):
  def tree_opt_maker(*args, **kwargs):
    init, update, get_params = opt_maker(*args, **kwargs)

    def tree_init(x0_tree):
      x0_flat, tree = tree_flatten(x0_tree)
      initial_states = [init(x0) for x0 in x0_flat]
      states_flat, subtrees = unzip2(map(tree_flatten, initial_states))
      return OptimizerState(states_flat, tree, subtrees)

    def tree_update(i, step_size_tree, grad_tree, opt_state):
      states_flat, tree, subtrees = opt_state
      step_size_flat, _ = tree_flatten(step_size_tree)
      grad_flat, _ = tree_flatten(grad_tree)
      states = map(tree_unflatten, subtrees, states_flat)
      new_states = map(partial(update, i), step_size_flat, grad_flat, states)
      new_states_flat, _ = unzip2(map(tree_flatten, new_states))
      return OptimizerState(new_states_flat, tree, subtrees)

    def tree_get_params(opt_state):
      states_flat, tree, subtrees = opt_state
      states = map(tree_unflatten, subtrees, states_flat)
      params = map(get_params, states)
      return tree_unflatten(tree, params)

    return tree_init, tree_update, tree_get_params
  return tree_opt_maker


@optimizer
def adam(b1=0.9, b2=0.999, eps=1e-8):
  def init(x0):
    m0 = jnp.zeros_like(x0)
    v0 = jnp.zeros_like(x0)
    return x0, m0, v0
  def update(i, step_size, g, state):
    x, m, v = state
    m = (1 - b1) * g + b1 * m
    v = (1 - b2) * jnp.square(g) + b2 * v
    mhat = m / (1 - b1 ** (i + 1))
    vhat = v / (1 - b2 ** (i + 1))
    x = x - step_size * mhat / (jnp.sqrt(vhat) + eps)
    return x, m, v
  def get_params(state):
    x, m, v = state
    return x
  return init, update, get_params

@optimizer
def momentum(mass=0.8):
  def init(x0):
    v0 = jnp.zeros_like(x0)
    return x0, v0
  def update(i, step_size, g, state):
    x, velocity = state
    velocity = mass * velocity + g
    x = x - step_size * velocity
    return x, velocity
  def get_params(state):
    x, _ = state
    return x
  return init, update, get_params


@optimizer
def sgd():
  def init(x0):
    return x0
  def update(i, step_size, g, x):
    x = x - step_size *g
    return x
  def get_params(state):
    x = state
    return x
  return init, update, get_params
