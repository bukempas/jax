#jax
JAX: Autograd and XLA
Continuous integration PyPI version

Quickstart | Transformations | Install guide | Neural net libraries | Change logs | Reference docs

What is JAX?
JAX is Autograd and XLA, brought together for high-performance machine learning research.

With its updated version of Autograd, JAX can automatically differentiate native Python and NumPy functions. It can differentiate through loops, branches, recursion, and closures, and it can take derivatives of derivatives of derivatives. It supports reverse-mode differentiation (a.k.a. backpropagation) via grad as well as forward-mode differentiation, and the two can be composed arbitrarily to any order.

What’s new is that JAX uses XLA to compile and run your NumPy programs on GPUs and TPUs. Compilation happens under the hood by default, with library calls getting just-in-time compiled and executed. But JAX also lets you just-in-time compile your own Python functions into XLA-optimized kernels using a one-function API, jit. Compilation and automatic differentiation can be composed arbitrarily, so you can express sophisticated algorithms and get maximal performance without leaving Python. You can even program multiple GPUs or TPU cores at once using pmap, and differentiate through the whole thing.

Dig a little deeper, and you'll see that JAX is really an extensible system for composable function transformations. Both grad and jit are instances of such transformations. Others are vmap for automatic vectorization and pmap for single-program multiple-data (SPMD) parallel programming of multiple accelerators, with more to come.

This is a research project, not an official Google product. Expect bugs and sharp edges. Please help by trying it out, reporting bugs, and letting us know what you think!

import jax.numpy as jnp
from jax import grad, jit, vmap

def predict(params, inputs):
  for W, b in params:
    outputs = jnp.dot(inputs, W) + b
    inputs = jnp.tanh(outputs)  # inputs to the next layer
  return outputs                # no activation on last layer

def loss(params, inputs, targets):
  preds = predict(params, inputs)
  return jnp.sum((preds - targets)**2)

grad_loss = jit(grad(loss))  # compiled gradient evaluation function
perex_grads = jit(vmap(grad_loss, in_axes=(None, 0, 0)))  # fast per-example grads
