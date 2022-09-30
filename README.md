# jax
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
Contents
Quickstart: Colab in the Cloud
Transformations
Current gotchas
Installation
Neural net libraries
Citing JAX
Reference documentation
Quickstart: Colab in the Cloud
Jump right in using a notebook in your browser, connected to a Google Cloud GPU. Here are some starter notebooks:

The basics: NumPy on accelerators, grad for differentiation, jit for compilation, and vmap for vectorization
Training a Simple Neural Network, with TensorFlow Dataset Data Loading
JAX now runs on Cloud TPUs. To try out the preview, see the Cloud TPU Colabs.

For a deeper dive into JAX:

The Autodiff Cookbook, Part 1: easy and powerful automatic differentiation in JAX
Common gotchas and sharp edges
See the full list of notebooks.
You can also take a look at the mini-libraries in jax.example_libraries, like stax for building neural networks and optimizers for first-order stochastic optimization, or the examples.

Transformations
At its core, JAX is an extensible system for transforming numerical functions. Here are four transformations of primary interest: grad, jit, vmap, and pmap.

Automatic differentiation with grad
JAX has roughly the same API as Autograd. The most popular function is grad for reverse-mode gradients:

from jax import grad
import jax.numpy as jnp

def tanh(x):  # Define a function
  y = jnp.exp(-2.0 * x)
  return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)  # Obtain its gradient function
print(grad_tanh(1.0))   # Evaluate it at x = 1.0
# prints 0.4199743
You can differentiate to any order with grad.

print(grad(grad(grad(tanh)))(1.0))
# prints 0.62162673
For more advanced autodiff, you can use jax.vjp for reverse-mode vector-Jacobian products and jax.jvp for forward-mode Jacobian-vector products. The two can be composed arbitrarily with one another, and with other JAX transformations. Here's one way to compose those to make a function that efficiently computes full Hessian matrices:

from jax import jit, jacfwd, jacrev

def hessian(fun):
  return jit(jacfwd(jacrev(fun)))
As with Autograd, you're free to use differentiation with Python control structures:

def abs_val(x):
  if x > 0:
    return x
  else:
    return -x

abs_val_grad = grad(abs_val)
print(abs_val_grad(1.0))   # prints 1.0
print(abs_val_grad(-1.0))  # prints -1.0 (abs_val is re-evaluated)
See the reference docs on automatic differentiation and the JAX Autodiff Cookbook for more.

Compilation with jit
You can use XLA to compile your functions end-to-end with jit, used either as an @jit decorator or as a higher-order function.

import jax.numpy as jnp
from jax import jit

def slow_f(x):
  # Element-wise ops see a large benefit from fusion
  return x * x + x * 2.0

x = jnp.ones((5000, 5000))
fast_f = jit(slow_f)
%timeit -n10 -r3 fast_f(x)  # ~ 4.5 ms / loop on Titan X
%timeit -n10 -r3 slow_f(x)  # ~ 14.5 ms / loop (also on GPU via JAX)
You can mix jit and grad and any other JAX transformation however you like.

Using jit puts constraints on the kind of Python control flow the function can use; see the Gotchas Notebook for more.

Auto-vectorization with vmap
vmap is the vectorizing map. It has the familiar semantics of mapping a function along array axes, but instead of keeping the loop on the outside, it pushes the loop down into a function’s primitive operations for better performance.

Using vmap can save you from having to carry around batch dimensions in your code. For example, consider this simple unbatched neural network prediction function:

def predict(params, input_vec):
  assert input_vec.ndim == 1
  activations = input_vec
  for W, b in params:
    outputs = jnp.dot(W, activations) + b  # `activations` on the right-hand side!
    activations = jnp.tanh(outputs)        # inputs to the next layer
  return outputs                           # no activation on last layer
We often instead write jnp.dot(activations, W) to allow for a batch dimension on the left side of activations, but we’ve written this particular prediction function to apply only to single input vectors. If we wanted to apply this function to a batch of inputs at once, semantically we could just write

from functools import partial
predictions = jnp.stack(list(map(partial(predict, params), input_batch)))
But pushing one example through the network at a time would be slow! It’s better to vectorize the computation, so that at every layer we’re doing matrix-matrix multiplication rather than matrix-vector multiplication.

The vmap function does that transformation for us. That is, if we write

from jax import vmap
predictions = vmap(partial(predict, params))(input_batch)
# or, alternatively
predictions = vmap(predict, in_axes=(None, 0))(params, input_batch)
then the vmap function will push the outer loop inside the function, and our machine will end up executing matrix-matrix multiplications exactly as if we’d done the batching by hand.

It’s easy enough to manually batch a simple neural network without vmap, but in other cases manual vectorization can be impractical or impossible. Take the problem of efficiently computing per-example gradients: that is, for a fixed set of parameters, we want to compute the gradient of our loss function evaluated separately at each example in a batch. With vmap, it’s easy:

per_example_gradients = vmap(partial(grad(loss), params))(inputs, targets)
Of course, vmap can be arbitrarily composed with jit, grad, and any other JAX transformation! We use vmap with both forward- and reverse-mode automatic differentiation for fast Jacobian and Hessian matrix calculations in jax.jacfwd, jax.jacrev, and jax.hessian.

SPMD programming with pmap
For parallel programming of multiple accelerators, like multiple GPUs, use pmap. With pmap you write single-program multiple-data (SPMD) programs, including fast parallel collective communication operations. Applying pmap will mean that the function you write is compiled by XLA (similarly to jit), then replicated and executed in parallel across devices.

Here's an example on an 8-GPU machine:

from jax import random, pmap
import jax.numpy as jnp

# Create 8 random 5000 x 6000 matrices, one per GPU
keys = random.split(random.PRNGKey(0), 8)
mats = pmap(lambda key: random.normal(key, (5000, 6000)))(keys)

# Run a local matmul on each device in parallel (no data transfer)
result = pmap(lambda x: jnp.dot(x, x.T))(mats)  # result.shape is (8, 5000, 5000)

# Compute the mean on each device in parallel and print the result
print(pmap(jnp.mean)(result))
# prints [1.1566595 1.1805978 ... 1.2321935 1.2015157]
In addition to expressing pure maps, you can use fast collective communication operations between devices:

from functools import partial
from jax import lax

@partial(pmap, axis_name='i')
def normalize(x):
  return x / lax.psum(x, 'i')

print(normalize(jnp.arange(4.)))
# prints [0.         0.16666667 0.33333334 0.5       ]
You can even nest pmap functions for more sophisticated communication patterns.

It all composes, so you're free to differentiate through parallel computations:

from jax import grad

@pmap
def f(x):
  y = jnp.sin(x)
  @pmap
  def g(z):
    return jnp.cos(z) * jnp.tan(y.sum()) * jnp.tanh(x).sum()
  return grad(lambda w: jnp.sum(g(w)))(x)

print(f(x))
# [[ 0.        , -0.7170853 ],
#  [-3.1085174 , -0.4824318 ],
#  [10.366636  , 13.135289  ],
#  [ 0.22163185, -0.52112055]]

print(grad(lambda x: jnp.sum(f(x)))(x))
# [[ -3.2369726,  -1.6356447],
#  [  4.7572474,  11.606951 ],
#  [-98.524414 ,  42.76499  ],
#  [ -1.6007166,  -1.2568436]]
When reverse-mode differentiating a pmap function (e.g. with grad), the backward pass of the computation is parallelized just like the forward pass.

See the SPMD Cookbook and the SPMD MNIST classifier from scratch example for more.

Current gotchas
For a more thorough survey of current gotchas, with examples and explanations, we highly recommend reading the Gotchas Notebook. Some standouts:

JAX transformations only work on pure functions, which don't have side-effects and respect referential transparency (i.e. object identity testing with is isn't preserved). If you use a JAX transformation on an impure Python function, you might see an error like Exception: Can't lift Traced... or Exception: Different traces at same level.
In-place mutating updates of arrays, like x[i] += y, aren't supported, but there are functional alternatives. Under a jit, those functional alternatives will reuse buffers in-place automatically.
Random numbers are different, but for good reasons.
If you're looking for convolution operators, they're in the jax.lax package.
JAX enforces single-precision (32-bit, e.g. float32) values by default, and to enable double-precision (64-bit, e.g. float64) one needs to set the jax_enable_x64 variable at startup (or set the environment variable JAX_ENABLE_X64=True). On TPU, JAX uses 32-bit values by default for everything except internal temporary variables in 'matmul-like' operations, such as jax.numpy.dot and lax.conv. Those ops have a precision parameter which can be used to simulate true 32-bit, with a cost of possibly slower runtime.
Some of NumPy's dtype promotion semantics involving a mix of Python scalars and NumPy types aren't preserved, namely np.add(1, np.array([2], np.float32)).dtype is float64 rather than float32.
Some transformations, like jit, constrain how you can use Python control flow. You'll always get loud errors if something goes wrong. You might have to use jit's static_argnums parameter, structured control flow primitives like lax.scan, or just use jit on smaller subfunctions.
Installation
JAX is written in pure Python, but it depends on XLA, which needs to be installed as the jaxlib package. Use the following instructions to install a binary package with pip or conda, or to build JAX from source.

We support installing or building jaxlib on Linux (Ubuntu 16.04 or later) and macOS (10.12 or later) platforms.

Windows users can use JAX on CPU and GPU via the Windows Subsystem for Linux. In addition, there is some initial community-driven native Windows support, but since it is still somewhat immature, there are no official binary releases and it must be built from source for Windows. For an unofficial discussion of native Windows builds, see also the Issue #5795 thread.

pip installation: CPU
To install a CPU-only version of JAX, which might be useful for doing local development on a laptop, you can run

pip install --upgrade pip
pip install --upgrade "jax[cpu]"
On Linux, it is often necessary to first update pip to a version that supports manylinux2014 wheels. These pip installations do not work with Windows, and may fail silently; see above.

pip installation: GPU (CUDA)
If you want to install JAX with both CPU and NVidia GPU support, you must first install CUDA and CuDNN, if they have not already been installed. Unlike some other popular deep learning systems, JAX does not bundle CUDA or CuDNN as part of the pip package.

JAX provides pre-built CUDA-compatible wheels for Linux only, with CUDA 11.1 or newer, and CuDNN 8.0.5 or newer. Other combinations of operating system, CUDA, and CuDNN are possible, but require building from source.

CUDA 11.1 or newer is required.
The supported cuDNN versions for the prebuilt wheels are:
cuDNN 8.2 or newer. We recommend using the cuDNN 8.2 wheel if your cuDNN installation is new enough, since it supports additional functionality.
cuDNN 8.0.5 or newer.
You must use an NVidia driver version that is at least as new as your CUDA toolkit's corresponding driver version. For example, if you have CUDA 11.4 update 4 installed, you must use NVidia driver 470.82.01 or newer if on Linux. This is a strict requirement that exists because JAX relies on JIT-compiling code; older drivers may lead to failures.
If you need to use an newer CUDA toolkit with an older driver, for example on a cluster where you cannot update the NVidia driver easily, you may be able to use the CUDA forward compatibility packages that NVidia provides for this purpose.
Next, run

pip install --upgrade pip
# Installs the wheel compatible with CUDA 11 and cuDNN 8.2 or newer.
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
These pip installations do not work with Windows, and may fail silently; see above.

The jaxlib version must correspond to the version of the existing CUDA installation you want to use. You can specify a particular CUDA and CuDNN version for jaxlib explicitly:

pip install --upgrade pip

# Installs the wheel compatible with Cuda >= 11.4 and cudnn >= 8.2
pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Installs the wheel compatible with Cuda >= 11.1 and cudnn >= 8.0.5
pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
You can find your CUDA version with the command:

nvcc --version
Some GPU functionality expects the CUDA installation to be at /usr/local/cuda-X.X, where X.X should be replaced with the CUDA version number (e.g. cuda-11.1). If CUDA is installed elsewhere on your system, you can either create a symlink:

sudo ln -s /path/to/cuda /usr/local/cuda-X.X
Please let us know on the issue tracker if you run into any errors or problems with the prebuilt wheels.

pip installation: Google Cloud TPU
JAX also provides pre-built wheels for Google Cloud TPU. To install JAX along with appropriate versions of jaxlib and libtpu, you can run the following in your cloud TPU VM:

pip install --upgrade pip
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip installation: Colab TPU
Colab TPU runtimes come with JAX pre-installed, but before importing JAX you must run the following code to initialize the TPU:

import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
Colab TPU runtimes use an older TPU architecture than Cloud TPU VMs, so installing jax[tpu] should be avoided on Colab. If for any reason you would like to update the jax & jaxlib libraries on a Colab TPU runtime, follow the CPU instructions above (i.e. install jax[cpu]).

Conda installation
There is a community-supported Conda build of jax. To install using conda, simply run

conda install jax -c conda-forge
To install on a machine with an NVidia GPU, run

conda install jax cuda-nvcc -c conda-forge -c nvidia
Note the cudatoolkit distributed by conda-forge is missing ptxas, which JAX requires. You must therefore either install the cuda-nvcc package from the nvidia channel, or install CUDA on your machine separately so that ptxas is in your path. The channel order above is important (conda-forge before nvidia). We are working on simplifying this.

If you would like to override which release of CUDA is used by JAX, or to install the CUDA build on a machine without GPUs, follow the instructions in the Tips & tricks section of the conda-forge website.

See the conda-forge jaxlib and jax repositories for more details.

Building JAX from source
See Building JAX from source.

Neural network libraries
Multiple Google research groups develop and share libraries for training neural networks in JAX. If you want a fully featured library for neural network training with examples and how-to guides, try Flax.

In addition, DeepMind has open-sourced an ecosystem of libraries around JAX including Haiku for neural network modules, Optax for gradient processing and optimization, RLax for RL algorithms, and chex for reliable code and testing. (Watch the NeurIPS 2020 JAX Ecosystem at DeepMind talk here)

Citing JAX
To cite this repository:

@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/google/jax},
  version = {0.3.13},
  year = {2018},
}
In the above bibtex entry, names are in alphabetical order, the version number is intended to be that from jax/version.py, and the year corresponds to the project's open-source release.

A nascent version of JAX, supporting only automatic differentiation and compilation to XLA, was described in a paper that appeared at SysML 2018. We're currently working on covering JAX's ideas and capabilities in a more comprehensive and up-to-date paper.

Reference documentation
For details about the JAX API, see the reference documentation.

For getting started as a JAX developer, see the developer documentation.

About
Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more

jax.readthedocs.io/
Topics
jax
Resources
 Readme
License
 Apache-2.0 license
Code of conduct
 Code of conduct
Citation
Stars
 20.4k stars
Watchers
 286 watching
Forks
 1.9k forks
Releases 45
jaxlib release v0.3.20
Latest
2 days ago
+ 44 releases
Used by 5.2k
@aidanscannell
@mvsoom
@ppnkl123
@araffin
@WallyTutor
@devzhk
@fxrshed
@GCABC123
+ 5,180
Contributors 431
@mattjj
@hawkinsp
@jakevdp
@gnecula
@froystig
@yashk2810
@skye
@apaszke
@sharadmv
@shoyer
+ 420 contributors
Languages
Python
92.6%
 
C++
4.7%
 
Jupyter Notebook
1.3%
 
Starlark
1.1%
 
Shell
0.2%
 
C
0.1%
