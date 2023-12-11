# folx - forward laplacian for JAX

This submodule implements the forward laplacian from https://arxiv.org/abs/2307.08214. It is implemented as a [custom interpreter for Jaxprs](https://jax.readthedocs.io/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html).


## Example
For simple usage, one can decorate any function with `forward_laplacian`.
```python
import numpy as np
from folx import forward_laplacian

def f(x):
    return (x**2).sum()

fwd_f = forward_laplacian(f)
result = fwd_f(np.arange(3, dtype=float))
result.x # f(x) 3
result.jacobian.dense_array # J_f(x) [0, 2, 4]
result.laplacian # tr(H_f(x)) 6
```

## Introduction
To avoid custom wrappers for all of JAX's commands, the forward laplacian is implemented as custom interpreter for Jaxpr. 
This means if you have a function
```python
class Fn(Protocol):
    def __call__(self, *args: PyTree[Array]) -> PyTree[Array]:
        ...
```
the resulting function will have the signature:
```python
class LaplacianFn(Protocol):
    def __call__(self, *args: PyTree[Array]) -> PyTree[FwdLaplArray]:
        ...
```
where `FwdLaplArray` is a triplet of 
```python
FwdLaplArray.x # jax.Array f(x) f(x).shape
FwdLaplArray.jacobian # FwdJacobian J_f(x)
FwdLaplArray.laplacian # jax.Array tr(H_f(x)) f(x).shape
```
The jacobian is implemented by a custom class as the forward laplacian supports automatic sparsity. To get the full jacobian:
```python
FwdLaplArray.jacobian.dense_array # jax.Array (*f(x).shape, x.size)
```

## Implementation idea
The idea is to rely on the original function and autodifferentiation to propagate `FwdLaplArray` forward instead of the regular `jax.Array`. The rules for updating `FwdLaplArray` are described by the pseudocode:
```python
x # FwdLaplArray
y = FwdLaplArray(
    x=f(x.x),
    jacobian=jvp(f, (x.x,), (x.jacobian)),
    laplacian=tr_vhv(f, x.jacobian) + jvp(f, (x.x,), (x.laplacian,))
)
# tr_vhv is tr(J_f H_f J_f^T)
```

## Implementation

When you call the function returned by `forward_laplacian(fn)`, we first use `jax.make_jaxpr` to obtain the jaxpr for `fn`.
But instead of using the [standard evaluation pipeline](https://github.com/google/jax/blob/776baba0a3fca15a909cb7d108eea830cbe3fc1d/jax/_src/core.py#L436), we use a custom interpreter that replaces all operations to propate `FwdLaplArray` forward instead of regular `jax.Array`.

### Package structure
The general structure of the package is
1. `interpreter.py` contains the evaluation of jaxpr and exported function decorator. 
2. `fwd_laplacian.py` contains subfunction decorator that maps a function that takes `jax.Array`s to a function that accepts `FwdLaplArray`s instead.
3. `jvp.py` contains logic for jacobian vector products.
4. `hessian.py` contains logic for tr(JHJ^T).
5. `api.py` contains general interfaces shared in the package.
6. `utils.py` contains several small utility functions.
7. `tree_utils.py` contains several utility functions for PyTrees. 


### Function Annotations
There is a default interpreter that will simply apply the rules outlined above but if additional information about a function is available, e.g., that it applies elementwise like `jnp.tanh`, we can do better.
These additional annotations are available in `interpreters.py`'s `_LAPLACE_FN_REGISTRY`. 
Specifically, to augment a function `fn` to accept `FwdLaplArray` instead of regular `jax.Array`, we wrap it with `add_forward_laplacian` from `fwd_laplacian.py`:
```python
add_forward_laplacian(jnp.tanh, in_axes=())
```
In this case, we annotate the function to be applied elementwise, i.e., `()` indicates that none of the axes are relevant for the function.

If we know nothing about which axes might be essential, one must pass `None` (the default value) to mark all axes as imporatnt, e.g.,
```python
add_forward_laplacian(jnp.sum, in_axes=None, flags=FunctionFlags.LINEAR)
```
However, in this case we know that a summation is a linear operation. This information is useful for fast hessian computations.


### Sparsity
Sparsity is detected at compile time, this has the advantage of avoiding expensive index computations at runtime and enables efficient reductions. However, it completely prohibits dynamic indexing, i.e., if indices are data-dependent we will simply default to full jacobians.

As we know a lot about the sparsity structure apriori, e.g., that we are only sparse in one dimension, we use a custom sparsity operations that are more efficient than relying on JAX's default `BCOO` (further, at the time of writing, the support for `jax.experimental.sparse` is quite bad).
So, the sparsity data format is implemented in `FwdJacobian` in `api.py`. Instead of storing a dense array `(m, n)` for a function `f:R^n -> R^m`, we store only the non-zero data in a `(m,d)` array where `d<n` is the maximum number of non-zero inputs any output depends on. 
To be able to recreate the larger `(m,n)` array from the `(m,d)` array, we additional keep track of the indices in the last dimension in a mask `(m,d)` dimensional array of integers `0<mask_ij<n`.

Masks are treated as compile time static and will be traced automatically. If the tracing is not possible, e.g., due to data dependent indexing, we will fall back to a dense implementation. These propagation rules are implemented in `jvp.py`.

##### Omnistaging
If arrays do not depend on the initial input, they are typically still traced to better optimize the final program. This is called [omnistaging](https://github.com/google/jax/pull/3370). While this generally is beneficial, it does not allow us to perform indexing as tracer hide the actual data. 
So, if we use sparsity we want to compute all arrays that do not explicitly depend on the input such that we could use them for index operations.
While this is not documented, it can be accomplished by overwriting the global trace via:
```python
from jax import core

with core.new_main(core.EvalTrace, dynamic=True):
    ...
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
