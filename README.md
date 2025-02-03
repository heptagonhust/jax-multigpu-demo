# JAX
JAX 可以理解为 Numpy+autograd+XLA

## Composable transformations
Composable transformations 是 JAX 的核心，主要包括grad、jit、vmap、pmap.
>Transformations，高阶函数，即“输入/输出是函数“的函数。  
>Composable，e.g. `jit(grad(grad(vmap(func))))`

函数要求 **pure function**，不过JAX本身不提供纯函数校验
### pmap
> 可以使用装饰器语法`@jax.pmap`，参考`pmap.py`。这与函数形式等价，但更简洁、更 Pythonic.

pmap 用于 single-program multiple-data (SPMD) 程序，支持单机多卡。  

函数签名：
```python
jax.pmap(func, axis_name=None, *, in_axes=0, out_axes=0, static_broadcasted_argnums=(), devices=None, backend=None, axis_size=None, donate_argnums=(), global_arg_shapes=None)(argus)
``` 
`pmap` 接受若干参数 `argu`，并将其按 `in_axes` 指定的轴（默认`in_axes=0`，**需满足该轴的维度$\leq$设备数**）自动分片到各 XLA 设备；接收一个函数 `func`，并将其复制到各 XLA 设备。

随后，pmap 在这些设备上并行地执行 func 的副本，每个副本处理对应的设备上的数据分片。

#### in_axes
- 若指定`in_axes=None`，该参数不分片，自动在多设备广播。  
- 若指定`in_axes=<n>`，该参数将按第 `<n>` 轴切分。

*注意指定轴的维度不能大于 GPU 设备数量，否则获得错误。
### vmap
## Collective operations