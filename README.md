# JAX
JAX是一个具备自动微分功能、且能够在CPU/GPU/TPU上高性能执行的NumPy（可以理解为Numpy+autograd+XLA）  

## Composable transformations
Composable transformations 是 JAX 的核心，主要包括grad、jit、vmap、pmap.
>Transformations 译为高阶函数，即“输入/输出是函数“的函数。  
>Composable，即可实现类似 `jit(grad(grad(vmap(some_func))))`的任意组合。

>这里函数要求 **pure function**，不过JAX本身不提供纯函数校验
### pmap
pmap 用于 single-program multiple-data (SPMD) 程序，支持单机多卡。  

函数签名：
```python
jax.pmap(func, axis_name=None, *, in_axes=0, out_axes=0, static_broadcasted_argnums=(), devices=None, backend=None, axis_size=None, donate_argnums=(), global_arg_shapes=None)
``` 
pmap 接收一个函数 func，并将其复制到各 XLA 设备上。 随后，pmap 在这些设备上并行地执行 func 的副本 (这些副本会被 XLA 编译)，每个副本处理对应的设备上的数据分片。

其中 axis_name 指定用于集体通信的轴名称，in_axes 指定输入参数的分片方式 (默认为沿第0轴)，out_axes 指定输出结果的拼接方式 (默认为沿第0轴)  

#### in_axes
- 若指定`in_axes=None`，输入参数不分片，自动在多设备广播。  
- 若指定`in_axes=<n>`，输入参数将按第 `<n>` 轴的维度 d 分成 d 份，并自动分发给 d 个设备。  


注意指定 axis 的维度不能大于 GPU 设备数量，否则获得错误。例如：
```python
import jax
import jax.numpy as jnp

def simple_func(x):
  return x + 1

input_data = jnp.arange(8)
result = jax.pmap(simple_func)(input_data)
print(result)
```
报错如下：
```
Traceback (most recent call last):
  File "test.py", line 9, in <module>
    result = pmap_simple(input_data)
             ^^^^^^^^^^^^^^^^^^^^^^^
ValueError: compiling computation that requires 8 logical devices, but only 2 XLA devices are available (num_replicas=8)
--------------------
```
可见，两卡确实只能处理指定axis维度为2的参数。改成`input_data = jnp.arange(8).reshape(2,4)`，或`input_data = jnp.arange(8).reshape(4,2)`并指定`in_axes=1`就过了。  

对程序`pmap_1.py`，仅当`matrix_size`=`jax.local_device_count()`，才能正常运行。

#### axes_name