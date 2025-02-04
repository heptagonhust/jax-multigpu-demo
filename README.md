注：jax 的 Composable transmations 都要求被变换的函数是纯函数

## `vmap`
`vmap`用于自动向量化，实现 SIMD (Single Instruction, Multiple Data) 的效果。

**工作流程：**

- **自动批处理：** 接受函数 `func`，自动为其添加一个额外的批处理维度（batch dimension）。

例如，函数 `f` 用于计算单个数字的平方，函数 `vmap(f)` 将接受一个数组作为输入，并返回一个包含每个数字平方的数组。

## `pmap`
`pmap` 用于 single-program multiple-data (SPMD) 程序，支持单机多卡

**用法：**

1. 装饰器 `@jax.pmap`
2. 函数式 `jax.pmap(...)` 

**签名：**

```python
jax.pmap(func, axis_name=None, *, in_axes=0, out_axes=0, static_broadcasted_argnums=(), devices=None, backend=None, axis_size=None, donate_argnums=(), global_arg_shapes=None)(argus)
```

**工作流程：**

1. **数据分片：** `pmap` 接受若干参数 `argus`，并将其按 `in_axes` 指定的轴（默认 `in_axes=0`）自动分片到各 XLA 设备。**注意：被指定为分片轴的维度必须小于或等于设备数量。**
2. **函数复制：** `pmap` 接收一个函数 `func`，并将其复制到每个 XLA 设备上。
3. **并行执行：** `pmap` 在这些设备上并行地执行 `func` 的副本，每个副本处理对应设备上的数据分片。

**参数：**

`in_axes`

-   若指定 `in_axes=None`，则对应的参数不会被分片，而是在所有设备上进行广播。
-   若指定 `in_axes=<n>`，则对应的参数将沿着第 `<n>` 个轴进行切分。
-   支持以 list 或者 tuple 的形式指定多个轴，如 `in_axes=[0, None, 1]`。

**注：** 指定的 `in_axes` 轴的维度必须小于或等于设备数量，否则会报错。

## `jax.lax`
`jax.lax.p*` 是一系列用于多设备并行操作的原语（primitives），包括`pmap`,`pmean`,`psum`等.