# JAX

JAX 可以理解为 **Numpy + autograd + XLA** .它提供了类似 NumPy 的 API，支持自动微分，并能利用 XLA 在 GPU 和 TPU 等硬件加速器上进行高性能计算。

## Composable Transformations

Composable transformations 是 JAX 的核心特性，主要包括：

-   **`grad`**:  计算函数的梯度。
-   **`jit`**:  即时编译（Just-In-Time Compilation），将 Python 函数编译成 XLA 优化的内核，以提高执行速度。
-   **`vmap`**:  自动向量化/批处理（Vectorizing Map）。
-   **`pmap`**:  并行映射（Parallel Map），用于在多个设备上并行执行函数。

这些 transformations 都是**高阶函数**，即它们接受函数作为输入，并返回新的函数。这种设计使得 transformations 可以组合使用，例如 `jit(grad(grad(vmap(func))))`，从而实现复杂的功能。

**函数要求：**

JAX 的 transformations 通常要求被转换的函数是 **pure function**，即：

1. **相同的输入总是产生相同的输出。**
2. **没有副作用**，即函数不会修改其作用域之外的状态。

虽然 JAX 本身不提供自动的纯函数校验，但遵循纯函数的原则对于确保 JAX transformations 的正确性和可预测性至关重要。


### `pmap`
---
`pmap` 用于 single-program multiple-data (SPMD) 程序，支持单机多卡、多机多卡（需配置分布式环境）。

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


### `vmap`
---
`vmap`用于自动向量化，实现 SIMD (Single Instruction, Multiple Data) 的效果。

**用法：**

1. 函数式： `vmap(...)`
2. 装饰器： `@vmap`

**工作流程：**

- **自动批处理：** 接受函数 `func`，自动为其添加一个额外的批处理维度（batch dimension）。

例如，函数 `f` 用于计算单个数字的平方，函数 `vmap(f)` 将接受一个数组作为输入，并返回一个包含每个数字平方的数组。


**参数：**

`in_axes`

- 指定输入参数中要进行批处理的轴。默认值为 `0`，表示沿着每个参数的第一个轴进行。
- 可以是整数、`None` 或一个元组/列表。

## Collective Operations
