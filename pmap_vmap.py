import jax
import jax.numpy as jnp
from jax import pmap, vmap

# 假设我们有一个函数，它对一个向量执行某些操作
def process_vector(x):
  return jnp.sin(x) + jnp.cos(x)

# 我们想对一批向量进行操作，并使用 vmap 矢量化处理每个向量
@vmap
def vectorized_process(x):
    return process_vector(x)

# 假设我们有多个设备，并希望在这些设备上并行处理多批向量
@pmap
def parallel_vectorized_process(data_batches):
  return vectorized_process(data_batches)

# 模拟数据：2个设备，每个设备处理4批数据，每批数据包含3个向量，每个向量长度为5
num_devices = 2
batches_per_device = 4
vectors_per_batch = 3
vector_length = 5
data = jnp.arange(num_devices * batches_per_device * vectors_per_batch * vector_length).reshape(
    num_devices, batches_per_device, vectors_per_batch, vector_length
)

# 执行并行和矢量化计算
results = parallel_vectorized_process(data)

# 结果将是一个形状为 (2, 4, 3, 5) 的数组，与输入数据的形状相同
print(results.shape)
