import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
import time

# 使用 jax.jit 优化编译
# @jax.jit
def matmul(matrix1, matrix2):
  return jnp.matmul(matrix1, matrix2)

# 使用 jax.pmap 进行多设备并行化
@jax.jit
def parallel_matmul(matrix1, matrix2):
    return matmul(matrix1, matrix2)

# 创建一个批量输入数据
# 假如有两个 gpu
key = jax.random.PRNGKey(0) #  添加 PRNGKey
key1, key2 = jax.random.split(key) # 使用不同的随机数序列

matrix1 = jax.random.normal(key1, (8192, 8191), dtype=jnp.float32).block_until_ready()
matrix2 = jax.random.normal(key2, (8191, 8190), dtype=jnp.float32).block_until_ready()
print ("matrix1: ", matrix1.shape)
print ("matrix2: ", matrix2.shape)

mesh = jax.make_mesh((2, 1), ('x', 'y'))
matrix1_sharding = NamedSharding(mesh, P('x', None))
matrix2_sharding = NamedSharding(mesh, P())

matrix1_d = jax.device_put(matrix1, matrix1_sharding).block_until_ready()
matrix2_d = jax.device_put(matrix2, matrix2_sharding).block_until_ready()

tick0 = time.time()
for __ in range(1000):
  output_d = parallel_matmul(matrix1_d, matrix2_d)
output_d.block_until_ready()
tick1 = time.time()
multi_time = (tick1 - tick0)
print("Multi GPU Time: ", multi_time, 'ms')

tick0 = time.time()
for __ in range(1000):
  output_h = parallel_matmul(matrix1, matrix2)
output_h.block_until_ready()
tick1 = time.time()
single_time = (tick1 - tick0)
print("Single GPU Time: ", single_time, 'ms')
print("Parallel efficiency: ", single_time / multi_time / 2 * 100 , '%')
assert jnp.allclose(output_d, output_h, atol=1e-3)
