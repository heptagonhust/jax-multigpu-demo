import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding

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

matrix1 = jax.random.normal(key1, (1024, 1024)) 
matrix2 = jax.random.normal(key2, (1024, 1024)) 

mesh = jax.make_mesh((2, 1), ('x', 'y'))
matrix1_sharding = NamedSharding(mesh, P('x', None))
matrix2_sharding = NamedSharding(mesh, P())

matrix1_d = jax.device_put(matrix1, matrix1_sharding)
matrix2_d = jax.device_put(matrix2, matrix2_sharding)

# 调用并行化的模型
output_d = parallel_matmul(matrix1_d, matrix2_d)

output_h = parallel_matmul(matrix1, matrix2)

assert jnp.allclose(output_d, output_h, atol=1e-3)
