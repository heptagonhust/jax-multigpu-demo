import jax
import jax.numpy as jnp
import functools

# 矩阵乘，使用 jax.jit 优化编译
@jax.jit
def matmul(matrix1, matrix2):
  return jnp.matmul(matrix1, matrix2)

# 使用 jax.pmap 进行多设备并行化
@jax.pmap
def parallel_matmul(matrix1, matrix2):
    return matmul(matrix1, matrix2)

# 创建一个批量输入数据
# 假如有两个 gpu
key = jax.random.PRNGKey(0) #  添加 PRNGKey
key1, key2 = jax.random.split(key) # 使用不同的随机数序列

matrix1 = jax.random.normal(key1, (2, 1024, 1024)) 
matrix2 = jax.random.normal(key2, (2, 1024, 1024)) 

# 调用并行化的模型
output = parallel_matmul(matrix1, matrix2)

# 输出结果
print(output)