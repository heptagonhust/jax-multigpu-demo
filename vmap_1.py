import jax
import jax.numpy as jnp

# 一个处理单个向量的函数
def square_elements(x):
  return x**2

# 使用 vmap 将其转换为处理一批向量的函数
batched_square = vmap(square_elements)

# 示例数据
vectors = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 使用 batched_square 计算每个向量的平方
result = batched_square(vectors)

print(result)
# 输出：
# [[ 1  4  9]
#  [16 25 36]
#  [49 64 81]]