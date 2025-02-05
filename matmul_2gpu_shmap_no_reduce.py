import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as PartSpec
from jax.experimental.shard_map import shard_map
import time
import functools

mesh = jax.make_mesh((2, 1), ('x', 'reduce'))
print ("mesh: ", type(mesh))

@functools.partial(shard_map, mesh=mesh, in_specs=(PartSpec('x', 'reduce'), PartSpec('reduce', None)), out_specs=PartSpec('x', None))
def shard_map_matmul(matrix1, matrix2):
    return jax.lax.psum(jnp.dot(matrix1, matrix2), 'reduce')


# 创建一个批量输入数据
# 假如有两个 gpu
key = jax.random.PRNGKey(0) #  添加 PRNGKey
key1, key2 = jax.random.split(key) # 使用不同的随机数序列

matrix1 = jax.random.normal(key1, (8192, 8191), dtype=jnp.float32).block_until_ready()
matrix2 = jax.random.normal(key2, (8191, 8190), dtype=jnp.float32).block_until_ready()
print ("matrix1: ", matrix1.shape)
print ("matrix2: ", matrix2.shape)

tick0 = time.time()
for __ in range(1000):
  output_d = shard_map_matmul(matrix1, matrix2)
output_d.block_until_ready()
tick1 = time.time()
shard_map_time = (tick1 - tick0)
print("shard_map GPU Time: ", shard_map_time, 'ms')
print("output_d: ", output_d.shape)
print("output_d: ", output_d)

tick0 = time.time()
for __ in range(1000):
  output_ref = jnp.dot(matrix1, matrix2)
output_ref.block_until_ready()
tick1 = time.time()
NamedSharding_time = (tick1 - tick0)
print("NamedSharding GPU Time: ", NamedSharding_time, 'ms')
print("output_ref: ", output_ref.shape)
print("output_ref: ", output_ref)
print("Parallel efficiency: ", NamedSharding_time / shard_map_time * 100 , '%')


# def close(a, b):
  # return jnp.allclose(a, b, atol=1e-2)

# breal_outer_loops = False

# for i in range(output_ref.shape[0]):
#   for j in range(output_ref.shape[1]):
#     if not close(output_ref[i][j], output_d[i][j]) :
#       print("i: ", i)
#       print("j: ", j)
#       print("ref: ", output_ref[i][j])
#       print("out: ", output_d[i][j])
#       breal_outer_loops = True
#       break
#     if breal_outer_loops:
#       break
if jnp.allclose(output_d, output_ref, atol=1e-2):
  print ("All close under 1e-2.")
else:
  print ("Not all close under 1e-2.")
# assert jnp.allclose(output_d, output_ref, atol=1e-2 * output_ref.mean())
