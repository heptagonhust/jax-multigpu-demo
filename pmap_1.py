import jax
import jax.numpy as jnp
import time

def serial_matmul(matrix1, matrix2):
  return jnp.dot(matrix1, matrix2)

def parallel_matmul(matrix1, matrix2_replicated):
  """
  参数:
    matrix1:             矩阵1 (jax.numpy.ndarray)，将被 pmap 自动分片。
    matrix2_replicated:  矩阵2 (jax.numpy.ndarray)，已在所有设备上复制一份。
  """
  
  # 自动将 matrix1 沿着第一个维度 (行) 分片，在多个 XLA 设备上并行执行lambda函数
  device_result_shard = jax.pmap(lambda m1_shard: jnp.dot(m1_shard, matrix2_replicated))(matrix1)
  
  # 拼接各设备的运算结果(ShardedDeviceArray)
  return jnp.concatenate(device_result_shard, axis=0)


if __name__ == "__main__":
  key = jax.random.PRNGKey(0)
  matrix_size = 1

  # 生成随机矩阵
  matrix1 = jax.random.normal(key, (matrix_size, matrix_size))
  matrix2 = jax.random.normal(key, (matrix_size, matrix_size))

  # 获取可用的 GPU 数量
  num_devices = len(jax.devices('gpu'))

  # 将 matrix2 复制到所有 GPU 设备 (在 pmap 外部广播)
  device_matrix2_replicated = jax.device_put_replicated(matrix2, jax.devices('gpu'))

  # 性能测试 - 并行版本
  start_time_parallel = time.time()
  result_parallel = parallel_matmul(matrix1, device_matrix2_replicated) # 直接传入 matrix1，pmap 会自动分片
  jax.block_until_ready(result_parallel) # 确保计算完成再计时
  end_time_parallel = time.time()
  parallel_time = end_time_parallel - start_time_parallel

  print("\n多卡并行矩阵乘法:")
  print(f"  矩阵大小: {matrix_size}x{matrix_size}")
  print(f"  设备数量: {num_devices} GPUs")
  print(f"  设备列表: {jax.devices('gpu')}")
  print(f"  计算时间: {parallel_time:.4f} 秒")

  # 性能测试 - 串行版本 (编译后)
  start_time_serial_compiled = time.time()
  result_serial_compiled = serial_matmul(matrix1, matrix2)
  jax.block_until_ready(result_serial_compiled) # 确保计算完成再计时
  end_time_serial_compiled = time.time()
  serial_time_compiled = end_time_serial_compiled - start_time_serial_compiled

  # 验证结果
  if jnp.allclose(result_serial_compiled, result_parallel, rtol=1e-5, atol=1e-5):
      print("\n结果验证: Pass")
  else:
      print("\n结果验证: Failed")


  # 性能比较
  speedup = serial_time_compiled / parallel_time
  print(f"Speed Up: {speedup:.2f} ")
 