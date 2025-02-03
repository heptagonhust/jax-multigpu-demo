import jax
import jax.numpy as jnp
import time

def serial_matmul(matrix1, matrix2):
  return jnp.dot(matrix1, matrix2)


if __name__ == "__main__":
  key = jax.random.PRNGKey(0) # 创建一个随机数种子，确保每次运行生成相同的随机数，方便复现
  matrix_size = 2048 # 定义矩阵大小

  # 生成随机矩阵
  matrix1 = jax.random.normal(key, (matrix_size, matrix_size))
  matrix2 = jax.random.normal(key, (matrix_size, matrix_size))

  # 性能测试
  start_time_serial = time.time()
  result_serial = serial_matmul(matrix1, matrix2)
  end_time_serial = time.time()
  serial_time = end_time_serial - start_time_serial

  print("串行矩阵乘法 (单设备):")
  print(f"  矩阵大小: {matrix_size}x{matrix_size}")
  print(f"  设备: {jax.devices()[0]}") 
  print(f"  计算时间: {serial_time:.4f} 秒")
