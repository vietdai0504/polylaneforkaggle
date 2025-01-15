import numpy as np
import matplotlib.pyplot as plt

# Định nghĩa hàm
def f(x):
    return -2/27 * x**3 + 2 * x + 4

# Tạo dữ liệu cho x >= 0
x = np.linspace(0, 10, 500)  # Chỉ lấy giá trị x >= 0
y = f(x)

# Cắt bỏ phần y < 0
x = x[y >= 0]  # Giữ lại x khi y >= 0
y = y[y >= 0]  # Giữ lại y khi y >= 0

# Vẽ đồ thị
plt.figure(figsize=(8, 6))
plt.plot(x, y, label="f(x) (x >= 0, y >= 0)", color="red")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("Cubic Polynomial with x >= 0 and y >= 0")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid()
plt.show()
