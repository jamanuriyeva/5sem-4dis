# Задача 1: оптимизация производства (полный код)
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt

# Параметры
profit = np.array([8000, 12000])  # прибыль с x1 и x2
c = -profit

# Ограничения A_ub @ x <= b_ub
A_ub = np.array([[2,3],[4,6],[1,2]], dtype=float)
b_ub = np.array([240,480,150], dtype=float)
bounds = [(0, None), (0, None)]

# Решение
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
print('Статус:', res.message)
x1_opt, x2_opt = res.x
print(f'Оптимум: x1={x1_opt:.4f}, x2={x2_opt:.4f}')
print('Максимальная прибыль:', -res.fun)

# Визуализация допустимой области и уровня прибыли (опционально)
x1 = np.linspace(0,150,400)
x2_1 = (240 - 2*x1)/3
x2_2 = (480 - 4*x1)/6
x2_3 = (150 - x1)/2
plt.figure(figsize=(8,6))
plt.plot(x1, np.where(x2_1>=0, x2_1, np.nan), label='2x1+3x2<=240')
plt.plot(x1, np.where(x2_2>=0, x2_2, np.nan), label='4x1+6x2<=480')
plt.plot(x1, np.where(x2_3>=0, x2_3, np.nan), label='x1+2x2<=150')
plt.plot(x1_opt, x2_opt, 'o', label='Оптимум')
plt.xlabel('x1 (смартфоны)'); plt.ylabel('x2 (планшеты)'); plt.legend(); plt.grid(True)
plt.show()