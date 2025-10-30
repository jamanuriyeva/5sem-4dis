import numpy as np
from scipy.optimize import linprog

# Стоимости по маршрутам (в порядке x11,x12,x13,x21,x22,x23)
c = [8,6,10,9,7,5]

# Ограничения-равенства A_eq @ x = b_eq
A_eq = [
    [1,1,1,0,0,0],  # склад 1
    [0,0,0,1,1,1],  # склад 2
    [1,0,0,1,0,0],  # база Альфа
    [0,1,0,0,1,0],  # база Бета
    [0,0,1,0,0,1]   # база Гамма
]
b_eq = [150,250,120,180,100]

bounds = [(0,None)]*6

res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
print('Статус:', res.message)
x = res.x
flows = x.reshape(2,3)  # reshape: [ [x11,x12,x13], [x21,x22,x23] ]
print('Оптимальные потоки (т):')
print(flows)
print('Минимальная стоимость:', res.fun)