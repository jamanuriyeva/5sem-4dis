import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
import warnings
warnings.filterwarnings('ignore')

# Загрузка и подготовка данных
df = pd.read_csv('tovar_moving.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 1. Разделение на train/test
train = df.iloc[:-1].copy()
test = df.iloc[-1:].copy()
actual_value = test['qty'].values[0]  # Исправлено на 'qty'
print("=== РАЗДЕЛЕНИЕ ДАННЫХ ===")
print(f"Train: {len(train)} записей, Test: {len(test)} записей")
print(f"Последнее значение (test): {actual_value}")

# 2. Анализ временного ряда
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.plot(train['date'], train['qty'])  # Исправлено на 'qty'
plt.title('Исходный временной ряд')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
rolling_mean = train['qty'].rolling(window=30).mean()  # Исправлено на 'qty'
plt.plot(train['date'], train['qty'], alpha=0.5, label='Исходный')  # Исправлено на 'qty'
plt.plot(train['date'], rolling_mean, color='red', label='Скользящее среднее')
plt.title('Тренд')
plt.legend()
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
plot_acf(train['qty'], lags=40, ax=plt.gca())  # Исправлено на 'qty'
plt.title('Автокорреляция')

plt.subplot(2, 2, 4)
plot_pacf(train['qty'], lags=40, ax=plt.gca())  # Исправлено на 'qty'
plt.title('Частичная автокорреляция')
plt.tight_layout()
plt.show()

# 3. Экспоненциальное сглаживание
model_ses = SimpleExpSmoothing(train['qty']).fit(smoothing_level=0.7, optimized=False)  # Исправлено на 'qty'
ses_prediction = model_ses.forecast(1).values[0]
print("\n=== ЭКСПОНЕНЦИАЛЬНОЕ СГЛАЖИВАНИЕ ===")
print(f"Прогноз: {ses_prediction:.2f}, Факт: {actual_value:.2f}")
print(f"Ошибка: {abs(ses_prediction - actual_value):.2f}")

# 4. Проверка на стационарность
def check_stationarity(series):
    result = adfuller(series.dropna())
    return result[1] <= 0.05

print("\n=== СТАЦИОНАРНОСТЬ ===")
order_integration = 0
current_series = train['qty'].copy()  # Исправлено на 'qty'
while not check_stationarity(current_series) and order_integration < 3:
    order_integration += 1
    current_series = current_series.diff().dropna()
print(f"Порядок интегрирования: {order_integration}")

# 5. Определение порядка AR
plt.figure(figsize=(12, 5))
plot_pacf(train['qty'], lags=40, alpha=0.05)  # Исправлено на 'qty'
plt.title('Частичная автокорреляция (определение порядка AR)')
plt.show()

best_aic = np.inf
best_order = 1
for p in range(1, 21):
    try:
        model = AutoReg(train['qty'], lags=p)  # Исправлено на 'qty'
        fitted = model.fit()
        if fitted.aic < best_aic:
            best_aic = fitted.aic
            best_order = p
    except:
        continue
print(f"Рекомендуемый порядок AR: {best_order}")

# 6. Построение AR модели
model_ar = AutoReg(train['qty'], lags=best_order)  # Исправлено на 'qty'
fitted_ar = model_ar.fit()
ar_prediction = fitted_ar.forecast(1).values[0]
print("\n=== AR МОДЕЛЬ ===")
print(f"Прогноз: {ar_prediction:.2f}, Факт: {actual_value:.2f}")
print(f"Ошибка: {abs(ar_prediction - actual_value):.2f}")

# 7. Сравнение результатов
comparison = pd.DataFrame({
    'Метод': ['Эксп. сглаживание', 'AR модель'],
    'Прогноз': [ses_prediction, ar_prediction],
    'Факт': [actual_value, actual_value],
    'Ошибка': [abs(ses_prediction - actual_value), abs(ar_prediction - actual_value)]
})
print("\n=== СРАВНЕНИЕ РЕЗУЛЬТАТОВ ===")
print(comparison)

best_method = comparison.loc[comparison['Ошибка'].idxmin(), 'Метод']
print(f"\nЛучший метод: {best_method}")

# Дополнительная визуализация
plt.figure(figsize=(12, 6))
methods = ['Эксп. сглаживание', 'AR модель']
predictions = [ses_prediction, ar_prediction]
plt.bar(methods, predictions, alpha=0.7, label='Прогноз')
plt.axhline(y=actual_value, color='red', linestyle='--', label='Фактическое значение')
plt.ylabel('Количество заказов')
plt.title('Сравнение прогнозов')
plt.legend()
for i, v in enumerate(predictions):
    plt.text(i, v + 0.1, f'{v:.1f}', ha='center', va='bottom')
plt.text(0.5, actual_value + 0.1, f'Факт: {actual_value:.1f}', ha='center', va='bottom', color='red')
plt.show()