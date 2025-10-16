import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
from sklearn import model_selection
from sklearn import metrics

voice_data = pd.read_csv('voice_gender.csv')

X = voice_data.drop('label', axis=1)
y = voice_data['label']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

dt_1 = tree.DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=42)
dt_1.fit(X_train, y_train)

plt.figure(figsize=(10, 6))
tree.plot_tree(dt_1, feature_names=X.columns, class_names=['female', 'male'], filled=True)
plt.show()

# Вопросы по заданию 1
# 1. На основе какого фактора будет построено решающее правило в корневой вершине?
# Ответ: meanfreq

# 2. Чему равно оптимальное пороговое значение для данного фактора?
threshold = dt_1.tree_.threshold[0]
print(f"Оптимальное пороговое значение: {threshold:.3f}")

# 3. Сколько процентов наблюдений, для которых выполняется заданное в корневой вершине условие, содержится в обучающей выборке?
y_pred_train_1 = dt_1.predict(X_train)
meanfreq_values = X_train['meanfreq']
threshold = dt_1.tree_.threshold[0]
if dt_1.tree_.children_left[0] != -1:
    indices = np.where(meanfreq_values <= threshold)[0]
else:
    indices = np.array([])
percentage = len(indices) / len(X_train) * 100
print(f"Процент наблюдений: {percentage:.1f}")

# 4. Сделайте предсказание и рассчитайте значение метрики accuracy на тестовой выборке.
y_pred_1 = dt_1.predict(X_test)
accuracy_1 = metrics.accuracy_score(y_test, y_pred_1)
print(f"Accuracy на тестовой выборке: {accuracy_1:.3f}")

# Задание 2. Увеличим глубину дерева

dt_2 = tree.DecisionTreeClassifier(max_depth=2, criterion='entropy', random_state=42)
dt_2.fit(X_train, y_train)

plt.figure(figsize=(10, 6))
tree.plot_tree(dt_2, feature_names=X.columns, class_names=['female', 'male'], filled=True)
plt.show()

# Вопросы по заданию 2
# 1. Из приведённых ниже факторов выберите те, что используются при построении данного дерева решений:
# Ответ: A, B, D

# 2. Сколько листьев в построенном дереве содержат в качестве предсказания класс female?
#  Нужно проанализировать визуализацию дерева.
# Ответ: 2

# 3. Сделайте предсказание и рассчитайте значение метрики accuracy на тестовой выборке.
y_pred_2 = dt_2.predict(X_test)
accuracy_2 = metrics.accuracy_score(y_test, y_pred_2)
print(f"Accuracy на тестовой выборке: {accuracy_2:.3f}")

# Задание 3. Дадим дереву решений б’ольшую свободу

dt_inf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
dt_inf.fit(X_train, y_train)

# 1. Чему равна глубина полученного дерева решения?
depth_inf = dt_inf.get_depth()
print(f"Глубина дерева: {depth_inf}")

# 2. Чему равно количество листьев в полученном дереве решений?
n_leaves_inf = dt_inf.get_n_leaves()
print(f"Количество листьев: {n_leaves_inf}")

# 3. Сделайте предсказание для обучающей и тестовой выборок и рассчитайте значение метрики accuracy на каждой из выборок
y_pred_train_inf = dt_inf.predict(X_train)
accuracy_train_inf = metrics.accuracy_score(y_train, y_pred_train_inf)
print(f"Accuracy на обучающей выборке: {accuracy_train_inf:.3f}")

y_pred_test_inf = dt_inf.predict(X_test)
accuracy_test_inf = metrics.accuracy_score(y_test, y_pred_test_inf)
print(f"Accuracy на тестовой выборке: {accuracy_test_inf:.3f}")

# Задание 4. Попробуем найти оптимальные внешние параметры модели дерева решений

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [3, 4, 5, 10]
}

cv = model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = model_selection.GridSearchCV(tree.DecisionTreeClassifier(random_state=0), param_grid, scoring='accuracy', cv=cv, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 1. Какой критерий информативности использует наилучшая модель?
best_criterion = grid_search.best_params_['criterion']
print(f"Лучший критерий информативности: {best_criterion}")

# 2. Чему равна оптимальная найденная автоматически (с помощью GridSearchCV) максимальная глубина?
best_max_depth = grid_search.best_params_['max_depth']
print(f"Оптимальная максимальная глубина: {best_max_depth}")

# 3. Чему равно оптимальное минимальное количество объектов, необходимое для разбиения?
best_min_samples_split = grid_search.best_params_['min_samples_split']
print(f"Оптимальное минимальное количество объектов для разбиения: {best_min_samples_split}")

# 4. С помощью наилучшей модели сделайте предсказание отдельно для обучающей и тестовой выборок. Рассчитайте значение метрики accuracy на каждой из выборок.
best_model = grid_search.best_estimator_
y_pred_train_best = best_model.predict(X_train)
accuracy_train_best = metrics.accuracy_score(y_train, y_pred_train_best)
print(f"Accuracy на обучающей выборке для лучшей модели: {accuracy_train_best:.3f}")

y_pred_test_best = best_model.predict(X_test)
accuracy_test_best = metrics.accuracy_score(y_test, y_pred_test_best)
print(f"Accuracy на тестовой выборке для лучшей модели: {accuracy_test_best:.3f}")

# Задание 5. Для оптимального дерева решений, построенного в задании 4, найдите важность каждого из факторов

feature_importances = best_model.feature_importances_

plt.figure(figsize=(12, 6))
sns.barplot(x=feature_importances, y=X.columns)
plt.title('Важность признаков')
plt.xlabel('Важность')
plt.ylabel('Признак')
plt.show()

# Выделение топ-3 наиболее важных факторов
importances_sorted = sorted(zip(feature_importances, X.columns), reverse=True)
top_3_features = [feature for importance, feature in importances_sorted[:3]]
print(f"Топ-3 наиболее важных факторов: {top_3_features}")