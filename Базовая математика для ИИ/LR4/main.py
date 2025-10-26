import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка и подготовка данных
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


class MLP:
    def __init__(self, layers, learning_rate=0.01, momentum=0.9, l2_reg=0.01):
        self.layers = layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_reg = l2_reg
        self.weights = []
        self.biases = []
        self.velocity_w = []  # скорость для весов
        self.velocity_b = []  # скорость для смещений

        # Инициализация весов и скоростей
        for i in range(1, len(layers)):
            # Инициализация Xavier/Glorot
            scale = np.sqrt(2.0 / (layers[i - 1] + layers[i]))
            self.weights.append(np.random.randn(layers[i - 1], layers[i]) * scale)
            self.biases.append(np.zeros((1, layers[i])))
            self.velocity_w.append(np.zeros_like(self.weights[-1]))
            self.velocity_b.append(np.zeros_like(self.biases[-1]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # защита от переполнения

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        activations = [X]
        zs = []

        for i in range(len(self.layers) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            if i < len(self.layers) - 2:  # скрытые слои
                a = self.relu(z)
            else:  # выходной слой
                a = self.sigmoid(z)
            activations.append(a)

        return activations, zs

    def backward(self, activations, zs, y):
        m = y.shape[0]
        deltas = []

        # Ошибка выходного слоя
        output = activations[-1]
        error = output - y.reshape(-1, 1)
        delta = error * self.sigmoid_derivative(output)
        deltas.append(delta)

        # Обратное распространение через скрытые слои
        for i in range(len(self.layers) - 2, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * self.relu_derivative(activations[i])
            deltas.append(delta)

        deltas.reverse()

        # Обновление весов и смещений с моментом
        for i in range(len(self.layers) - 1):
            # Градиенты
            dw = np.dot(activations[i].T, deltas[i]) / m + self.l2_reg * self.weights[i]
            db = np.sum(deltas[i], axis=0, keepdims=True) / m

            # Обновление скорости
            self.velocity_w[i] = self.momentum * self.velocity_w[i] - self.learning_rate * dw
            self.velocity_b[i] = self.momentum * self.velocity_b[i] - self.learning_rate * db

            # Обновление весов и смещений
            self.weights[i] += self.velocity_w[i]
            self.biases[i] += self.velocity_b[i]

    def compute_loss(self, y_true, y_pred):
        """Вычисление бинарной кросс-энтропии с L2 регуляризацией"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        # Бинарная кросс-энтропия
        bce = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        # L2 регуляризация
        l2_loss = 0
        for w in self.weights:
            l2_loss += np.sum(w ** 2)
        l2_loss *= self.l2_reg / (2 * len(y_true))

        return bce + l2_loss

    def predict(self, X, threshold=0.5):
        """Предсказание классов"""
        activations, _ = self.forward(X)
        probabilities = activations[-1]
        return (probabilities > threshold).astype(int).flatten()

    def predict_proba(self, X):
        """Предсказание вероятностей"""
        activations, _ = self.forward(X)
        return activations[-1].flatten()


# Обучение модели
mlp = MLP(layers=[30, 64, 32, 1], learning_rate=0.01, momentum=0.9, l2_reg=0.001)

print("Начало обучения...")
epochs = 1000
train_losses = []
val_losses = []

for epoch in range(epochs):
    # Прямой проход на обучающих данных
    activations, zs = mlp.forward(X_train)

    # Обратное распространение
    mlp.backward(activations, zs, y_train)

    # Вычисление потерь
    train_pred = activations[-1]
    train_loss = mlp.compute_loss(y_train, train_pred)
    train_losses.append(train_loss)

    # Валидационные потери
    val_activations, _ = mlp.forward(X_val)
    val_pred = val_activations[-1]
    val_loss = mlp.compute_loss(y_val, val_pred)
    val_losses.append(val_loss)

    if epoch % 100 == 0:
        train_acc = np.mean(mlp.predict(X_train) == y_train)
        val_acc = np.mean(mlp.predict(X_val) == y_val)
        print(f"Epoch {epoch:3d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# Финальная оценка на тестовой выборке
test_pred = mlp.predict(X_test)
test_accuracy = np.mean(test_pred == y_test)
test_proba = mlp.predict_proba(X_test)

print(f"\n=== ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ===")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Дополнительные метрики
from sklearn.metrics import classification_report, roc_auc_score

print(f"Test ROC-AUC: {roc_auc_score(y_test, test_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, test_pred, target_names=data.target_names))