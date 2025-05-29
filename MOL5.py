import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss


# ==============================
# 1. Загрузка или генерация данных
# ==============================

def generate_demo_data():
    """Генерация демонстрационных данных"""
    np.random.seed(42)
    N = 1000  # Количество примеров

    # Генерация 12 признаков
    X = np.random.randn(N, 12) * 2 + 1

    # Генерация меток классов на основе комбинации признаков
    class0 = (X[:, 0] > 0.5) & (X[:, 5] < 1.2)
    Y_labels = np.where(class0, 0, 1)

    # Преобразование в one-hot encoding
    Y = np.zeros((N, 2))
    Y[Y_labels == 0, 0] = 1
    Y[Y_labels == 1, 1] = 1

    return X, Y, Y_labels


# Попытка загрузки данных
data_loaded = False
try:
    if os.path.exists('dataIn.txt') and os.path.exists('dataOut.txt'):
        X = np.loadtxt('dataIn.txt')
        Y = np.loadtxt('dataOut.txt')
        data_loaded = True
        print("Данные успешно загружены из файлов")
except Exception as e:
    print("Ошибка при загрузке данных:", e)

# Если данные не загружены, генерируем демо-данные
if not data_loaded:
    print("Файлы данных не найдены. Используются демонстрационные данные.")
    X, Y, Y_labels = generate_demo_data()
else:
    # Преобразование данных при необходимости
    print("Исходная форма X:", X.shape)
    print("Исходная форма Y:", Y.shape)

    if X.shape[0] != 1000 and X.shape[1] == 12:
        X = X.T
    if Y.shape[0] != 1000 and Y.shape[1] == 2:
        Y = Y.T

    # Преобразуем one-hot в метки классов
    Y_labels = np.argmax(Y, axis=1)

print("Форма данных X:", X.shape)
print("Форма данных Y:", Y.shape)

# Разделение данных
X_train, X_test, Y_train_labels, Y_test_labels = train_test_split(
    X, Y_labels, test_size=0.3, random_state=42
)

# ==============================
# 2. Создание и обучение модели
# ==============================

model = MLPClassifier(
    hidden_layer_sizes=(10,),
    activation='logistic',
    solver='adam',
    learning_rate_init=0.01,
    batch_size=16,
    max_iter=100,
    early_stopping=True,
    validation_fraction=0.2,
    random_state=42,
    verbose=True
)

# Обучение модели
history = model.fit(X_train, Y_train_labels)

# ==============================
# 3. Оценка модели
# ==============================

# Предсказания
Y_pred = model.predict(X_test)
Y_pred_proba = model.predict_proba(X_test)

# Метрики
test_accuracy = accuracy_score(Y_test_labels, Y_pred)
test_loss = log_loss(Y_test_labels, Y_pred_proba)

print("\nTest Loss: {:.4f}".format(test_loss))
print("Test Accuracy: {:.4f}".format(test_accuracy))

# ==============================
# 4. Визуализация результатов
# ==============================

plt.figure(figsize=(10, 5))
plt.plot(Y_test_labels[:100], 'bo', label='Истинные метки')
plt.plot(Y_pred[:100], 'rx', label='Предсказанные метки')
plt.title('Сравнение истинных и предсказанных меток (первые 100 примеров)')
plt.xlabel('Номер примера')
plt.ylabel('Класс (0: победа правящей партии, 1: победа оппозиции)')
plt.legend()
plt.show()

# История обучения
plt.figure(figsize=(12, 5))

# График функции потерь
plt.subplot(1, 2, 1)
plt.plot(history.loss_curve_, label='Тренировочный loss')
plt.title('График Loss')
plt.xlabel('Итерации')
plt.ylabel('Loss')
plt.legend()

# График точности (если доступны данные валидации)
try:
    plt.subplot(1, 2, 2)
    plt.plot(history.validation_scores_, label='Валидационная точность')
    plt.title('График Accuracy')
    plt.xlabel('Итерации')
    plt.ylabel('Accuracy')
    plt.legend()
except AttributeError:
    print("Данные валидационной точности недоступны")

plt.tight_layout()
plt.show()