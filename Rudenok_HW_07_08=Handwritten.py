import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from collections import Counter
import seaborn as sns
from datetime import datetime


class Perceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.output_size = output_size
        self.learning_rate = learning_rate

        # Ініціалізація ваг (використаємо метод Xavier)
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(1. / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))

        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(1. / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))

        self.loss_history = []
        self.accuracy_history = []

    def predict(self, X):
        y_pred = self.forward_propagation(X)
        return np.argmax(y_pred, axis=1)

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)   # Запобігання переповненню
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward_propagation(self, X):
        # Прямий прохід
        self.Z1 = np.dot(X, self.W1) + self.b1  # Лінійне перетворення
        self.A1 = self.sigmoid(self.Z1)         # Активація

        self.Z2 = np.dot(self.A1, self.W2) + self.b2    # Лінійне перетворення
        self.A2 = self.softmax(self.Z2)

        return self.A2

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]     # Кількість зразків
        epsilon = 1e-15         # Маленьке значення для запобігання log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Запобігання log(0)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        return loss

    def sigmoid_derivative(self, A):
        return A * (1 - A)

    def backward_propagation(self, X, y_true, y_pred):
        m = X.shape[0]

        # Обраховуємо помилку вихідного шару
        dZ2 = y_pred - y_true
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # Обраховуємо помилку прихованого шару
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Оновлюємо ваги та зсуви
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def caluclate_accuracy(self, y_true, y_pred):
        y_true_labels = np.argmax(y_true, axis=1)
        y_pred_labels = np.argmax(y_pred, axis=1)
        accuracy = np.mean(y_true_labels == y_pred_labels)
        return accuracy

    def train(self, X_train, y_train, X_val, y_val, epochs=1000, verbose=True):
        for epoch in range(epochs):
            # Прямий прохід
            y_pred_train = self.forward_propagation(X_train)

            # Обчислення втрат
            loss = self.compute_loss(y_train, y_pred_train)
            self.loss_history.append(loss)

            # Зворотний прохід (оновлення ваг)
            self.backward_propagation(X_train, y_train, y_pred_train)

            # Обчислення точності на валідаційному наборі
            if epoch % 50 == 0 or epoch == epochs - 1:
                val_pred = self.forward_propagation(X_val)
                val_accuracy = self.caluclate_accuracy(y_val, val_pred)
                self.accuracy_history.append(val_accuracy)

                if verbose and (epoch % 50 == 0 or epoch == epochs - 1):
                    current_time = datetime.now().strftime("%H:%M:%S")  # формат ГГ:ХХ:СС
                    print(f"[{current_time}] Епоха {epoch + 1}/{epochs} - Втрата: {loss:.4f} - Точність на валідації: {val_accuracy:.4f}")

def one_hot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

def balance_dataset(X, y):
    # Балансування датасету: вирівнює кількість прикладів для всіх класів

    # Створюємо два порожні списки для збереження збалансованих даних
    X_balanced, y_balanced = [], []

    # Рахуємо кількість прикладів для кожного класу і знаходимо найменшу кількість
    # (це буде цільовий розмір вибірки для кожного класу)
    min_count = min(np.bincount(y))

    # Проходимо по кожному унікальному класу в мітках
    for cls in np.unique(y):
        # Вибираємо всі приклади X, що належать до поточного класу cls
        X_cls = X[y == cls]

        # Вибираємо відповідні мітки (усі вони дорівнюють cls)
        y_cls = y[y == cls]

        # Балансуємо дані цього класу:
        # - Якщо прикладів більше за min_count → undersampling (беремо підмножину)
        # - Якщо прикладів менше за min_count → oversampling (беремо з повторенням)
        X_res, y_res = resample(
            X_cls,              # дані класу
            y_cls,              # мітки класу
            replace=len(y_cls) < min_count,  # якщо клас рідкісний → дозволяємо дублювати приклади
            n_samples=min_count,             # робимо рівно min_count прикладів
            random_state=42                  # фіксуємо випадковість для повторюваності результату
        )

        # Додаємо збалансовані приклади цього класу у загальний список
        X_balanced.append(X_res)
        y_balanced.append(y_res)

    # Об’єднуємо усі збалансовані класи в одну матрицю (X) та один вектор (y)
    return np.vstack(X_balanced), np.hstack(y_balanced)

def load_and_prepare_data():
    print(f"[{datetime.now().strftime("%H:%M:%S")}] Завантаження датасету A-Z...")

    # завантажуємо CSV (знаходиться в моїй папці з д/з)
    data = pd.read_csv("A_Z Handwritten Data.csv").values

    X = data[:, 1:]  # пікселі
    y = data[:, 0].astype(int)  # мітки (0-25)

    print(f"[{datetime.now().strftime("%H:%M:%S")}] Кількість прикладів кожного класу до балансування:", Counter(y))

    # Балансуємо датасет
    X, y = balance_dataset(X, y)

    print(f"[{datetime.now().strftime("%H:%M:%S")}] Кількість прикладів кожного класу після балансування:", Counter(y))

    # Розділення даних на навчальні, валідаційні та тестові набори
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    # Нормалізація даних
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Перетворення міток у формат one-hot encoding
    y_train_onehot = one_hot_encode(y_train, num_classes=26)
    y_val_onehot = one_hot_encode(y_val, num_classes=26)
    y_test_onehot = one_hot_encode(y_test, num_classes=26)

    return (
        X_train, y_train_onehot,
        X_val, y_val_onehot,
        X_test, y_test_onehot,
        y_test, scaler      # Збереження оригінальних міток для оцінки
    )

def evaluate_model(model, X_test, y_test_onehot, y_test_labels):
    y_pred = model.forward_propagation(X_test)
    test_accuracy = model.caluclate_accuracy(y_test_onehot, y_pred)
    print(f"\nТочність на тестовому наборі: {test_accuracy:.4f}")

    y_pred_labels = np.argmax(y_pred, axis=1)
    print("\nЗвіт про класифікацію:")
    print(classification_report(y_test_labels, y_pred_labels))

    # Матриця плутанини
    cm = confusion_matrix(y_test_labels, y_pred_labels)

    # Підписи класів — літери A–Z
    labels = [chr(i + 65) for i in range(26)]

    mpl.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels)
    mpl.xlabel('Передбачені літери')
    mpl.ylabel('Справжні літери')
    mpl.title('Матриця плутанини')
    mpl.show()

    return test_accuracy

def show_predictions(model, X_test, y_test_labels, scaler=None):
    print("Відображення передбачень моделі...")
    predictions = model.predict(X_test)

    # Деанормалізація
    if scaler is not None:
        X_test_denorm = scaler.inverse_transform(X_test)
    else:
        X_test_denorm = X_test

    # Функція для перетворення індексу у літеру 0 -> 'A', 1 -> 'B', ..., 25 -> 'Z'
    def idx_to_letter(idx):
        return chr(idx + 65)

    # Беремо всі 26 класів (0..25)
    all_classes = np.arange(26)

    # Вибираємо по одному випадковому зразку з кожного класу
    indices = [np.random.choice(np.where(y_test_labels == cls)[0]) for cls in all_classes]

    # Створюємо сітку 5x6 (бо 26 класів)
    fig, axes = mpl.subplots(5, 6, figsize=(18, 14))
    fig.suptitle("Передбачення моделі для всіх 26 літер", fontsize=18)

    for i, idx in enumerate(indices):
        row = i // 6
        col = i % 6

        image = X_test_denorm[idx].reshape(28, 28)
        axes[row, col].imshow(image, cmap='gray')

        correct = predictions[idx] == y_test_labels[idx]
        color = 'green' if correct else 'red'

        axes[row, col].set_title(
            f"{idx_to_letter(y_test_labels[idx])} → {idx_to_letter(predictions[idx])}",
            color=color,
            fontsize=10
        )
        axes[row, col].axis('off')

    # Прибираємо порожні клітинки (якщо їх більше, ніж 26)
    for j in range(len(indices), 5 * 6):
        axes[j // 6, j % 6].axis('off')

    mpl.show()

def main():
    print(" -- ПЕРЦЕПТРОН ДЛЯ РОЗПІЗНАВАННЯ ЛІТЕР (A-Z) --")

    X_train, y_train, X_val, y_val, X_test, y_test_onehot, y_test_labels, scaler = load_and_prepare_data()

    # Створення та налаштування моделі перцептрона
    input_size = X_train.shape[1]  # 784 для зображень 28x28
    output_size = 26        # Кількість класів (літери A-Z)
    hidden_size = 256       # Кількість нейронів у прихованому шарі
    learning_rate = 2       # Швидкість навчання
    epochs = 1000           # Кількість епох

    print("Параметри моделі:")
    print(f" - Розмір вхідного шару: {input_size}")
    print(f" - Розмір прихованого шару: {hidden_size}")
    print(f" - Розмір вихідного шару: {output_size}")
    print(f" - Швидкість навчання: {learning_rate}")

    # Ініціалізація моделі
    perceptron = Perceptron(input_size, hidden_size, output_size, learning_rate)

    # Навчання моделі
    perceptron.train(X_train, y_train, X_val, y_val, epochs=epochs, verbose=True)

    # Оцінка моделі на тестовому наборі
    evaluate_model(perceptron, X_test, y_test_onehot, y_test_labels)

    # Відображення передбачень
    show_predictions(perceptron, X_test, y_test_labels, scaler)

    return perceptron


if __name__ == "__main__":
    model = main()
