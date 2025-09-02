import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_train[i], cmap="gray")
    plt.axis("off")
plt.suptitle("Primjeri iz train skupa")
plt.show()

x_train_s = x_train.reshape(-1, 784).astype("float32") / 255
x_test_s = x_test.reshape(-1, 784).astype("float32") / 255

y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)

model = Sequential([
    Input(shape=(784,)),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
model.summary()

model.fit(x_train_s, y_train_s, epochs=5, batch_size=32)

train_loss, train_acc = model.evaluate(x_train_s, y_train_s)
test_loss, test_acc = model.evaluate(x_test_s, y_test_s)
print(f"Tocnost na train skupu: {train_acc:.4f}")
print(f"Tocnost na test skupu: {test_acc:.4f}")

y_pred_classes = np.argmax(model.predict(x_test_s), axis=1)
cm = confusion_matrix(y_test, y_pred_classes)
ConfusionMatrixDisplay(cm, display_labels=np.arange(10)).plot(cmap=plt.cm.Blues)
plt.title("Matrica zabune na test skupu")
plt.show()

wrong = np.where(y_test != y_pred_classes)[0]
plt.figure(figsize=(10, 5))
for i in range(5):
    idx = wrong[i]
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[idx], cmap="gray")
    plt.title(f"Stvarna: {y_test[idx]}\nPred: {y_pred_classes[idx]}")
    plt.axis("off")
plt.suptitle("Pogrešne klasifikacije")
plt.show()
