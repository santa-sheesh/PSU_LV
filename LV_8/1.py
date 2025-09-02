from tensorflow.keras import datasets, models, layers, callbacks, utils
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train[..., np.newaxis] / 255.0
x_test = x_test[..., np.newaxis] / 255.0

y_train_cat = utils.to_categorical(y_train, 10)
y_test_cat = utils.to_categorical(y_test, 10)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

cb = [
    callbacks.TensorBoard(log_dir='logs', update_freq=100),
    callbacks.ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
]

model.fit(x_train, y_train_cat, epochs=10, batch_size=64, validation_split=0.1, callbacks=cb)

best_model = models.load_model('best_model.h5')

train_loss, train_acc = best_model.evaluate(x_train, y_train_cat, verbose=0)
test_loss, test_acc = best_model.evaluate(x_test, y_test_cat, verbose=0)
print(f'Točnost na train skupu: {train_acc:.4f}')
print(f'Točnost na testnom skupu: {test_acc:.4f}')

y_pred = np.argmax(best_model.predict(x_test), axis=1)
y_true = y_test

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matrica zabune - Testni skup')
plt.xlabel('Predviđeno')
plt.ylabel('Stvarno')
plt.show()
