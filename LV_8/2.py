import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.transform import resize
from skimage.color import rgb2gray
from tensorflow.keras import models
import numpy as np

img_original = mpimg.imread('test.png')
img = rgb2gray(img_original)
img = resize(img, (28, 28))

plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()

img = img.reshape(1, 28, 28, 1).astype('float32')

model = models.load_model('best_model.h5')
prediction = model.predict(img)
predicted_class = np.argmax(prediction)

print(f'Predikcija: {predicted_class}')
