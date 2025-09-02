from matplotlib import pyplot as plt
import numpy as np

img = plt.imread("tiger.png")[:, :, 0]

img_bright = np.clip(img + 0.6, 0, 1)
img_rot = np.rot90(img, 3)
img_flip = np.fliplr(img)
img_small = img[::5, ::5]

pr_img = img.copy()
pr_img[:, :img.shape[1]//4] = 0
pr_img[:, img.shape[1]//2:] = 0

plt.figure(1)
plt.title("a) brightness")
plt.imshow(img_bright, cmap="gray")
plt.figure(2)
plt.title("b) rotirana slika")
plt.imshow(img_rot, cmap="gray")
plt.figure(3)
plt.title("c) zrcaljena slika")
plt.imshow(img_flip, cmap="gray")
plt.figure(4)
plt.title("d) smanjena kvaliteta slike")
plt.imshow(img_small, cmap="gray")
plt.figure(5)
plt.title("e) stupci")
plt.imshow(pr_img, cmap="gray")
plt.show()
