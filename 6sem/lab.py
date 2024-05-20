from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

"""
Лаюораторная работа № 6
Сегментация текста

Разделение строки на буквы и построение профилей

"""

def convert_to_grayscale(pixels, fk=(1, 1, 1)):
    new_pixels = np.zeros((pixels.shape[0], pixels.shape[1]), dtype=np.uint8)
    for x in range(0, pixels.shape[0]):
        for y in range(0, pixels.shape[1]):
            new_value = (fk[0] * int(pixels[x, y, 0]) + fk[1] * int(pixels[x, y, 1]) + fk[2] * int(pixels[x, y, 2])) // 3
            new_pixels[x, y] = new_value
    return new_pixels

def make_image():
    img = Image.open("love.jpg")
    pixels = np.array(img, dtype=np.uint8)

    image = Image.fromarray(convert_to_grayscale(pixels))
    image.save("love.bmp")


def f(a):
    if (a < 255 // 2):
        return 1
    return 0

def profile_x(pixels, index):
    new_pixels = np.zeros((pixels.shape[0], pixels.shape[1]), dtype=np.uint8)
    for y in range(pixels.shape[0]):
        for x in range(pixels.shape[1]):
            new_pixels[y, x] = f(pixels[y, x])
    height, width = pixels.shape
    plt.bar(x=np.arange(1, width + 1), height=np.sum(new_pixels, axis=0))
    plt.xlim(0, width + 1)
    plt.ylim(0, height + 1)
    plt.savefig("profils/x_" + str(index) + ".png")
    plt.clf()
    return (np.arange(1, width + 1), np.sum(new_pixels, axis=0))

def profile_y(pixels, index):
    new_pixels = np.zeros((pixels.shape[0], pixels.shape[1]), dtype=np.uint8)
    for y in range(pixels.shape[0]):
        for x in range(pixels.shape[1]):
            new_pixels[y, x] = f(pixels[y, x])
    height, width = pixels.shape
    plt.barh(y=np.arange(1, height + 1), width=np.sum(new_pixels, axis=1))
    plt.xlim(0, width + 1)
    plt.ylim(height + 1, 0)
    plt.savefig("profils/y_" + str(index) + ".png")
    plt.clf()
    return (np.arange(1, height + 1), np.sum(new_pixels, axis=1))

# make_image()

def split_chars(x_h): # Разделение предложения на буквы
    indexes = []
    for i in range(1, len(x_h) - 1):
        if (x_h[i] != 0 and x_h[i + 1] == 0) or (x_h[i] != 0 and x_h[i - 1] == 0):
            indexes.append(i)

    if indexes[0] > 4 and x_h[0] != 0:
        indexes.insert(0, 0)
    if len(x_h) - indexes[-1] > 4 and x_h[-1] != 0:
        indexes.append(len(x_h) - 1)
    pairs = []
    print(len(x_h))
    print(indexes)

    for i in range(0, len(indexes), 2):
        pairs.append([indexes[i], indexes[i + 1]])
    return pairs

def draw_char(pixels, start, end, ind):
    pixels_new = pixels[:, start: end + 1]
    image = Image.fromarray(pixels_new)
    image.save("chars/" + str(ind) + ".bmp")
    # image.show()


make_image()

img = Image.open("love.bmp")
pixels = np.array(img, dtype=np.uint8)
x_x, x_h = profile_x(pixels, 0)
y_y, y_w = profile_y(pixels, 0)



mas = split_chars(x_h)
ind = 1
for i in mas:
    draw_char(pixels, i[0], i[1], ind)
    profile_x(pixels[:, i[0]: i[1] + 1], ind)
    profile_y(pixels[:, i[0]: i[1] + 1], ind)
    ind += 1
# draw_char(pixels, 0, 10)