from PIL import Image
import numpy as np


"""

Лабораторная работа №4. Выделение контуров на 
изображении

Вариант 1
Оператор Робертса

"""

def convert_to_grayscale(pixels, fk=(1, 1, 1)):
    new_pixels = np.zeros((pixels.shape[0], pixels.shape[1]), dtype=np.uint8)
    for x in range(0, pixels.shape[0] - 1):
        for y in range(0, pixels.shape[1] - 1):
            new_value = (fk[0] * int(pixels[x, y, 0]) + fk[1] * int(pixels[x, y, 1]) + fk[2] * int(pixels[x, y, 2])) // 3
            new_pixels[x, y] = new_value
    return new_pixels

def make_image():
    img = Image.open("images/img.jpg")
    pixels = np.array(img, dtype=np.uint8)

    image = Image.fromarray(convert_to_grayscale(pixels))
    image.save("images/img.bmp")


def find_max_g(pixels):
    max_g = 0
    for y in range(0, pixels.shape[0] - 3):
        for x in range(0, pixels.shape[1] - 3):
            wind = pixels[y: y + 3, x: x + 3]
            Gx_ = 1 * wind[2, 2] - (-1) * wind[1, 1]
            Gy_ = 1 * wind[2, 1] - (-1) * wind[1, 2]
            G = (Gx_ ** 2 + Gy_ ** 2) ** 0.5
            max_g = max(G, max_g)
    return max_g
def task1(T=200, index=0):
    Gx = np.array([[0, 0, 0], # Частные производные в матричном видe
                   [0, -1, 0],
                   [0, 0, 1]])

    Gy = np.array([[0, 0, 0], # Частные производные в матричном видe
                   [0, 0, -1],
                   [0, 1, 0]])

    img = Image.open("images/img.bmp")
    pixels = np.array(img, dtype=np.uint8)
    new_pixels = np.zeros((pixels.shape[0], pixels.shape[1]), dtype=np.uint8)
    max_G = find_max_g(pixels) # Максимальный градиент для нормировки
    window_size = 3
    for y in range(0, img.height - 3):
        for x in range(0, img.width - 3):
            wind = pixels[y: y + 3, x: x + 3]
            Gx_ = 1 * wind[2, 2] - (-1) * wind[1, 1] # частная произвожная по x
            Gy_ = 1 * wind[2, 1] - (-1) * wind[1, 2] # частная произвожная по y
            G = (Gx_ ** 2 + Gy_ ** 2) ** 0.5
            G_norm = G * 255 / max_G # Нормализованный градиент в точке
            if G_norm > T:
                new_pixels[y, x] = 255
            else:
                new_pixels[y, x] = 0

    image = Image.fromarray(new_pixels)
    image.save("images/img_new_" + str(index) + ".bmp")

make_image()

for T in range(50, 251, 25): # перебираем пороги
    task1(T, T)

# 75 лучше всего
