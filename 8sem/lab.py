from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

"""
Лабораторная работа № 8
Текстурный анализ и контрастирование

Вариант 12 

"""

def load_image(filename):
    img = Image.open(filename)
    img_gray = img.convert('L')  # оттенки серого
    img_gray.save(filename.split(".")[0] + "_gray.jpg")
    return np.array(img_gray)

def plot_histogram(image, name):
    plt.figure()
    plt.hist(image.ravel(), bins=256, range=(0, 256), density=True, color='gray', alpha=0.7)
    plt.title('Гистограмма изображения')
    plt.xlabel('Уровень серого')
    plt.ylabel('Частота')
    plt.savefig("images/" + name + "_hist.jpg")
    plt.close()

def equalize_histogram(image, name): # выравнивания гистограммы изображения
    img = Image.fromarray(image)
    img_eq = ImageOps.equalize(img)
    img_eq.save("images/" + name.split(".")[0] + "_eq.jpg")
    return np.array(img_eq)


def get_glcm(image, distance, angle): # создания матрицы Харалика
    max_gray_level = np.max(image) + 1
    glcm = np.zeros((max_gray_level, max_gray_level))

    # смещения по углам
    angle_radians = np.deg2rad(angle)  # в радианы
    dx = round(np.cos(angle_radians) * distance)
    dy = round(np.sin(angle_radians) * distance)

    # находим GLCM
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            # Координаты соседних пикселей
            x2, y2 = x + dx, y + dy
            if 0 <= x2 < image.shape[1] and 0 <= y2 < image.shape[0]:
                glcm[image[y, x], image[y2, x2]] += 1

    # нормализация
    glcm = glcm / np.sum(glcm)
    return glcm


# вычисление CORR
def get_corr(glcm):
    mean_x = np.sum(glcm * np.arange(glcm.shape[0])[:, None])
    mean_y = np.sum(glcm * np.arange(glcm.shape[1])[None, :])
    std_x = np.sqrt(np.sum(glcm * (np.arange(glcm.shape[0])[:, None] - mean_x) ** 2))
    std_y = np.sqrt(np.sum(glcm * (np.arange(glcm.shape[1])[None, :] - mean_y) ** 2))

    numerator = np.sum(((np.arange(glcm.shape[0])[:, None] - mean_x) * (np.arange(glcm.shape[1])[None, :] - mean_y) * glcm))

    corr = numerator / (std_x * std_y)
    return corr


# Визуализация GLCM
def draw_glcm(glcm, angle, name):
    plt.figure()
    plt.imshow(glcm, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.title('GLCM (Матрица Харалика)')
    plt.xlabel('Уровень серого')
    plt.ylabel('Уровень серого')
    plt.savefig("images/" + name + "_glcm_" + str(angle) + ".jpg")
    plt.close()



images = ["img1.jpg", "img4.jpg", "img5.jpg"]


for im in images:
    print(im)
    image = load_image('images/' + im)

    distance = 1

    angles = [45, 135, 225, 315]

    for angle in angles:
        glcm = get_glcm(image, distance, angle)

        corr = get_corr(glcm)
        print("Угол", angle, "градусов - Корреляция (CORR):", corr)

        draw_glcm(glcm, angle, im.split(".")[0])

    print()

for im in images:
    print(im + "_eq_his")
    image = load_image('images/' + im)

    image_eq = equalize_histogram(image, im) # выравнивание гистограммы изображения

    distance = 1

    angles = [45, 135, 225, 315]

    for angle in angles:
        glcm = get_glcm(image_eq, distance, angle)

        corr = get_corr(glcm)
        print("Угол", angle, "градусов - Корреляция (CORR):", corr)

        draw_glcm(glcm, angle, im.split(".")[0])

    plot_histogram(image, im.split(".")[0])
    plot_histogram(image_eq, im.split(".")[0] + "_eq")
    print()



