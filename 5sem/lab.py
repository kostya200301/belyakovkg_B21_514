from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


"""
Лабораторная работа №5
Выделение признаков символов

Вариант ИСПАНСКИЕ ЗАГЛАВНЫЕ БУКВЫ

"""

def weight_black(pixels): # Вычисление веса черного
    np_image = pixels
    height, width = np_image.shape

    q = np_image
    q1 = np_image[:height // 2, :width // 2] # верхний левый уол (UL)
    q2 = np_image[:height // 2, width // 2:] # верхний правый уол (UR)
    q3 = np_image[height // 2:, :width // 2] # нижний левый уол (DL)
    q4 = np_image[height // 2:, width // 2:] # нижний правый уол (DR)

    threshold = 128

    weight_q = np.sum(q < threshold)
    weight_q1 = np.sum(q1 < threshold)
    weight_q2 = np.sum(q2 < threshold)
    weight_q3 = np.sum(q3 < threshold)
    weight_q4 = np.sum(q4 < threshold)
    return {"all_weight": weight_q, "all_weight_norm": weight_q / (height * width), "UL_weight": weight_q1, "UR_weight": weight_q2, "DL_weight": weight_q3, "DR_weight": weight_q4, "UL_weight_norm": weight_q1 / ((height / 4) * (width / 4)), "UR_weight_norm": weight_q2 / ((height / 4) * (width / 4)), "DL_weight_norm": weight_q3 / ((height / 4) * (width / 4)), "DR_weight_norm": weight_q4 / ((height / 4) * (width / 4))}


def get_centre_weight(pixels): # Координаты центра тяжести
    y_coords, x_coords = np.where(pixels == 0)
    return(np.mean(x_coords), np.mean(y_coords))

def get_centre_weight_norm(pixels): # Нормированные координаты центра тяжести
    height, width = pixels.shape
    y_coords, x_coords = np.where(pixels == 0)
    return(np.mean(x_coords) / width, np.mean(y_coords) / height)

def f(a):
    if (a < 255 // 2):
        return 1
    return 0

def get_iners(pixels): # Моменты инерции
    height, width = pixels.shape
    summX = 0
    summY = 0
    x_, y_ = get_centre_weight(pixels)
    for y in range(height):
        for x in range(width):
            summY += f(pixels[y, x]) * (y - y_) ** 2
            summX += f(pixels[y, x]) * (x - x_) ** 2
    return (summX, summY)
def get_iners_norm(pixels): # Нормаированные моменты инерции
    height, width = pixels.shape
    summX = 0
    summY = 0
    x_, y_ = get_centre_weight(pixels)
    for y in range(height):
        for x in range(width):
            summY += f(pixels[y, x]) * (y - y_) ** 2
            summX += f(pixels[y, x]) * (x - x_) ** 2
    return (summX / ((height ** 2) * (width ** 2)), summY / ((height ** 2) * (width ** 2)))

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


# for i in range(1, 28):
#     make_image(i)

def make_all_chars():
    mas = []
    for i in range(1, 28):
        print(i)
        img = Image.open("alphavitBLACK/" + str(i) + ".bmp")
        pixels = np.array(img, dtype=np.uint8)
        dat = [i] + list(weight_black(pixels).values())
        x, y = get_centre_weight(pixels)
        dat.append(x)
        dat.append(y)
        x, y = get_centre_weight_norm(pixels)
        dat.append(x)
        dat.append(y)
        x, y = get_iners(pixels)
        dat.append(x)
        dat.append(y)
        x, y = get_iners_norm(pixels)
        dat.append(x)
        dat.append(y)

        profile_x(pixels, i)
        profile_y(pixels, i)

        mas.append(dat)
    return mas


data = make_all_chars()

import pandas as pd
index_names = ['char_index', 'all_weight',  'all_weight_norm', 'UL_weight', 'UR_weight', 'DL_weight', 'DR_weight', 'UL_weight_norm',
               'UR_weight_norm', 'DL_weight_norm', 'DR_weight_norm', 'centre_weight_x', 'centre_weight_y',
               'centre_weight_x_norm', 'centre_weight_y_norm', 'iner_x', 'iner_y', 'iner_x_norm', 'iner_y_norm']
df = pd.DataFrame(data, columns=index_names)
df.set_index('char_index', inplace=True)
excel_file = 'dataframe.xlsx'
df.to_excel(excel_file, index=True)
