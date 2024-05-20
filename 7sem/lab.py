from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

"""

Лабораторная работа № 7

Классификация на основе признаков, анализ профилей



"""

ALPHAVIT = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
STR = "AMOMUCHOTUSOJOS"


class Feature:
    def __init__(self, pixels):
        self.pixels = pixels

    def weight_black(self):
        np_image = self.pixels
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


    def get_centre_weight(self):
        y_coords, x_coords = np.where(self.pixels == 0)
        return(np.mean(x_coords), np.mean(y_coords))

    def get_centre_weight_norm(self):
        height, width = self.pixels.shape
        y_coords, x_coords = np.where(self.pixels == 0)
        return(np.mean(x_coords) / width, np.mean(y_coords) / height)

    def f(self, a):
        if (a < 255 // 2):
            return 1
        return 0

    def get_iners(self):
        height, width = self.pixels.shape
        summX = 0
        summY = 0
        x_, y_ = self.get_centre_weight()
        for y in range(height):
            for x in range(width):
                summY += self.f(self.pixels[y, x]) * (y - y_) ** 2
                summX += self.f(self.pixels[y, x]) * (x - x_) ** 2
        return (summX, summY)

    def get_iners_norm(self):
        height, width = self.pixels.shape
        summX = 0
        summY = 0
        x_, y_ = self.get_centre_weight()
        for y in range(height):
            for x in range(width):
                summY += self.f(self.pixels[y, x]) * (y - y_) ** 2
                summX += self.f(self.pixels[y, x]) * (x - x_) ** 2
        return (summX / ((height ** 2) * (width ** 2)), summY / ((height ** 2) * (width ** 2)))




def make_chars_features_df(alf=0): # Рассчет параметров для алфавита/строки12/строки14
    mas = []

    import os
    directory = "../LAB6/chars"
    if (alf == 1):
        directory = "../LAB5/alphavitBLACK"
    elif (alf == 2):
        directory = "../LAB6/chars14"

    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    for i in range(len(files)):
        if (alf == 1):
            img = Image.open("../LAB5/alphavitBLACK/" + str(i + 1) + ".bmp")
        elif (alf == 0):
            img = Image.open("../LAB6/chars/" + str(i + 1) + ".bmp")
        else:
            img = Image.open("../LAB6/chars14/" + str(i + 1) + ".bmp")

        pixels = np.array(img, dtype=np.uint8)
        ff = Feature(pixels)
        dat = [i] + list(ff.weight_black().values())
        x, y = ff.get_centre_weight()
        dat.append(x)
        dat.append(y)
        x, y = ff.get_centre_weight_norm()
        dat.append(x)
        dat.append(y)
        x, y = ff.get_iners()
        dat.append(x)
        dat.append(y)
        x, y = ff.get_iners_norm()
        dat.append(x)
        dat.append(y)

        mas.append(dat)

    import pandas as pd
    index_names = ['char_index', 'all_weight', 'all_weight_norm', 'UL_weight', 'UR_weight', 'DL_weight',
                   'DR_weight', 'UL_weight_norm',
                   'UR_weight_norm', 'DL_weight_norm', 'DR_weight_norm', 'centre_weight_x', 'centre_weight_y',
                   'centre_weight_x_norm', 'centre_weight_y_norm', 'iner_x', 'iner_y', 'iner_x_norm', 'iner_y_norm']
    df = pd.DataFrame(mas, columns=index_names)
    df.set_index('char_index', inplace=True)
    return df


# def get_distance(char1_prop, char2_prop): # получить меру близости
#     return ((char1_prop["all_weight_norm"] - char2_prop["all_weight_norm"]) ** 2 + \
#             (char1_prop["UL_weight_norm"] - char2_prop["UL_weight_norm"]) ** 2 + \
#             (char1_prop["UR_weight_norm"] - char2_prop["UR_weight_norm"]) ** 2 + \
#             (char1_prop["DL_weight_norm"] - char2_prop["DL_weight_norm"]) ** 2 + \
#             (char1_prop["DR_weight_norm"] - char2_prop["DR_weight_norm"]) ** 2 + \
#             (char1_prop["centre_weight_x_norm"] - char2_prop["centre_weight_x_norm"]) ** 2 + \
#             (char1_prop["centre_weight_y_norm"] - char2_prop["centre_weight_y_norm"]) ** 2 + \
#             (char1_prop["iner_x_norm"] - char2_prop["iner_x_norm"]) ** 2 + \
#             (char1_prop["iner_y_norm"] - char2_prop["iner_y_norm"]) ** 2) ** 0.5

def get_distance(char1_prop, char2_prop): # получить меру близости
    a1 = (char1_prop["all_weight_norm"] - char2_prop["all_weight_norm"]) ** 2
    a2 = (char1_prop["UL_weight_norm"] - char2_prop["UL_weight_norm"]) ** 2
    a3 = (char1_prop["UR_weight_norm"] - char2_prop["UR_weight_norm"]) ** 2
    a4 = (char1_prop["DL_weight_norm"] - char2_prop["DL_weight_norm"]) ** 2
    a5 = (char1_prop["DR_weight_norm"] - char2_prop["DR_weight_norm"]) ** 2
    a6 = (char1_prop["centre_weight_x_norm"] - char2_prop["centre_weight_x_norm"]) ** 2
    a7 = (char1_prop["centre_weight_y_norm"] - char2_prop["centre_weight_y_norm"]) ** 2
    a8 = (char1_prop["iner_x_norm"] - char2_prop["iner_x_norm"]) ** 2
    a9 = (char1_prop["iner_y_norm"] - char2_prop["iner_y_norm"]) ** 2
    return ((a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9) ** 0.5) / 9

def get_measure_proximity(char1_prop, char2_prop):
    return 1 - get_distance(char1_prop, char2_prop)


def get_str_measures(df_str, df_alf): # Расстрояние каждой буквы строки до каждой буквы алфавита
    data = []
    for i in range(df_str.shape[0]):
        mas = []
        for j in range(df_alf.shape[0]):
            mas.append((ALPHAVIT[j], get_measure_proximity(df_str.iloc[i], df_alf.iloc[j])))
        sorted_mas = sorted(mas, key=lambda x: x[1], reverse=True)
        data.append(sorted_mas)
    return data

df_chars_features_str_12 = make_chars_features_df() # Получаем все признаки букв строки (шрифт 12)
df_chars_features_str_14 = make_chars_features_df(2) # Получаем все признаки букв строки (шрифт 14)
df_chars_features_alphavit = make_chars_features_df(1) # Получаем все признаки букв алфавита

# Для шрифта того же размера что и в алфавите
data = get_str_measures(df_chars_features_str_12, df_chars_features_alphavit)
count_ygadal = 0
for i in range(len(data)):
    print(data[i])
    if data[i][0][0] == STR[i]:
        count_ygadal += 1

print("\nшрифт в алафмте 12 в строке 12")
print("Ygadal", str(count_ygadal) + "/" + str(len(STR)), " Bykv, eto", count_ygadal / len(STR) * 100, "procentov\n")


# Для шрифта большего размера чем в алфавите
data = get_str_measures(df_chars_features_str_14, df_chars_features_alphavit)
count_ygadal = 0
for i in range(len(data)):
    print(data[i])
    if data[i][0][0] == STR[i]:
        count_ygadal += 1

print("\nшрифт в алафмте 12 в строке 14")
print("Ygadal", str(count_ygadal) + "/" + str(len(STR)), " Bykv, eto", count_ygadal / len(STR) * 100, "procentov\n")


