from PIL import Image
import numpy as np
import math

def convert_to_grayscale(pixels, fk=(1, 1, 1)):
    new_pixels = np.zeros((pixels.shape[0], pixels.shape[1]), dtype=np.uint8)
    for x in range(0, pixels.shape[0] - 1):
        for y in range(0, pixels.shape[1] - 1):
            new_value = (fk[0] * int(pixels[x, y, 0]) + fk[1] * int(pixels[x, y, 1]) + fk[2] * int(pixels[x, y, 2])) // 3
            new_pixels[x, y] = new_value
    return new_pixels

def task1():
    img = Image.open("images/img.jpeg")
    pixels = np.array(img, dtype=np.uint8)

    image = Image.fromarray(convert_to_grayscale(pixels))
    image.save("images/img1.bmp")



def bin_nick(pixels, window_size=25, k=-0.1):
    new_h = pixels.shape[0] // window_size * window_size
    new_w = pixels.shape[1] // window_size * window_size

    new_pixels = np.zeros((new_h, new_w), dtype=np.uint8)

    for x in range(new_w // window_size - 1):
        for y in range(new_h // window_size - 1):
            wind = pixels[y * window_size: (y + 1) * window_size, x * window_size: (x + 1) * window_size, :]
            wind = np.sum(wind, axis=2) / 3
            m = np.mean(wind)
            squared_sum = np.sum(np.square(wind))
            N = window_size * window_size

            T = m + k * math.sqrt(squared_sum / N - m * m / N)

            for x1 in range(window_size):
                for y1 in range(window_size):
                    if (wind[y1, x1] < T):
                        new_pixels[y * window_size + y1, x * window_size + x1] = 0
                    else:
                        new_pixels[y * window_size + y1, x * window_size + x1] = 255
    return new_pixels

def task2():
    img = Image.open("images/img.jpeg")
    pixels = np.array(img, dtype=np.uint8)


    image = Image.fromarray(bin_nick(pixels))
    image.save("images/img2.bmp")


if __name__ == "__main__":
    task1()
    task2()



