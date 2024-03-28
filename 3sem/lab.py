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

def make_image():
    img = Image.open("images/im.jpg")
    pixels = np.array(img, dtype=np.uint8)

    image = Image.fromarray(convert_to_grayscale(pixels))
    image.save("images/img.bmp")



def task1():
    img = Image.open("images/img.bmp")
    pixels = np.array(img, dtype=np.uint8)

    n = 3
    mask = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.uint8)
    # mask = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    new_pixels = np.zeros((pixels.shape[0], pixels.shape[1]), dtype=np.uint8)

    for y in range(n // 2, img.height - n // 2):
        for x in range(n // 2, img.width - n // 2):
            wind = pixels[y - n // 2: y + n // 2 + 1, x - n // 2: x + n // 2 + 1]
            new_pixels[y, x] = np.sum(wind * mask) / np.sum(mask)

    image = Image.fromarray(new_pixels)
    image.save("images/img2.bmp")

def task2():
    img1 = Image.open("images/img.bmp")
    img2 = Image.open("images/img2.bmp")

    arr1 = np.array(img1, dtype=np.uint8)
    arr2 = np.array(img2, dtype=np.uint8)

    xor_result = np.bitwise_xor(arr1, arr2)

    xor_image = Image.fromarray(xor_result)
    xor_image.save("images/img3.bmp")


if __name__ == "__main__":
    # task1()
    task2()
