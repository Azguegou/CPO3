# Read directory sorted by expected value
import os
import random

from PIL import Image


def read_image(path="number.png"):
    with open(path, "rb") as image:
        img = image.read()
        image_in_bytes = bytearray(img)
    return image_in_bytes


def read_image_to_normalized_pixels(path):
    image = Image.open(path, "r")
    return uniformize_pixel_list(
        list(image.getdata())
    )


def uniformize_pixel_list(pixel_list):
    return [1 if pixel == (255, 255, 255) else 0 for pixel in pixel_list]


def load_images():
    mapped_answers = []

    for i in range(10):
        for file_name in os.listdir("numbers/" + str(i)):
            mapped_answers.append((i, read_image_to_normalized_pixels("numbers/" + str(i) + "/" + file_name)))

    random.shuffle(mapped_answers)

    return mapped_answers
