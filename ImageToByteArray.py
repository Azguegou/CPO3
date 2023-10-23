import os
import random
from PIL import Image
import numpy as np


def read_image_to_normalized_pixels(path):
    image = Image.open(path, "r")
    return uniformize_pixel_list(
        list(image.getdata())
    )


def uniformize_pixel_list(pixel_list):
    return np.array([[1] if pixel == (255, 255, 255) else [0] for pixel in pixel_list])


def load_images():
    mapped_answers = []

    for i in range(10):
        folder_path = os.path.join("numbers", str(i))
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            normalized_pixels = read_image_to_normalized_pixels(file_path)
            mapped_answers.append((i, normalized_pixels))

    random.shuffle(mapped_answers)
    return mapped_answers
