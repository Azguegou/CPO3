# Read directory sorted by expected value
#

def read_image(path="number.png"):
    with open(path, "rb") as image:
        img = image.read()
        image_in_bytes = bytearray(img)
    return image_in_bytes
