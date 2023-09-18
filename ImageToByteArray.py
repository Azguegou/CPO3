# Read directory sorted by expected value
#

def readImage():
    with open("number.png", "rb") as image:
        img = image.read()
        image_in_bytes = bytearray(img)
    return image_in_bytes
