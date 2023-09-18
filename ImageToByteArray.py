# Read directory sorted by expected value
#
with open("number.png", "rb") as image:
    img= image.read()
    imgByte = bytearray(img)
    print(imgByte)
