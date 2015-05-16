import os
from PIL import Image

IMG_SIZE = 32, 32


def openImage(inputFilePath):
    outputFilePath = os.path.splitext(inputFilePath)[0] + ".thumbnail"
    try:
        img = Image.open(inputFilePath)
        img = img.resize(IMG_SIZE, Image.ANTIALIAS)
        img = img.convert("L")
        img.save(outputFilePath, "JPEG")
        return img
    except IOError:
        print("cannot create thumbnail for '%s'" % inputFilePath)


def printPixel(pix):
    for i in range(0, IMG_SIZE[0]):
        for y in range(0, IMG_SIZE[1]):
            print(pix[i, y]),
        print


img = openImage("m001.jpg")
pix = img.load()
# printPixel(pix)

