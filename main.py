import os
from PIL import Image

IMG_SIZE = 32, 32


def openImage(inputFilePath):
    thumbnailFilePath = os.path.splitext(inputFilePath)[0] + ".thumbnail"
    if os.path.isfile(thumbnailFilePath):
        try:
            img = Image.open(thumbnailFilePath)
            img = img.convert("L")
            return img
        except IOError:
            print("cannot open thumbnail for '%s'" % inputFilePath)
    else:
        try:
            img = Image.open(inputFilePath)
            img = img.resize(IMG_SIZE, Image.ANTIALIAS)
            img = img.convert("L")
            img.save(thumbnailFilePath, "JPEG")
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

