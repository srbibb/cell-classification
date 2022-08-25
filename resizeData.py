from PIL import Image
from pathlib import Path
import tifffile

def resizeData(inputDirectory, filename):
    with Image.open(inputDirectory) as img:
        img = img.resize([224,224])
        img.save("D:/icm/data/resized/" + filename + "_resized.png", "PNG")

paths = Path("D:/icm/data/cropped/pngs/").glob("*.png")
for path in paths:
    resizeData(path, path.stem)

def rotateData(inputDirectory, filename):
    with Image.open(inputDirectory) as img:
        img = img.rotate(180)
        img.save("D:/icm/data/cropped/resized/" + filename + "_rotated.png", "PNG")

'''
paths = Path("D:/icm/data/cropped/resized/").glob("*.png")
for path in paths:
    rotateData(path, path.stem)
    '''