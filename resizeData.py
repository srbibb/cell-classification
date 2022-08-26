from PIL import Image
from pathlib import Path

# Resizes the files to the format that the VGG16 neural network wants which is images that are 224x224.
# The images are saved to a folder which is used as the input for the network.

def resizeData(inputDirectory, filename):
    with Image.open(inputDirectory) as img:
        img = img.resize([224,224])
        img.save("D:/icm/data/resized/" + filename + "_resized.png", "PNG")

paths = Path("D:/icm/data/cropped/pngs/").glob("*.png")
for path in paths:
    resizeData(path, path.stem)
