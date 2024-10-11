from PIL import Image, ImageDraw
import os


def add_rectangle(image_path, size=(1030, 282), color='black'):
    """Adds a rectangle to the bottom right corner of an image"""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Calculate the bottom right corner of the rectangle
    x = image.size[0] - size[0]
    y = image.size[1] - size[1]

    draw.rectangle(((x, y), (image.size[0], image.size[1])), fill=color)
    image.save(image_path)


# specify the directory that contains the images
image_directory = r"G:\blue"

# loop over all files in the directory and its subdirectories
for root, dirs, files in os.walk(image_directory):
    for filename in files:
        # check if the file is a jpg or jpeg
        if filename.lower().endswith('.jpg'):  # png for masks
            # construct the full file path
            file_path = os.path.join(root, filename)
            # add a rectangle to the image
            add_rectangle(file_path)
