from PIL import Image, ImageFont, ImageDraw
import numpy as np
import os

# Copied fonts directory from C:\Windows\Fonts to the project directory.
FONT_DIRECTORY = "Fonts"
TEXT_FILE = "1984.txt"
IMAGE_SIZE = 224

def get_font_list():
    """
    Get a list of all the fonts in the FONT_DIRECTORY

    Returns:
        fonts: list of tuples (font_filepath, font_name)
    """
    fonts = []

    for file in os.listdir(FONT_DIRECTORY):

        if file.endswith(".ttf"):

            font_name = file.split(".")[0]

            fonts.append((FONT_DIRECTORY + "\\" + file, font_name))

    return fonts


def getsize(font, text):
    """
    Get the size of the text with the given font
    
    Args:
        font: ImageFont
        text: str
        
    Returns:
        width: int
        height: int
    """
    left, top, right, bottom = font.getbbox(text)
    return right + left, bottom + top


def draw_rotated_text(image, font, text, angle, x, y):
    txt = Image.new(image.mode, getsize(font, text))
    d = ImageDraw.Draw(txt)
    d.text((0, 0), text, font=font, fill="black")
    txt = txt.rotate(angle, expand=1)
    image.paste(txt, (x, y), txt)
    return txt.width, txt.height


def text_to_image(
    text,
    font_filepath,
    font_size,
    start_point=[0, 0],
    line_spacing=20,
):

    font = ImageFont.truetype(font_filepath, size=font_size)

    img = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), "white")
    draw_point = [start_point[0], start_point[1]]

    for c in text:

        if draw_point[0] > IMAGE_SIZE:
            draw_point[0] = start_point[0]
            draw_point[1] += line_spacing
            continue

        txt_width, txt_height = draw_rotated_text(
            img, font, c, 0, draw_point[0], draw_point[1]
        )
        draw_point[0] += txt_width

    return img


if __name__ == "__main__":

    fonts = get_font_list()

    text = "Hello Worldasdf\nasdfasdfasdfasdfasdfasdfasdfasdfs"
    font_filepath = fonts[0][0]
    font_size = 20
    color = (255, 255, 255)
    img = text_to_image(text, font_filepath, font_size)
    img.save("output.png")
    print("Image saved as output.png")
