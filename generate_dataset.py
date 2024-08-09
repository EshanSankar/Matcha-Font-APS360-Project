from PIL import Image, ImageFont, ImageDraw
import random
import os

# Copied fonts directory from C:\Windows\Fonts to the project directory.
FONT_DIRECTORY = "fonts"
OUTPUT_DIRECTORY = "fonts_image_dataset"
TEXT_FILE = "1984.txt"
IMAGE_SIZE = 224
NUM_FONTS = 10
LINE_SPACING = 20
NUM_IMAGES_PER_FONT = 1000
ROTATION = 0
FONT_SIZE = (15,15)
NUM_CHARS = (3,100)

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
    """
    Draw rotated text on the image

    Args:
        image: Image
        font: ImageFont
        text: str
        angle: int
        x: int
        y: int

    Returns:
        width: int
        height: int
    """
    txt = Image.new(image.mode, getsize(font, text))

    draw = ImageDraw.Draw(txt)

    draw.text((0, 0), text, font=font, fill="black")

    txt = txt.rotate(angle, expand=1)

    image.paste(txt, (x, y), txt)

    return txt.width, txt.height


def text_to_image(
    text,
    font_filepath,
    font_size,
    start_position=[0, 0],
    line_spacing=0,
):
    """
    Convert text to image
    
    Args:
        text: str
        font_filepath: str
        font_size: int
        start_position: list
        line_spacing: int
        rotation: int
    """
    font = ImageFont.truetype(font_filepath, size=font_size)

    # Create new image with white background
    img = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), "white")
    
     # determine maximum character height (ascent)
    ascent, descent = font.getmetrics()    
    
    # Calculate the start point (also ensure text does not go offscreen)
    start_point = [int(start_position[0]*(IMAGE_SIZE-ascent)), int(start_position[1]*(IMAGE_SIZE-ascent))]
    # Create variable to store the current point to draw the text
    draw_point = [start_point[0], start_point[1]]

   
    
    # Draw each character individually
    for c in text:

        # If the character is offscreen, move to the next line
        if draw_point[0] > IMAGE_SIZE:
            draw_point[0] = start_point[0]
            draw_point[1] += line_spacing + ascent
            continue
        
        # Randomize Rotation
        rotation = random.randint(-ROTATION, ROTATION)
        
        # Draw the character
        char_width, char_height = draw_rotated_text(
            img, font, c, rotation, draw_point[0], draw_point[1]
        )
        # Move the draw point to the right
        draw_point[0] += char_width

    return img

def generate_dataset():
    """
    Generate dataset from the text file
    """
   
    # Read the text file
    with open(TEXT_FILE, "r") as file:
        text = file.read()

    # Get the list of fonts
    fonts = get_font_list()
    
    # Create output directory if does not exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    # Generate images for each font
    for i in range(NUM_FONTS):

        font_filepath, font_name = fonts[i]
        
        # Create font directory if does not exist
        if not os.path.exists(f"{OUTPUT_DIRECTORY}/{font_name}"):
            os.makedirs(f"{OUTPUT_DIRECTORY}/{font_name}")

        
        text_index = 0
        
        # Generate random values
        for j in range(NUM_IMAGES_PER_FONT):
            
            random.seed(j)
            
            font_size = random.randint(FONT_SIZE[0], FONT_SIZE[1])
            line_spacing = random.randint(0, LINE_SPACING)
            start_position = [random.triangular(0,1,0), random.triangular(0,1,0)]
            num_chars = random.randint(NUM_CHARS[0], NUM_CHARS[1])

            # Generate image
            img = text_to_image(
                text[text_index:text_index+num_chars],
                font_filepath,
                font_size,
                start_position=start_position,
                line_spacing=line_spacing,
            )

            text_index += num_chars
            
            # Save image
            img.save(f"{OUTPUT_DIRECTORY}/{font_name}/image_{j}.png")

        print(f"Generated images for font {font_name}")

if __name__ == "__main__":

    print(len(get_font_list()))
    
    generate_dataset()
