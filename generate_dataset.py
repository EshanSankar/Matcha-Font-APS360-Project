from PIL import Image, ImageFont, ImageDraw
import random
import os

# Copied fonts directory from C:\Windows\Fonts to the project directory.
FONT_DIRECTORY = "fonts"
OUTPUT_DIRECTORY = "fonts_image_dataset"
OUTPUT_DIRECTORY2 = "fonts_image_dataset_no_rotations"
TEXT_FILE = "1984.txt"
IMAGE_SIZE = 224
NUM_FONTS = 42
LINE_SPACING = 20
NUM_IMAGES_PER_FONT = 4000
ROTATION = 10
FONT_SIZE = (15, 15)
NUM_CHARS = (3,100)
UNIQUE_FONTS = True
TWO_DATSETS = True
REMOVED_FONTS = [('fonts\\arialbd.ttf', 'arialbd'), ('fonts\\arialbi.ttf', 'arialbi'), ('fonts\\ariali.ttf', 'ariali'), ('fonts\\calibrib.ttf', 'calibrib'), ('fonts\\calibrii.ttf', 'calibrii'), ('fonts\\calibril.ttf', 'calibril'), ('fonts\\calibrili.ttf', 'calibrili'), ('fonts\\calibriz.ttf', 'calibriz'), ('fonts\\calibrili.ttf', 'calibrili'), ('fonts\\Candarab.ttf', 'Candarab'), ('fonts\\Candarai.ttf', 'Candarai'), ('fonts\\Candaral.ttf', 'Candaral'), ('fonts\\Candarali.ttf', 'Candarali'), ('fonts\\Candaraz.ttf', 'Candaraz'), ('fonts\\Candarali.ttf', 'Candarali'), ('fonts\\comicbd.ttf', 'comicbd'), ('fonts\\comici.ttf', 'comici'), ('fonts\\comicz.ttf', 'comicz'), ('fonts\\consolab.ttf', 'consolab'), ('fonts\\consolai.ttf', 'consolai'), ('fonts\\consolaz.ttf', 'consolaz'), ('fonts\\constanb.ttf', 'constanb'), ('fonts\\constani.ttf', 'constani'), ('fonts\\constanz.ttf', 'constanz'), ('fonts\\corbelb.ttf', 'corbelb'), ('fonts\\corbeli.ttf', 'corbeli'), ('fonts\\corbell.ttf', 'corbell'), ('fonts\\corbelli.ttf', 'corbelli'), ('fonts\\corbelz.ttf', 'corbelz'), ('fonts\\corbelli.ttf', 'corbelli'), ('fonts\\courbd.ttf', 'courbd'), ('fonts\\courbi.ttf', 'courbi'), ('fonts\\couri.ttf', 'couri'), ('fonts\\Dengb.ttf', 'Dengb'), ('fonts\\Dengl.ttf', 'Dengl'), ('fonts\\ebrimabd.ttf', 'ebrimabd'), ('fonts\\framdit.ttf', 'framdit'), ('fonts\\gadugib.ttf', 'gadugib'), ('fonts\\georgiab.ttf', 'georgiab'), ('fonts\\georgiai.ttf', 'georgiai'), ('fonts\\georgiaz.ttf', 'georgiaz'), ('fonts\\HPSimplified_BdIt.ttf', 'HPSimplified_BdIt'), ('fonts\\HPSimplified_It.ttf', 'HPSimplified_It'), ('fonts\\HPSimplified_Lt.ttf', 'HPSimplified_Lt'), ('fonts\\HPSimplified_LtIt.ttf', 'HPSimplified_LtIt'), ('fonts\\HPSimplified_Rg.ttf', 'HPSimplified_Rg'), ('fonts\\HPSimplified_LtIt.ttf', 'HPSimplified_LtIt'), 
('fonts\\malgunbd.ttf', 'malgunbd'), ('fonts\\malgunsl.ttf', 'malgunsl'), ('fonts\\mmrtextb.ttf', 'mmrtextb'), ('fonts\\NirmalaB.ttf', 'NirmalaB'), ('fonts\\NirmalaS.ttf', 'NirmalaS'), ('fonts\\ntailub.ttf', 'ntailub'), ('fonts\\palab.ttf', 'palab'), ('fonts\\palabi.ttf', 'palabi'), ('fonts\\palai.ttf', 'palai'), ('fonts\\palabi.ttf', 'palabi'), ('fonts\\phagspab.ttf', 'phagspab'), ('fonts\\segoeprb.ttf', 'segoeprb'), ('fonts\\segoescb.ttf', 'segoescb'), ('fonts\\segoeuib.ttf', 'segoeuib'), ('fonts\\segoeuii.ttf', 'segoeuii'), ('fonts\\segoeuil.ttf', 'segoeuil'), ('fonts\\segoeuisl.ttf', 'segoeuisl'), ('fonts\\segoeuiz.ttf', 'segoeuiz'), ('fonts\\seguibli.ttf', 'seguibli'), ('fonts\\seguisbi.ttf', 'seguisbi'), ('fonts\\SitkaVF-Italic.ttf', 'SitkaVF-Italic'), ('fonts\\tahomabd.ttf', 'tahomabd'), ('fonts\\taileb.ttf', 'taileb'), ('fonts\\timesbd.ttf', 'timesbd'), ('fonts\\timesbi.ttf', 'timesbi'), ('fonts\\timesi.ttf', 'timesi'), ('fonts\\trebucbd.ttf', 'trebucbd'), ('fonts\\trebucbi.ttf', 'trebucbi'), ('fonts\\trebucit.ttf', 'trebucit'), ('fonts\\verdanab.ttf', 'verdanab'), ('fonts\\verdanai.ttf', 'verdanai'), ('fonts\\verdanaz.ttf', 'verdanaz'), ('fonts\\arialbd.ttf', 'arialbd'), ('fonts\\arialbi.ttf', 'arialbi'), ('fonts\\ariali.ttf', 'ariali'), ('fonts\\calibrib.ttf', 'calibrib'), ('fonts\\calibrii.ttf', 'calibrii'), ('fonts\\calibril.ttf', 'calibril'), ('fonts\\calibrili.ttf', 'calibrili'), ('fonts\\calibriz.ttf', 'calibriz'), ('fonts\\calibrili.ttf', 'calibrili'), ('fonts\\Candarab.ttf', 'Candarab'), ('fonts\\Candarai.ttf', 'Candarai'), ('fonts\\Candaral.ttf', 'Candaral'), ('fonts\\Candarali.ttf', 'Candarali'), ('fonts\\Candaraz.ttf', 'Candaraz'), ('fonts\\Candarali.ttf', 'Candarali'), ('fonts\\comicbd.ttf', 'comicbd'), ('fonts\\comici.ttf', 'comici'), ('fonts\\comicz.ttf', 'comicz'), ('fonts\\consolab.ttf', 'consolab'), ('fonts\\consolai.ttf', 'consolai'), ('fonts\\consolaz.ttf', 'consolaz'), ('fonts\\constanb.ttf', 'constanb'), ('fonts\\constani.ttf', 'constani'), ('fonts\\constanz.ttf', 'constanz'), ('fonts\\corbelb.ttf', 'corbelb'), ('fonts\\corbeli.ttf', 'corbeli'), ('fonts\\corbell.ttf', 'corbell'), ('fonts\\corbelli.ttf', 'corbelli'), ('fonts\\corbelz.ttf', 'corbelz'), ('fonts\\corbelli.ttf', 'corbelli'), ('fonts\\courbd.ttf', 'courbd'), ('fonts\\courbi.ttf', 'courbi'), ('fonts\\couri.ttf', 'couri'), ('fonts\\Dengb.ttf', 'Dengb'), 
('fonts\\Dengl.ttf', 'Dengl'), ('fonts\\ebrimabd.ttf', 'ebrimabd'), ('fonts\\framdit.ttf', 'framdit'), ('fonts\\gadugib.ttf', 'gadugib'), ('fonts\\georgiab.ttf', 'georgiab'), ('fonts\\georgiai.ttf', 'georgiai'), ('fonts\\georgiaz.ttf', 'georgiaz'), ('fonts\\HPSimplified_BdIt.ttf', 'HPSimplified_BdIt'), ('fonts\\HPSimplified_It.ttf', 'HPSimplified_It'), ('fonts\\HPSimplified_Lt.ttf', 'HPSimplified_Lt'), ('fonts\\HPSimplified_LtIt.ttf', 'HPSimplified_LtIt'), ('fonts\\HPSimplified_Rg.ttf', 'HPSimplified_Rg'), ('fonts\\HPSimplified_LtIt.ttf', 'HPSimplified_LtIt'), ('fonts\\malgunbd.ttf', 'malgunbd'), ('fonts\\malgunsl.ttf', 'malgunsl'), ('fonts\\mmrtextb.ttf', 'mmrtextb'), ('fonts\\NirmalaB.ttf', 'NirmalaB'), ('fonts\\NirmalaS.ttf', 'NirmalaS'), ('fonts\\ntailub.ttf', 'ntailub'), ('fonts\\palab.ttf', 'palab'), ('fonts\\palabi.ttf', 'palabi'), ('fonts\\palai.ttf', 'palai'), ('fonts\\palabi.ttf', 'palabi'), ('fonts\\phagspab.ttf', 'phagspab'), ('fonts\\segoeprb.ttf', 'segoeprb'), ('fonts\\segoescb.ttf', 'segoescb'), ('fonts\\segoeuib.ttf', 'segoeuib'), ('fonts\\segoeuii.ttf', 'segoeuii'), ('fonts\\segoeuil.ttf', 'segoeuil'), ('fonts\\segoeuisl.ttf', 'segoeuisl'), ('fonts\\segoeuiz.ttf', 'segoeuiz'), ('fonts\\seguibli.ttf', 'seguibli'), ('fonts\\seguisbi.ttf', 'seguisbi'), ('fonts\\SitkaVF-Italic.ttf', 'SitkaVF-Italic'), ('fonts\\tahomabd.ttf', 'tahomabd'), ('fonts\\taileb.ttf', 'taileb'), ('fonts\\timesbd.ttf', 'timesbd'), ('fonts\\timesbi.ttf', 'timesbi'), ('fonts\\timesi.ttf', 'timesi'), ('fonts\\trebucbd.ttf', 'trebucbd'), ('fonts\\trebucbi.ttf', 'trebucbi'), ('fonts\\trebucit.ttf', 'trebucit'), ('fonts\\verdanab.ttf', 'verdanab'), ('fonts\\verdanai.ttf', 'verdanai'), ('fonts\\verdanaz.ttf', 'verdanaz'),
('fonts\\cambriai.ttf', 'cambriai'), ('fonts\\cambriaz.ttf', 'cambriaz'), ('fonts\\hpsimplifiedhans-light.ttf', 'hpsimplifiedhans-light'), ('fonts\\hpsimplifiedhans-regular.ttf', 'hpsimplifiedhans-regular'), ('fonts\\hpsimplifiedjpan-light.ttf', 'hpsimplifiedjpan-light'), ('fonts\\hpsimplifiedjpan-regular.ttf', 'hpsimplifiedjpan-regular'), ('fonts\\holomdl2.ttf', 'holomdl2'), ('fonts\\LeelaUIb.ttf', 'LeelaUIb'), ('fonts\\LeelUIsl.ttf', 'LeelUIsl'), ('fonts\\Montserrat-Medium.ttf', 'Montserrat-Medium'), ('fonts\\marlett.ttf', 'marlett'), ('fonts\\SansSerifCollection.ttf', 'SansSerifCollection'), ('fonts\\segmdl2.ttf', 'segmdl2'), ('fonts\\SegoeIcons.ttf', 'SegoeIcons'), ('fonts\\seguibl.ttf', 'seguibl'), ('fonts\\seguihis.ttf', 'seguihis'), ('fonts\\seguisli.ttf', 'seguisli'), ('fonts\\seguisb.ttf', 'seguisb'), ('fonts\\seguisym.ttf', 'seguisym'), ('fonts\\SegUIVar.ttf', 'SegUIVar'), ('fonts\\simhei.ttf', 'simhei'), ('fonts\\simsunb.ttf', 'simsunb'), ('fonts\\webdings.ttf', 'webdings'), ('fonts\\wingding.ttf', 'wingding'),
('fonts\\gadugi.ttf', 'gadugi'), ('fonts\\LeelawUI.ttf', 'LeelawUI'), ('fonts\\mmrtext.ttf', 'mmrtext'), ('fonts\\ntailu.ttf', 'ntailu'), ('fonts\\segoeui.ttf', 'segoeui'), ('fonts\\seguiemj.ttf', 'seguiemj'), ('fonts\\Nirmala.ttf', 'Nirmala'), ('fonts\\ariblk.ttf', 'ariblk'), ('fonts\\taile.ttf', 'taile')]

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
            
    fonts_tobe_removed = REMOVED_FONTS
    # if UNIQUE_FONTS:
    #     for x in range(len(fonts)):
    #         for y in range(len(fonts)):
    #             if fonts[y] == fonts[x]:
    #                 continue
    #             if fonts[y][1].startswith(fonts[x][1]):
    #                 fonts_tobe_removed.append(fonts[y])
    #                 print(fonts_tobe_removed)
                    
    for f in range(len(fonts_tobe_removed)):
        if fonts_tobe_removed[f] in fonts:
            fonts.remove(fonts_tobe_removed[f])
            
    print(fonts)
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
    with_rotation = True
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
        rotation=0
        if with_rotation:
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
        
    if TWO_DATSETS:
        if not os.path.exists(OUTPUT_DIRECTORY2):
            os.makedirs(OUTPUT_DIRECTORY2)

    # Generate images for each font
    for i in range(NUM_FONTS):

        font_filepath, font_name = fonts[i]
        
        # Create font directory if does not exist
        if not os.path.exists(f"{OUTPUT_DIRECTORY}/{font_name}"):
            os.makedirs(f"{OUTPUT_DIRECTORY}/{font_name}")

        if TWO_DATSETS:
            if not os.path.exists(f"{OUTPUT_DIRECTORY2}/{font_name}"):
                os.makedirs(f"{OUTPUT_DIRECTORY2}/{font_name}")
        
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
            
            if TWO_DATSETS:
                img2 = text_to_image(
                text[text_index:text_index+num_chars],
                font_filepath,
                font_size,
                start_position=start_position,
                line_spacing=line_spacing,
                with_rotation=False
                )
                img2.save(f"{OUTPUT_DIRECTORY2}/{font_name}/image_{j}.png")

            text_index += num_chars
            
            # Save image
            img.save(f"{OUTPUT_DIRECTORY}/{font_name}/image_{j}.png")

        print(f"Generated images for font {font_name}")

if __name__ == "__main__":

    print(len(get_font_list()))
    
    generate_dataset()