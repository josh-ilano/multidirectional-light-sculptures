from PIL import Image
import numpy as np
import os

def load_binary_image(path: str, size=(128, 128), threshold=128, invert=False) -> np.ndarray:
    """
    Function which loads an image and converts it to a binary silhouette mask used as the reference silhouettes
    represented as a numpy matrix
    """

    # load and resize the image to the world size
    img = Image.open(path).convert("RGBA").resize(size)

    # remove possible transparent backgrounds
    white = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white, img).convert("L")

    # convert the image into numpy array
    arr = np.array(img)

    # dark pixels = True, light pizels = False
    mask = arr < threshold
    if invert:
        mask = ~mask

    # return the boolean matrix
    return mask.astype(bool)

def save_mask(mask, path):
    """
    Function to convert a boolean matrix back into an image. Used for debugging and testing.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    img.save(path)