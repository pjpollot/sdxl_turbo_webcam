from PIL import Image

import os
import numpy as np
import cv2

from backgroundremover.bg import remove

def square_resize(image: Image, resolution: int) -> Image:
    image_array = np.array(image)
    height, width, _ = image_array.shape
    if width > height:
        dw = (width - height) // 2
        left, top, right, bottom = (
            dw, 0, width-dw, height,
        )
    elif width < height:
        dh = (height - width) / 2
        left, top, right, bottom = (
            0, dh, width, height-dh,
        )
    else:
        left, top, right, bottom = (
            0, 0, width, height,
        )
    image_array = image_array[top:bottom, left:right, :]
    image_array = cv2.resize(image_array, (resolution, resolution))
    return Image.fromarray(image_array)


_cache_folder = os.path.join(os.path.dirname(__file__), "cache")
_img_path = os.path.join(_cache_folder, "in.jpg")
_out_img_path = os.path.join(_cache_folder, "out.jpg")

def remove_background(image: Image) -> Image:
    image.save(_img_path)
    with open(_img_path, "rb") as f:
        data = f.read()
    out_data = remove(data)
    with open(_out_img_path, "wb") as f:
        f.write(out_data)
    out_image = Image.open(_out_img_path)
    for path in [_img_path, _out_img_path]:
        os.remove(path)
    return out_image


def canny_transform(image: Image) -> Image:
    image_array = np.array(image)
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    image_array = cv2.Canny(image_array, 100, 200)
    return Image.fromarray(image_array)
