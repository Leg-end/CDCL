import os
import random
import math
import cv2


def resize(
        image,
        size,
        method=cv2.INTER_LINEAR):
    return cv2.resize(src=image, dsize=size[::-1], interpolation=method)


def keep_aspect_resize_padding(
        image,
        resize_height,
        resize_width,
        random_seed=None,
        method=cv2.INTER_LINEAR,
        border_value=(0, 0, 0)):
    raw_aspect = float(image.shape[1]) / image.shape[0]
    resize_aspect = float(resize_width) / resize_height
    if raw_aspect > resize_aspect:
        height = math.floor(resize_width / raw_aspect)
        resize_img = resize(image=image, size=(height, resize_width), method=method)
        h = resize_img.shape[0]
        if random_seed:
            random.seed(random_seed)
            padding = random.randint(0, resize_height-h)
        else:
            padding = math.floor((resize_height - h) / 2.0)
        resize_img = cv2.copyMakeBorder(src=resize_img, top=padding, bottom=resize_height - h - padding, left=0,
                                        right=0, borderType=cv2.BORDER_CONSTANT, value=border_value)
    else:
        width = math.floor(raw_aspect * resize_height)
        resize_img = resize(image=image, size=(resize_height, width), method=method)
        w = resize_img.shape[1]
        if random_seed:
            random.seed(random_seed)
            padding = random.randint(0, resize_width-w)
        else:
            padding = math.floor((resize_width - w) / 2.0)
        resize_img = cv2.copyMakeBorder(src=resize_img, top=0, bottom=0, left=padding,
                                        right=resize_width - w - padding,
                                        borderType=cv2.BORDER_CONSTANT, value=border_value)

    return resize_img


def process(input_path):
    im = cv2.imread(input_path)
    im = keep_aspect_resize_padding(im, 224, 224, border_value=(255, 255, 255))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    outputName = input_path.split('.')[0] + ".bin"
    im.tofile(outputName)


if __name__ == '__main__':
    images = os.listdir(r'/')
    for image_name in images:
        if not image_name.endswith("jpg"):
            continue

        print("start to process image {}....".format(image_name))
        process(image_name)
