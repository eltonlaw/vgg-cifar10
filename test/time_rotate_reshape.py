import numpy as np
import argparse


def rotate_reshape1(images, n_data):
    new_images = []
    for i, img in enumerate(images):
        img = np.reshape(img, n_data, order="F")
        img = np.rot90(img, k=3)
        new_images.append(img)
    return new_images


def rotate_reshape2(images, n_data):
    def single_rotate_reshape(img):
        img = np.reshape(img, n_data, order="F")
        img = np.rot90(img, k=3)
        return img
    new_images = list(map(lambda img: single_rotate_reshape(img), images))
    return new_images


if __name__ == "__main__":
    """
    RUN THE FOLLOWING:
        $ time python3 time_rotate_reshape.py
        $ time python3 time_rotate_reshape.py --fn=1
        $ time python3 time_rotate_reshape.py --fn=2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', help='foo help')
    args = parser.parse_args()
    flag = int(args.fn)
    print("FLAG:", flag)

    images = np.ones((10000, 3072))
    if flag == 1:
        rotate_reshape1(images, [32, 32, 3])
    if flag == 2:
        rotate_reshape2(images, [32, 32, 3])
