import os
import sys
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

size_image = (256, 256)


class LSB:

    # convert integer to 8-bit binary
    def int2bin(self, image):
        r, g, b = image
        return (f'{r:08b}', f'{g:08b}', f'{b:08b}')

    # convert 8-bit binary to integer
    def bin2int(self, image):
        r, g, b = image
        return (int(r, 2),  int(g, 2), int(b, 2))

    # define the encryption function
    def encryption(self, original, secret):
        pixel_1 = original.load()
        pixel_2 = secret.load()
        outcome = Image.new(original.mode, original.size)
        pixel_new = outcome.load()

        for i in range(size_image[0]):
            for j in range(size_image[1]):
                r1, g1, b1 = self.int2bin(pixel_1[i, j])
                r2, g2, b2 = self.int2bin(pixel_2[i, j])
                pixel_new[i, j] = self.bin2int((r1[:4] + r2[:4], g1[:4] + g2[:4], b1[:4] + b2[:4]))

        return outcome

    # define the decryption function
    def decryption(self, image):
        pixel_merge = image.load()
        secret = Image.new(image.mode, image.size)
        pixel_secret = secret.load()

        for i in range(size_image[0]):
            for j in range(size_image[1]):
                r, g, b = self.int2bin(pixel_merge[i, j])
                pixel_secret[i, j] = self.bin2int((r[4:] + '0000',  g[4:] + '0000', b[4:] + '0000'))

        return secret


if __name__ == '__main__':

    test_images = []
    for imgnames in os.listdir("./images_test/"):
        test_images.append(Image.open("./images_test/" + imgnames).resize(size_image, Image.ANTIALIAS))

    np.random.shuffle(test_images)

    lsb_implementation = LSB()
    test_original = test_images[0:12]
    test_secret = test_images[12:24]
    test_merge = []
    test_reveal = []
    for i in range(12):
        test_merge.append(lsb_implementation.encryption(test_original[i], test_secret[i]))
        test_reveal.append(lsb_implementation.decryption(test_merge[-1]))

    # Number of secret and cover pairs to show.
    n = 12

    def show_image(img, n_rows, n_col, idx, gray=False, first_row=False, title=None):
        ax = plt.subplot(n_rows, n_col, idx)
        plt.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if first_row:
            plt.title(title)


    plt.figure(figsize=(4, 12))
    for i in range(12):
        n_col = 4

        show_image(test_original[i], n, n_col, i * n_col + 1, first_row=i == 0, title='Cover')

        show_image(test_secret[i], n, n_col, i * n_col + 2, first_row=i == 0, title='Secret')

        show_image(test_merge[i], n, n_col, i * n_col + 3, first_row=i == 0, title='Merge')

        show_image(test_reveal[i], n, n_col, i * n_col + 4, first_row=i == 0, title='Reveal')

    plt.savefig('./result_1.jpg')
    plt.show()
