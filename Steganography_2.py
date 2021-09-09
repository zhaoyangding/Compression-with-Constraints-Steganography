import tensorflow as tf
from tensorflow.keras import layers, models, backend
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from tqdm import *
import random

size_image = (64, 64)


# define loss function for reveal network
def rev_loss(x, y):
    return backend.sum(backend.square(x - y))


# define loss function for full model
def full_loss(x, y):
    x_ori, x_sec = x[..., 0:3], x[..., 3:6]
    y_ori, y_sec = y[..., 0:3], y[..., 3:6]
    return backend.sum(backend.square(x_ori - y_ori)) + backend.sum(backend.square(x_sec - y_sec))


# load images for training set
train_set = []
for imgnames in os.listdir("./images/"):
    train_set.append(np.array(Image.open("./images/" + imgnames).resize(size_image, Image.ANTIALIAS)))

train_set = np.array(train_set)[:400] / 255

# shuffle the data
ran_train = np.random.permutation(train_set.shape[0])
train_set_random = train_set[ran_train, :, :]

# make the model
# preparation network
input_original = layers.Input(shape=(64, 64, 3))
input_secret = layers.Input(shape=(64, 64, 3))
x3 = layers.Conv2D(50, (3,3), activation='relu', padding='same')(input_original)
x4 = layers.Conv2D(10, (4,4), activation='relu', padding='same')(input_original)
x5 = layers.Conv2D(5, (5,5), activation='relu', padding='same')(input_original)
x = layers.concatenate([x3, x4, x5])
x3 = layers.Conv2D(50, (3,3), activation='relu', padding='same')(x)
x4 = layers.Conv2D(10, (4,4), activation='relu', padding='same')(x)
x5 = layers.Conv2D(5, (5,5), activation='relu', padding='same')(x)
x = layers.concatenate([x3, x4, x5])

# hiding network
x = layers.concatenate([input_secret, x])
x3 = layers.Conv2D(50, (3,3), activation='relu', padding='same')(x)
x4 = layers.Conv2D(10, (4,4), activation='relu', padding='same')(x)
x5 = layers.Conv2D(5, (5,5), activation='relu', padding='same')(x)
x = layers.concatenate([x3, x4, x5])
x3 = layers.Conv2D(50, (3,3), activation='relu', padding='same')(x)
x4 = layers.Conv2D(10, (4,4), activation='relu', padding='same')(x)
x5 = layers.Conv2D(5, (5,5), activation='relu', padding='same')(x)
x = layers.concatenate([x3, x4, x5])
x3 = layers.Conv2D(50, (3,3), activation='relu', padding='same')(x)
x4 = layers.Conv2D(10, (4,4), activation='relu', padding='same')(x)
x5 = layers.Conv2D(5, (5,5), activation='relu', padding='same')(x)
x = layers.concatenate([x3, x4, x5])
x3 = layers.Conv2D(50, (3,3), activation='relu', padding='same')(x)
x4 = layers.Conv2D(10, (4,4), activation='relu', padding='same')(x)
x5 = layers.Conv2D(5, (5,5), activation='relu', padding='same')(x)
x = layers.concatenate([x3, x4, x5])
x3 = layers.Conv2D(50, (3,3), activation='relu', padding='same')(x)
x4 = layers.Conv2D(10, (4,4), activation='relu', padding='same')(x)
x5 = layers.Conv2D(5, (5,5), activation='relu', padding='same')(x)
x = layers.concatenate([x3, x4, x5])

# encryption model
outcome = layers.Conv2D(3, (3,3), activation='relu', padding='same')(x)
encrypt = models.Model(inputs=[input_original, input_secret], outputs=outcome, name='Encrypt')

# reveal network
input_merge = layers.Input(shape=(64, 64, 3))
input_with_noise = layers.GaussianNoise(0.01)(input_merge)
x3 = layers.Conv2D(50, (3,3), activation='relu', padding='same')(input_with_noise)
x4 = layers.Conv2D(10, (4,4), activation='relu', padding='same')(input_with_noise)
x5 = layers.Conv2D(5, (5,5), activation='relu', padding='same')(input_with_noise)
x = layers.concatenate([x3, x4, x5])
x3 = layers.Conv2D(50, (3,3), activation='relu', padding='same')(x)
x4 = layers.Conv2D(10, (4,4), activation='relu', padding='same')(x)
x5 = layers.Conv2D(5, (5,5), activation='relu', padding='same')(x)
x = layers.concatenate([x3, x4, x5])
x3 = layers.Conv2D(50, (3,3), activation='relu', padding='same')(x)
x4 = layers.Conv2D(10, (4,4), activation='relu', padding='same')(x)
x5 = layers.Conv2D(5, (5,5), activation='relu', padding='same')(x)
x = layers.concatenate([x3, x4, x5])
x3 = layers.Conv2D(50, (3,3), activation='relu', padding='same')(x)
x4 = layers.Conv2D(10, (4,4), activation='relu', padding='same')(x)
x5 = layers.Conv2D(5, (5,5), activation='relu', padding='same')(x)
x = layers.concatenate([x3, x4, x5])
x3 = layers.Conv2D(50, (3,3), activation='relu', padding='same')(x)
x4 = layers.Conv2D(10, (4,4), activation='relu', padding='same')(x)
x5 = layers.Conv2D(5, (5,5), activation='relu', padding='same')(x)
x = layers.concatenate([x3, x4, x5])

# decryption model
reveal = layers.Conv2D(3, (3,3), activation='relu', padding='same')(x)
decrypt = models.Model(inputs=input_merge, outputs=reveal, name='Decrypt')
decrypt.compile(optimizer='adam', loss=rev_loss)
decrypt.trainable = False

# full model
output_merge = encrypt([input_original, input_secret])
output_reveal = decrypt(output_merge)
full_model = models.Model(inputs=[input_original, input_secret], outputs=layers.concatenate([output_merge, output_reveal]))
full_model.compile(optimizer='adam', loss=full_loss)

# training process
train_original = train_set_random[0:200, ...]
train_secret = train_set_random[200:400, ...]

NB_EPOCHS = 20
BATCH_SIZE = 32

full_model.load_weights('model.hdf5')

m = train_original.shape[0]
loss_history = []
for epoch in range(NB_EPOCHS):
    np.random.shuffle(train_original)
    np.random.shuffle(train_secret)

    t = tqdm(range(0, train_original.shape[0], BATCH_SIZE), mininterval=0)
    ae_loss = []
    rev_loss = []
    for idx in t:
        batch_S = train_original[idx:min(idx + BATCH_SIZE, m)]
        batch_C = train_secret[idx:min(idx + BATCH_SIZE, m)]

        C_prime = encrypt.predict([batch_S, batch_C])

        ae_loss.append(full_model.train_on_batch(x=[batch_S, batch_C],
                                                        y=np.concatenate((batch_S, batch_C), axis=3)))
        rev_loss.append(decrypt.train_on_batch(x=C_prime,
                                                    y=batch_S))

        backend.set_value(full_model.optimizer.lr, 0.001)
        backend.set_value(decrypt.optimizer.lr, 0.001)

        t.set_description('Epoch {} | Batch: {:3} of {}. Loss AE {:10.2f} | Loss Rev {:10.2f}'.format(epoch + 1, idx, m,
                                                                                                      np.mean(ae_loss),
                                                                                                      np.mean(
                                                                                                          rev_loss)))
    loss_history.append(np.mean(ae_loss))

# show result of loss versus epochs
plt.plot(loss_history)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

full_model.save_weights('model.hdf5')

# load test images
test_images = []
for imgnames in os.listdir("./images_test/"):
    test_images.append(np.array(Image.open("./images_test/" + imgnames).resize(size_image, Image.ANTIALIAS)))

test_images = np.array(test_images[:]) / 255

np.random.shuffle(test_images)

test_original = test_images[0:12]
test_secret = test_images[12:24]
test_merge = []
test_reveal = []
test_result = full_model.predict([test_original, test_secret])
for i in range(12):
    test_merge.append(test_result[i, :, :, 0:3])
    test_reveal.append(test_result[i, :, :, 3:6])

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

plt.show()
