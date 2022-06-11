
"""
Original SRGAN paper: https://arxiv.org/abs/1609.04802

References for code: 
    1) https://github.com/bnsreenu/python_for_microscopists
    2) https://blog.paperspace.com/super-resolution-generative-adversarial-networks
    3) K. Ahirwar, Generative Adversarial Networks Projects: Build next-generation generative models using Tensorflow and keras. Packt Publishing, 2019.
    
Dataset FFHQ: https://github.com/NVlabs/ffhq-dataset

This code is used for the purposes of the graduation project whose topic is: ENHANCING FACE IMAGES FROM SURVEILLANCE CAMERAS USING GENERATIVE ADVERSARIAL NETWORKS


"""



import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, PReLU, BatchNormalization, Flatten
from tensorflow.keras.layers import UpSampling2D, LeakyReLU, Dense, Input, Add
from tensorflow.keras.applications import VGG19


# Residual block
# x is input
# k3n64s1
def residual_block(x):
    residual_model = Conv2D(64, (3,3), padding = "same")(x)
    residual_model = BatchNormalization(momentum = 0.8)(residual_model)
    residual_model = PReLU(shared_axes=[1, 2])(residual_model)
    
    residual_model = Conv2D(64, (3,3), padding = "same")(residual_model)
    residual_model = BatchNormalization(momentum = 0.8)(residual_model)
    residual_model = Add()([residual_model, x])
    return residual_model


# Upscale block
# x is input
# k3n256s1
def upscale_block(x):
    upscale_model = Conv2D(256, (3,3), padding = "same")(x)
    upscale_model = UpSampling2D(size = 2)(upscale_model)
    upscale_model = PReLU(shared_axes = [1,2])(upscale_model)
    return upscale_model


# Generator block
# res_copy is skip connection
# number of res blocks --> 16
# number of upscale blocks --> 2
# k9n64s1
def generator(input_layer):
    residual_blocks = 16
    upscale_blocks = 2
    
    #First conv layer
    res = Conv2D(64, (9,9), padding = "same")(input_layer)
    res = PReLU(shared_axes = [1,2])(res)
    
    # skip connection
    res_copy = res
    
    # 16 residual blocks
    for i in range(residual_blocks):
        res = residual_block(res)
    
    res = Conv2D(64, (3,3), padding = "same")(res)
    res = BatchNormalization(momentum = 0.8)(res)
    
    # Elementwise sum
    res = Add()([res_copy, res])
   
    # 2 pixel shuffle blocks
    for i in range(upscale_blocks):
        res = upscale_block(res)
    
    # Last conv layer
    res = Conv2D(3, (9,9), padding = "same")(res)

    return Model(inputs=input_layer, outputs=res)


# Discriminator block
def discriminator_block(input_layer, filters, strides=1, bn=True):
    disc = Conv2D(filters, (3,3), strides = strides, padding = "same")(input_layer)
    if bn:
        disc = BatchNormalization(momentum = 0.8)(disc)
    disc = LeakyReLU(alpha = 0.2)(disc)
    
    return disc


# Discriminator
def discriminator(input_layer):
    
    d = discriminator_block(input_layer, 64, bn=False)
    d = discriminator_block(d, 64, strides=2)
    
    d = discriminator_block(d, 128)
    d = discriminator_block(d, 128, strides=2)
    
    d = discriminator_block(d, 256)
    d = discriminator_block(d, 256, strides=2)
    
    d = discriminator_block(d, 512)
    d = discriminator_block(d, 512, strides=2)
    
    d = Flatten()(d)
    d = Dense(1024)(d)
    d = LeakyReLU(alpha = 0.2)(d)
    
    d = Dense(1, activation = 'sigmoid')(d)
    return Model(inputs=input_layer, outputs=d)


# Pre-trained VGG19 model 
def vgg_model(hr_shape):
    vgg = VGG19(weights = "imagenet", include_top = False, input_shape = hr_shape)
    return Model(inputs = vgg.input, outputs = vgg.layers[10].output)


# SR model, combining generator and disscriminator
def sr_model(gen_model, disc_model, vgg, lr_input, hr_input):
    gen_img = gen_model(lr_input)
    gen_features = vgg(gen_img)
    disc_model.trainable = False
    validity = disc_model(gen_img)
    return Model(inputs = [lr_input, hr_input], outputs = [validity, gen_features])



# Loading images, n is number of images
n = 70000
lr_list = os.listdir("lr/")[:n]

lr_images = []
for img in lr_list:
    img_lr = cv2.imread("lr/" + img)
    img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
    lr_images.append(img_lr)

hr_list = os.listdir("hr/")[:n]
hr_images = []
for img in hr_list:
    img_hr = cv2.imread("hr/" + img)
    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
    hr_images.append(img_hr)
    
lr_images = np.array(lr_images)
hr_images = np.array(hr_images)

# Converting to range from 0 to 1
lr_images = lr_images/255
hr_images = hr_images/255

#Split to train and test
lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images, 
                                                      test_size=0.2, random_state=27)


# Getting shape of images to provide them as an input in generator and discriminator
hr_shape = (hr_train.shape[1], hr_train.shape[2], hr_train.shape[3])
lr_shape = (lr_train.shape[1], lr_train.shape[2], lr_train.shape[3])

lr_input = Input(shape = lr_shape)
hr_input = Input(shape = hr_shape)

gen = generator(lr_input)

disc = discriminator(hr_input)
disc.compile(loss = "binary_crossentropy", optimizer = "adam", metrics=['accuracy'])

vgg = vgg_model((256, 256, 3))
vgg.trainable = False
    
gan_model = sr_model(gen, disc, vgg, lr_input, hr_input)
    
gan_model.compile(loss=["binary_crossentropy", "mse"], loss_weights=[1e-3, 1], optimizer="adam")


#Batch
batch_size = 1  
train_lr_batches = []
train_hr_batches = []
for it in range(int(hr_train.shape[0] / batch_size)):
    start_idx = it * batch_size
    end_idx = start_idx + batch_size
    train_hr_batches.append(hr_train[start_idx:end_idx])
    train_lr_batches.append(lr_train[start_idx:end_idx])
    
# Training
epochs = 200
for e in range(epochs):

    fake_label = np.zeros((batch_size, 1))
    real_label = np.ones((batch_size, 1))

    #gen and disc losses.
    g_losses = []
    d_losses = []

    for b in tqdm(range(len(train_hr_batches))):
        lr_imgs = train_lr_batches[b]
        hr_imgs = train_hr_batches[b]

        fake_imgs = gen.predict_on_batch(lr_imgs)

        #train the discriminator
        disc.trainable = True
        d_loss_gen = disc.train_on_batch(fake_imgs, fake_label)
        d_loss_real = disc.train_on_batch(hr_imgs, real_label)

        # train the generator
        disc.trainable = False
        d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)

        #VGG
        image_features = vgg.predict(hr_imgs)

        g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features])

        #Append losses
        d_losses.append(d_loss)
        g_losses.append(g_loss)

    g_losses=np.array(g_losses)
    d_losses=np.array(d_losses)

    # average losses for generator and discriminator
    g_loss=np.sum(g_losses, axis = 0) / len(g_losses)
    d_loss=np.sum(d_losses, axis = 0) / len(d_losses)


    print("epoch:", e+1, "g_loss:", g_loss, "d_loss:", d_loss)

    #Saving model
    if (e+1) % 5 == 0:
        gen.save("model.h5")


