import os
import cv2
import sys
import pandas as pd 
from tqdm import tqdm
from tensorflow.keras.models import load_model
from numpy.random import randint
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sewar import full_ref
from skimage import metrics

train_dir = "images"
for img in os.listdir(train_dir + "/faces_original"):
    img_array = cv2.imread(train_dir + "/faces_original/" + img)
    
    lr_img_array = cv2.resize(img_array, (256,256))
    cv2.imwrite(train_dir + "/faces_original/" + img, lr_img_array)
    
# BICUBIC
for img in os.listdir(train_dir + "/faces_original"):
    img_array = cv2.imread(train_dir + "/faces_original/" + img)
    
    lr_img_array = cv2.resize(img_array, (256,256))
    cv2.imwrite(train_dir + "/faces_bicubic/" + img, lr_img_array)
    
    
# Loading model
generator = load_model('model.h5', compile=False)
# test images on trained model
for img in tqdm(os.listdir("data/faces_original")):
    img_lr = cv2.imread("data/faces_original/" + img)
    img_lr = cv2.resize(img_lr, (64,64))
    img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
    img_lr = img_lr/255
    img_lr = np.expand_dims(img_lr, axis=0)
    gen_image = generator.predict(img_lr) 
    for i, image in enumerate(gen_image, 1):
        tf.keras.preprocessing.image.save_img('data/faces_generated/'+img, image)


    
psnr_list=[]
mse_list=[]
ssim_list=[]
    
for i in tqdm(range(1000)):
    psnr_img=metrics.peak_signal_noise_ratio(original_images[i], generated_images[i])
    psnr_list.append(psnr_img)
    
    mse_img = metrics.mean_squared_error(original_images[i], generated_images[i])
    mse_list.append(mse_img)
    
    ssim_img = metrics.structural_similarity(original_images[i], generated_images[i], channel_axis=3, multichannel=True)
    ssim_list.append(ssim_img)
    

# plot loss during training that was saved in log file
plt.figure(figsize=(8, 5), dpi=80)
plt.plot(gloss.index.values, gloss["g_loss"])
plt.plot(df.index.values, df["average"])
plt.xlabel('epochs', fontsize=15)
plt.ylabel('g loss', fontsize=15)
plt.show()

plt.figure(figsize=(8, 5), dpi=80)
plt.plot(df.index.values, df["average"], color='orange')
plt.xlabel('epochs', fontsize=15)
plt.ylabel('d_loss', fontsize=15)
plt.show()


# Plot PSNR
plt.figure(figsize=(8, 4), dpi=80)
plt.plot(df.index.values, df["psnr"])
plt.xlabel('test images', fontsize=15)
plt.ylabel('PSNR', fontsize=15)
plt.show()


#Plot MSE
plt.figure(figsize=(15, 7), dpi=80)
plt.plot(df.index.values, df["mse"])
plt.xlabel('test images', fontsize=20)
plt.tick_params(axis='both', labelsize=20)
plt.ylabel('MSE', fontsize=20)
plt.show()



# Compare generated and original images with their evaluation metrics
a = randint(30000, 31000)
print(a)
img1 = cv2.imread('small_data/generated/' + str(a) + '.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('small_data/hr/' + str(a) + '.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img3 = cv2.imread('small_data/bicubic/' + str(a) + '.png')
img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)

loss1 = 'MSE='+str(round(mse1, 2))+', PSNR='+str(round(psnr1, 2))+', SSIM='+str(round(ssim1, 2))

plt.figure(figsize=(16, 8))
plt.subplot(231)
plt.title('Bicubic Interpolation\n MSE='+str(round(mse2, 2))+', PSNR='+str(round(psnr2, 2))+', SSIM='+str(round(ssim2, 2)), fontsize=15)
plt.imshow(img3)
plt.subplot(232)
plt.title('Superresolution\n MSE='+str(round(mse1, 2))+', PSNR='+str(round(psnr1, 2))+', SSIM='+str(round(ssim1, 2)), fontsize=15)
plt.imshow(img1)
plt.subplot(233)
plt.title('Orig. HR image\n MSE=0, PSNR=inf, SSIM=1', fontsize=15)
plt.imshow(img2)
