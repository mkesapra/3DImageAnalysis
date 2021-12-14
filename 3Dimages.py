#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 21:09:55 2021

@author: kesaprm
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure, io, util,measure
from skimage import img_as_ubyte
import cv2
import segDefinitions

sk_3dimg = io.imread('M2 gel dic.tif')
sk_3dimg = img_as_ubyte(sk_3dimg)

def show_plane(ax, plane, cmap="gray", title=None):
    ax.imshow(plane,cmap=cmap)
    ax.axis("off")
    
    if title:
        ax.set_title(title)
        

(n_plane, n_row, n_col) = sk_3dimg.shape
_, (a,b,c) = plt.subplots(ncols =3,figsize=(15,5))

show_plane(a,sk_3dimg[n_plane//2], title =f'Plane = {n_plane//2}')
show_plane(b,sk_3dimg[:,n_plane//2,:], title =f'Row = {n_row//2}')
show_plane(c,sk_3dimg[:,:,n_plane//2], title =f'Column = {n_col//2}')

def display(im3d, cmap="gray",step =2):
    _, axes = plt.subplots(nrows=5,ncols =6,figsize=(16,14))
    vmin = im3d.min()
    vmax = im3d.max()
    
    for ax, image in zip(axes.flatten(),im3d[::step]):
        ax.imshow(image, cmap=cmap,vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        
display(sk_3dimg)

from skimage.filters import gaussian
gaussian_smoothed = gaussian(sk_3dimg)

fig = plt.figure(figsize=(12,12))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(sk_3dimg[9,:,:])
ax1.title.set_text('Original')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(gaussian_smoothed[9,:,:])
ax2.title.set_text('Gaussian Smoothed')

binary_img = []
for image in range(sk_3dimg.shape[0]):
    input_img = sk_3dimg[image,:,:]
    gradient_for_M2 = segDefinitions.preSegmentGradBlur(input_img)
    labels = segDefinitions.generateLabels(gradient_for_M2)
    #df = segDefinitions.imgregionProps(input_img,labels,area,imgName)
    
    retBG,th =cv2.threshold(input_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binary = input_img > th
    binary_img.append(labels)
    binary_img_8bit = img_as_ubyte(labels)

processed_img = np.array(labels)

display(processed_img)

plt.imshow(processed_img[10,:,:],cmap="gray")

#io.imsave('processed.tif',processed_img)

import tiffile

tiff_3dimg = tiffile.imread("M1 gel dic.tif")

