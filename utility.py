'''
MIT License

Copyright (c) 2018 Andrew Chalmers

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

'''
Spherical harmonics for radiance maps using numpy

Assumes:
Equirectangular format
theta: [0 to pi], from top to bottom row of pixels
phi: [0 to 2*Pi], from left to right column of pixels
'''

import os
import numpy as np
import math
import argparse
import imageio.v3 as im
import cv2 # resize images with float support
from scipy import ndimage # gaussian blur
import skimage.measure # max_pooling with block_reduce
import time

def blur_ibl(ibl, amount=5):
	x = ibl.copy()
	x[:,:,0] = ndimage.gaussian_filter(ibl[:,:,0], sigma=amount)
	x[:,:,1] = ndimage.gaussian_filter(ibl[:,:,1], sigma=amount)
	x[:,:,2] = ndimage.gaussian_filter(ibl[:,:,2], sigma=amount)
	return x

def resize_image(img, width, height, interpolation=cv2.INTER_CUBIC):
	if img.shape[1]<width: # up res
		if interpolation=='max_pooling':
			return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
		else:
			return cv2.resize(img, (width, height), interpolation=interpolation)
	if interpolation=='max_pooling': # down res, max pooling
		try:
			scale_factor = int(float(img.shape[1])/width)
			factored_width = width*scale_factor
			img = cv2.resize(img, (factored_width, int(factored_width/2)), interpolation=cv2.INTER_CUBIC)
			block_size = scale_factor
			r = skimage.measure.block_reduce(img[:,:,0], (block_size,block_size), np.max)
			g = skimage.measure.block_reduce(img[:,:,1], (block_size,block_size), np.max)
			b = skimage.measure.block_reduce(img[:,:,2], (block_size,block_size), np.max)
			img = np.dstack((np.dstack((r,g)),b)).astype(np.float32)
			return img
		except:
			print("Failed to do max_pooling, using default")
			return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
	else: # down res, using interpolation
		return cv2.resize(img, (width, height), interpolation=interpolation)

def grey_to_colour(grey_img):
	return (np.repeat(grey_img[:,:][:, :, np.newaxis], 3, axis=2)).astype(np.float32)

def colour_to_grey(col_img):
	return ((col_img[:,:,0]+col_img[:,:,1]+col_img[:,:,2])/3).astype(np.float32)

def pole_scale(y, width, relative=True):
	"""
	y = y pixel position (cast as a float)
	Scaling pixels lower toward the poles
	Sample scaling in latitude-longitude maps:
	http://webstaff.itn.liu.se/~jonun/web/teaching/2012-TNM089/Labs/IBL/scale_factors.pdf
	"""
	height = int(width/2)
	piHalf = np.pi/2
	pi4 = np.pi*4
	pi2_over_width = (np.pi*2)/width
	pi_over_height = np.pi/height
	theta = (1.0 - ((y + 0.5) / height)) * np.pi
	scale_factor = (1.0 / pi4) * pi2_over_width * (np.cos(theta - (pi_over_height / 2.0)) - np.cos(theta + (pi_over_height / 2.0)))
	if relative:
		scale_factor /= (1.0 / pi4) * pi2_over_width * (np.cos(piHalf - (pi_over_height / 2.0)) - np.cos(piHalf + (pi_over_height / 2.0)))
	return scale_factor

def get_solid_angle(y, width, is3D=False):
	"""
	y = y pixel position (cast as a float)
	Solid angles in latitude-longitude maps:
	http://webstaff.itn.liu.se/~jonun/web/teaching/2012-TNM089/Labs/IBL/scale_factors.pdf
	"""
	height = int(width/2)
	pi2_over_width = (np.pi*2)/width
	pi_over_height = np.pi/height
	theta = (1.0 - ((y + 0.5) / height)) * np.pi
	return pi2_over_width * (np.cos(theta - (pi_over_height / 2.0)) - np.cos(theta + (pi_over_height / 2.0)))

def get_solid_angle_map(width):
	height = int(width/2)
	return np.repeat(get_solid_angle(np.arange(0,height), width)[:, np.newaxis], width, axis=1)

def get_diffuse_map(ibl_name, width=600, width_low_res=32, output_width=None):
	if output_width is None:
		output_width = width
	height = int(width/2)
	height_low_res = int(width_low_res/2)

	img = im.imread(ibl_name, plugin='EXR-FI')[:,:,0:3]
	img = resize_image(img, width, height)

	uv_x = np.arange(float(width))/width
	uv_x = np.tile(uv_x, (height, 1))

	uv_y = np.arange(float(height))/height
	uv_y = 1-np.tile(uv_y, (width,1)).transpose()

	phi = np.pi*(uv_y-0.5)
	theta = 2*np.pi*(1-uv_x)

	cos_phi = np.cos(phi) 
	d_x = cos_phi*np.sin(theta)
	d_y = np.sin(phi)
	d_z = cos_phi*np.cos(theta)

	solid_angles = get_solid_angle_map(width)

	print("Convolving", (width_low_res,height_low_res))
	output_diffuse_map = np.zeros((height_low_res,width_low_res,3))
	def compute(x_i, y_i):
		x_i_s = int((float(x_i)/width_low_res)*width)
		y_i_s = int((float(y_i)/height_low_res)*height)
		dot = np.maximum(0, d_x[y_i_s, x_i_s]*d_x + d_y[y_i_s, x_i_s]*d_y + d_z[y_i_s, x_i_s]*d_z)
		for c_i in range(0,3):
			output_diffuse_map[y_i, x_i, c_i] = np.sum(dot * img[:,:,c_i] * solid_angles) / np.pi

	start = time.time()
	for x_i in range(0,output_diffuse_map.shape[1]):
		#print(float(x_i)/output_diffuse_map.shape[1])
		for y_i in range(0,output_diffuse_map.shape[0]):
			compute(x_i, y_i)
	end = time.time()
	print("Elapsed time: %.4f seconds" % (end - start))

	if width_low_res < output_width:
		output_diffuse_map = resize_image(output_diffuse_map, output_width, int(output_width/2), cv2.INTER_LANCZOS4)

	return output_diffuse_map.astype(np.float32)
