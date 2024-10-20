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

import os, sys
import numpy as np
import math
import argparse
import imageio.v2 as im
import cv2 # resize images with float support
from scipy import ndimage # gaussian blur
import skimage.measure # max_pooling with block_reduce
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Spherical harmonics functions
def P(l, m, x):
	pmm = 1.0
	if(m>0):
		somx2 = np.sqrt((1.0-x)*(1.0+x))
		fact = 1.0
		for i in range(1,m+1):
			pmm *= (-fact) * somx2
			fact += 2.0
	
	if(l==m):
		return pmm * np.ones(x.shape)
	
	pmmp1 = x * (2.0*m+1.0) * pmm
	
	if(l==m+1):
		return pmmp1
	
	pll = np.zeros(x.shape)
	for ll in range(m+2, l+1):
		pll = ( (2.0*ll-1.0)*x*pmmp1-(ll+m-1.0)*pmm ) / (ll-m)
		pmm = pmmp1
		pmmp1 = pll
	
	return pll

def divfact(a, b):
	# PBRT style
	if (b == 0):
		return 1.0
	fa = a
	fb = abs(b)
	v = 1.0

	x = fa-fb+1.0
	while x <= fa+fb:
		v *= x;
		x+=1.0

	return 1.0 / v;

def factorial(x):
	if(x == 0):
		return 1.0
	return x * factorial(x-1)

def K(l, m):
	#return np.sqrt((2.0 * l + 1.0) * 0.07957747154594766788 * divfact(l, m))
	return np.sqrt( ((2 * l + 1) * factorial(l-m)) / (4*np.pi*factorial(l+m)) )

def K_fast(l, m):
	cAM = abs(m)
	uVal = 1.0
	k = l+cAM
	while k > (l - cAM):
		uVal *= k
		k-=1
	return np.sqrt( (2.0 * l + 1.0) / ( 4 * np.pi * uVal ) )

def sh(l, m, theta, phi):
	sqrt2 = np.sqrt(2.0)
	if(m==0):
		if np.isscalar(phi):
			return K(l,m)*P(l,m,np.cos(theta))
		else:
			return K(l,m)*P(l,m,np.cos(theta))*np.ones(phi.shape)
	elif(m>0):
		return sqrt2*K(l,m)*np.cos(m*phi)*P(l,m,np.cos(theta))
	else:
		return sqrt2*K(l,-m)*np.sin(-m*phi)*P(l,-m,np.cos(theta))

def sh_evaluate(theta, phi, l_max):
	if np.isscalar(theta):
		coeffsMatrix = np.zeros((1,1,sh_terms(l_max)))
	else:
		coeffsMatrix = np.zeros((theta.shape[0],phi.shape[0],sh_terms(l_max)))

	for l in range(0,l_max+1):
		for m in range(-l,l+1):
			index = sh_index(l, m)
			coeffsMatrix[:,:,index] = sh(l, m, theta, phi)
	return coeffsMatrix

def get_coefficients_matrix(xres,l_max=2):
	yres = int(xres/2)
	# setup fast vectorisation
	x = np.arange(0,xres)
	y = np.arange(0,yres).reshape(yres,1)

	# Setup polar coordinates
	lat_lon = xy_to_ll(x,y,xres,yres)

	# Compute spherical harmonics. Apply thetaOffset due to EXR spherical coordiantes
	Ylm = sh_evaluate(lat_lon[0], lat_lon[1], l_max)
	return Ylm

def get_coefficients_from_file(ibl_filename, l_max=2, resize_width=None, filder_amount=None):
	ibl = im.imread(os.path.join(os.path.dirname(__file__), ibl_filename))
	return get_coefficients_from_image(ibl, l_max=l_max, resize_width=resize_width, filder_amount=filder_amount)

def get_coefficients_from_image(ibl, l_max=2, resize_width=None, filder_amount=None):
	# Resize if necessary (I recommend it for large images)
	if resize_width is not None:
		#ibl = cv2.resize(ibl, dsize=(resize_width,int(resize_width/2)), interpolation=cv2.INTER_CUBIC)
		ibl = resize_image(ibl, resize_width, int(resize_width/2), cv2.INTER_CUBIC)
	elif ibl.shape[1] > 1000:
		#print("Input resolution is large, reducing for efficiency")
		#ibl = cv2.resize(ibl, dsize=(1000,500), interpolation=cv2.INTER_CUBIC)
		ibl = resize_image(ibl, 1000, 500, cv2.INTER_CUBIC)
	xres = ibl.shape[1]
	yres = ibl.shape[0]

	# Pre-filtering, windowing
	if filder_amount is not None:
		ibl = blur_ibl(ibl, amount=filder_amount)

	# Compute sh coefficients
	sh_basis_matrix = get_coefficients_matrix(xres,l_max)

	# Sampling weights
	solid_angles = get_solid_angle_map(xres)

	# Project IBL into SH basis
	n_coeffs = sh_terms(l_max)
	ibl_coeffs = np.zeros((n_coeffs,3))
	for i in range(0,sh_terms(l_max)):
		ibl_coeffs[i,0] = np.sum(ibl[:,:,0]*sh_basis_matrix[:,:,i]*solid_angles)
		ibl_coeffs[i,1] = np.sum(ibl[:,:,1]*sh_basis_matrix[:,:,i]*solid_angles)
		ibl_coeffs[i,2] = np.sum(ibl[:,:,2]*sh_basis_matrix[:,:,i]*solid_angles)

	return ibl_coeffs

def find_windowing_factor(coeffs, max_laplacian=10.0):
	# http://www.ppsloan.org/publications/StupidSH36.pdf 
	# Based on probulator implementation, empirically chosen max_laplacian
	l_max = sh_l_max_from_terms(coeffs.shape[0])
	tableL = np.zeros((l_max+1))
	tableB = np.zeros((l_max+1))

	def sqr(x):
		return x*x
	def cube(x):
		return x*x*x

	for l in range(1, l_max+1):
		tableL[l] = float(sqr(l) * sqr(l + 1))
		B = 0.0
		for m in range(-1, l+1):
			B += np.mean(coeffs[sh_index(l,m),:])
		tableB[l] = B;

	squared_laplacian = 0.0
	for l in range(1, l_max+1):
		squared_laplacian += tableL[l] * tableB[l]

	target_squared_laplacian = max_laplacian * max_laplacian;
	if (squared_laplacian <= target_squared_laplacian): return 0.0

	windowing_factor = 0.0
	iterationLimit = 10000000;
	for i in range(0, iterationLimit):
		f = 0.0
		fd = 0.0
		for l in range(1, l_max+1):
			f += tableL[l] * tableB[l] / sqr(1.0 + windowing_factor * tableL[l]);
			fd += (2.0 * sqr(tableL[l]) * tableB[l]) / cube(1.0 + windowing_factor * tableL[l]);

		f = target_squared_laplacian - f;

		delta = -f / fd;
		windowing_factor += delta;
		if (abs(delta) < 0.0000001):
			break
	return windowing_factor

def apply_windowing(coeffs, windowing_factor=None, verbose=False):
	# http://www.ppsloan.org/publications/StupidSH36.pdf 
	l_max = sh_l_max_from_terms(coeffs.shape[0])
	if windowing_factor is None:
		windowing_factor = find_windowing_factor(coeffs)
	if windowing_factor <= 0: 
		if verbose: print("No windowing applied")
		return coeffs
	if verbose: print("Using windowing_factor: %s" % (windowing_factor))
	for l in range(0, l_max+1):
		s = 1.0 / (1.0 + windowing_factor * l * l * (l + 1.0) * (l + 1.0))
		for m in range(-l, l+1):
			coeffs[sh_index(l,m),:] *= s;
	return coeffs

# Misc functions
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

	img = im.imread(ibl_name, 'EXR-FI')[:,:,0:3]
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

# Spherical harmonics reconstruction
def get_diffuse_coefficients(l_max):
	# From "An Efficient Representation for Irradiance Environment Maps" (2001), Ramamoorthi & Hanrahan
	diffuse_coeffs = [np.pi, (2*np.pi)/3]
	for l in range(2,l_max+1):
		if l%2==0:
			a = (-1.)**((l/2.)-1.)
			b = (l+2.)*(l-1.)
			#c = float(np.math.factorial(l)) / (2**l * np.math.factorial(l/2)**2)
			c = math.factorial(int(l)) / (2**l * math.factorial(int(l//2))**2)
			#s = ((2*l+1)/(4*np.pi))**0.5
			diffuse_coeffs.append(2*np.pi*(a/b)*c)
		else:
			diffuse_coeffs.append(0)
	return np.asarray(diffuse_coeffs) / np.pi

def sh_reconstruct_signal(coeffs, sh_basis_matrix=None, width=600):
	if sh_basis_matrix is None:
		l_max = sh_l_max_from_terms(coeffs.shape[0])
		sh_basis_matrix = get_coefficients_matrix(width,l_max)
	return np.dot(sh_basis_matrix,coeffs).astype(np.float32)

def sh_render(ibl_coeffs, width=600):
	l_max = sh_l_max_from_terms(ibl_coeffs.shape[0])
	diffuse_coeffs =get_diffuse_coefficients(l_max)
	sh_basis_matrix = get_coefficients_matrix(width,l_max)
	rendered_image = np.zeros((int(width/2),width,3))
	for idx in range(0,ibl_coeffs.shape[0]):
		l = l_from_idx(idx)
		coeff_rgb = diffuse_coeffs[l] * ibl_coeffs[idx,:]
		rendered_image[:,:,0] += sh_basis_matrix[:,:,idx] * coeff_rgb[0]
		rendered_image[:,:,1] += sh_basis_matrix[:,:,idx] * coeff_rgb[1]
		rendered_image[:,:,2] += sh_basis_matrix[:,:,idx] * coeff_rgb[2]
	return rendered_image

def get_normal_map_axes_duplicate_rgb(normal_map):
	# Make normal for each axis, but in 3D so we can multiply against RGB
	N3Dx = np.repeat(normal_map[:,:,0][:, :, np.newaxis], 3, axis=2)
	N3Dy = np.repeat(normal_map[:,:,1][:, :, np.newaxis], 3, axis=2)
	N3Dz = np.repeat(normal_map[:,:,2][:, :, np.newaxis], 3, axis=2)
	return N3Dx, N3Dy, N3Dz

def sh_render_l2(ibl_coeffs, normal_map):
	# From "An Efficient Representation for Irradiance Environment Maps" (2001), Ramamoorthi & Hanrahan
	C1 = 0.429043
	C2 = 0.511664
	C3 = 0.743125
	C4 = 0.886227
	C5 = 0.247708
	N3Dx, N3Dy, N3Dz = get_normal_map_axes_duplicate_rgb(normal_map)
	return	(C4 * ibl_coeffs[0,:] + \
			2.0 * C2 * ibl_coeffs[3,:] * N3Dx + \
			2.0 * C2 * ibl_coeffs[1,:] * N3Dy + \
			2.0 * C2 * ibl_coeffs[2,:] * N3Dz + \
			C1 * ibl_coeffs[8,:] * (N3Dx * N3Dx - N3Dy * N3Dy) + \
			C3 * ibl_coeffs[6,:] * N3Dz * N3Dz - C5 * ibl_coeffs[6] + \
			2.0 * C1 * ibl_coeffs[4,:] * N3Dx * N3Dy + \
			2.0 * C1 * ibl_coeffs[7,:] * N3Dx * N3Dz + \
			2.0 * C1 * ibl_coeffs[5,:] * N3Dy * N3Dz ) / np.pi

def get_normal_map(width):
	height = int(width/2)
	x = np.arange(0,width)
	y = np.arange(0,height).reshape(height,1)
	lat_lon = xy_to_ll(x,y,width,height)
	return spherical_to_cartesian_2(lat_lon[0], lat_lon[1])

def sh_reconstruct_diffuse_map(ibl_coeffs, width=600):
	# Rendering 
	if ibl_coeffs.shape[0] == 9: # L2
		# setup fast vectorisation
		xyz = get_normal_map(width)
		rendered_image = sh_render_l2(ibl_coeffs, xyz)
	else: # !L2
		rendered_image = sh_render(ibl_coeffs, width)

	return rendered_image.astype(np.float32)

def write_reconstruction(c, l_max, fn='', width=600, output_dir='./output/'):
	im.imwrite(output_dir+'_sh_light_l'+str(l_max)+fn+'.exr',sh_reconstruct_signal(c, width=width))
	im.imwrite(output_dir+'_sh_render_l'+str(l_max)+fn+'.exr',sh_reconstruct_diffuse_map(c, width=width))

# Utility functions for SPH
def sh_print(coeffs, precision=3):
	n_coeffs = coeffs.shape[0]
	l_max = sh_l_max_from_terms(coeffs.shape[0])
	currentBand = -1
	for idx in range(0,n_coeffs):
		band = l_from_idx(idx)
		if currentBand!=band:
			currentBand = band
			print('L'+str(currentBand)+":")
		print(np.around(coeffs[idx,:],precision))
	print('')

def sh_print_to_file(coeffs, precision=3, output_file_path="./output/_coefficients.txt"):
	n_coeffs = coeffs.shape[0]
	l_max = sh_l_max_from_terms(coeffs.shape[0])
	currentBand = -1

	with open(output_file_path, 'w') as file:		
		for idx in range(0,n_coeffs):

			# Print the band level
			band = l_from_idx(idx)
			if currentBand!=band:
				currentBand = band
				outputData_BandLevel = "L"+str(currentBand)+":"
				print(outputData_BandLevel)
				file.write(outputData_BandLevel+'\n')

			# Print the coefficients at this band level
			outputData_Coefficients = np.around(coeffs[idx,:],precision)
			print(outputData_Coefficients)
			file.write("[")
			for i in range(0,len(outputData_Coefficients)):
				file.write(str(outputData_Coefficients[i]))
				if i<len(outputData_Coefficients)-1: # don't add white space after last coefficient
					file.write(' ')
			file.write(']\n')


def sh_terms_within_band(l):
	return (l*2)+1

def sh_terms(l_max):
	return (l_max + 1) * (l_max + 1)

def sh_l_max_from_terms(terms):
	return int(np.sqrt(terms)-1)

def sh_index(l, m):
	return l*l+l+m

def l_from_idx(idx):
	return int(np.sqrt(idx))

def paint_negatives(img):
	indices = [img[:,:,0] < 0 or img[:,:,1] < 0 or img[:,:,2] < 0]
	img[indices[0],0] = abs((img[indices[0],0]+img[indices[0],1]+img[indices[0],2])/3)*10
	img[indices[0],1] = 0
	img[indices[0],2] = 0

def blur_ibl(ibl, amount=5):
	x = ibl.copy()
	x[:,:,0] = ndimage.gaussian_filter(ibl[:,:,0], sigma=amount)
	x[:,:,1] = ndimage.gaussian_filter(ibl[:,:,1], sigma=amount)
	x[:,:,2] = ndimage.gaussian_filter(ibl[:,:,2], sigma=amount)
	return x

def spherical_to_cartesian_2(theta, phi):
	phi = phi + np.pi
	x = np.sin(theta)*np.cos(phi)
	y = np.cos(theta)
	z = np.sin(theta)*np.sin(phi)
	if not np.isscalar(x):
		y = np.repeat(y, x.shape[1], axis=1)
	return np.moveaxis(np.asarray([x,z,y]), 0,2)

def spherical_to_cartesian(theta, phi):
	x = np.sin(theta)*np.cos(phi)
	y = np.sin(theta)*np.sin(phi)
	z = np.cos(theta)
	if not np.isscalar(x):
		z = np.repeat(z, x.shape[1], axis=1)
	return np.moveaxis(np.asarray([x,z,y]), 0,2)

def xy_to_ll(x,y,width,height):
	def yLocToLat(yLoc, height):
		return (yLoc / (float(height)/np.pi))
	def xLocToLon(xLoc, width):
		return (xLoc / (float(width)/(np.pi * 2)))
	return np.asarray([yLocToLat(y, height), xLocToLon(x, width)], dtype=object)

def get_cartesian_map(width):
	height = int(width/2)
	image = np.zeros((height,width))
	x = np.arange(0,width)
	y = np.arange(0,height).reshape(height,1)
	lat_lon = xy_to_ll(x,y,width,height)
	return spherical_to_cartesian(lat_lon[0], lat_lon[1])

# Example functions
def cosine_lobe_example(dir, width):
	# https://github.com/google/spherical-harmonics/blob/master/sh/spherical_harmonics.cc
	xyz = get_cartesian_map(width)
	return grey_to_colour(np.clip(np.sum(dir*xyz, axis=2), 0.0, 1.0))
	#return grey_to_colour(1.0 * (np.exp(-(np.arccos(np.sum(dir*xyz, axis=2))/(0.5**2)))))

def robin_green_example(width):
	# "The Gritty Details" by Robin Green
	#dir1 = np.asarray([0,1,0])
	theta = np.repeat(lat_lon[0][:, np.newaxis], width, axis=1).reshape((height,width))
	phi = np.repeat(lat_lon[1][np.newaxis,:], height, axis=0).reshape((height,width))
	return grey_to_colour(	np.maximum(0.0, 5*np.cos(theta)-4) + 
						np.maximum(0.0, -4*np.sin(theta-np.pi) * np.cos(phi-2.5)-3) )

# Visualisations
def sh_visualise(l_max=2, sh_basis_matrix=None, show_it=False, output_dir='./output/'):
	cdict =	{'red':	((0.0, 1.0, 1.0),
					(0.5, 0.0, 0.0),
					(1.0, 0.0, 0.0)),
			'green':((0.0, 0.0, 0.0),
					(0.5, 0.0, 0.0),
					(1.0, 1.0, 1.0)),
			'blue':	((0.0, 0.0, 0.0),
					(1.0, 0.0, 0.0))}

	if sh_basis_matrix is None:
		sh_basis_matrix = get_coefficients_matrix(600,l_max)

	l_max = sh_l_max_from_terms(sh_basis_matrix.shape[2])

	rows = l_max+1
	cols = sh_terms_within_band(l_max)
	img_index = 0

	if l_max==0:
		plt.imshow(sh_basis_matrix[:,:,0], cmap=LinearSegmentedColormap('RedGreen', cdict), vmin=-1, vmax=1)
		plt.axis("off")
		plt.savefig(output_dir+'_fig_sh_l'+str(l_max)+'.jpg')
		if show_it:
			plt.show()
		return

	fig, axs = plt.subplots(nrows=rows, ncols=cols, gridspec_kw={'wspace':0.1, 'hspace':-0.4}, squeeze=True, figsize=(16, 8))
	for c in range(0,cols):
		for r in range(0,rows):
			axs[r,c].axis('off')

	for l in range(0,l_max+1):
		n_in_band = sh_terms_within_band(l)
		col_offset = int(cols/2) - int(n_in_band/2)
		row_offset = (l*cols)+1
		index = row_offset+col_offset
		for i in range(0,n_in_band):
			axs[l, i+col_offset].axis("off")
			axs[l, i+col_offset].imshow(sh_basis_matrix[:,:,img_index], cmap=LinearSegmentedColormap('RedGreen', cdict), vmin=-1, vmax=1)
			img_index+=1

	#plt.tight_layout()
	plt.savefig(output_dir+'_fig_sh_l'+str(l_max)+'.jpg')
	if show_it:
		plt.show()

if __name__ == "__main__":
	# Parsing input
	parser = argparse.ArgumentParser(description="Process IBL filename and number of bands.")
	parser.add_argument('--ibl_filename', type=str, default='./images/grace-new.exr',
	                    help="Path to the ibl filename (default: './images/grace-new.exr').")
	parser.add_argument('--l_max', type=int, default=3,
	                    help="Number of bands (must be an integer, default: 3).")
	parser.add_argument('--output_dir', type=str, default='./output/', 
	                    help="Output directory (default: './output/').")
	parser.add_argument('--resize_width', type=int, default=100, 
	                    help="Width to resize the image (default: 1000).")
	args = parser.parse_args()

	print("Spherical Harmonics for latitude-longitude radiance maps")
	print(f"Input File (IBL): {args.ibl_filename}")
	print(f"Number of Bands (l_max): {args.l_max}")
	print(f"Resize Width: {args.resize_width}")
	print(f"Output Directory: {args.output_dir}")
	print("")

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	resize_height = int(args.resize_width/2)

	# Visualise the SPH
	print("Plotting spherical harmonic functions...")
	sh_visualise(args.l_max, show_it=False, output_dir=args.output_dir)

	# Read image
	print("Reading in image...")
	ibl_GT = resize_image(im.imread(args.ibl_filename, 'EXR-FI')[:,:,0:3], args.resize_width, resize_height, cv2.INTER_CUBIC)
	im.imwrite(args.output_dir+'_ibl_GT.exr', ibl_GT.astype(np.float32))

	# SPH projection
	print("Running spherical harmonics...")
	ibl_coeffs = get_coefficients_from_file(args.ibl_filename, args.l_max, resize_width=args.resize_width)
	#ibl_coeffs = get_coefficients_from_image(ibl_GT, args.l_max, resize_width=args.resize_width)
	write_reconstruction(ibl_coeffs, args.l_max, '_SPH', width=args.resize_width, output_dir=args.output_dir)
	#sh_print(ibl_coeffs)
	sh_print_to_file(ibl_coeffs)
	print("Spherical harmonics processing is complete.\n")

	# Diffuse convolution diffuse map (ground truth)
	print("Generating ground truth diffuse map for comparison...")
	diffuse_low_res_width = 32 # changing resolution is a trade off between processing time and ground truth quality
	output_width = args.resize_width
	fn = args.output_dir+'_diffuse_ibl_gt_'+str(args.resize_width)+'_'+str(diffuse_low_res_width)+'_'+str(output_width)+'.exr'
	diffuse_ibl_gt = get_diffuse_map(args.ibl_filename, width=args.resize_width, width_low_res=diffuse_low_res_width, output_width=output_width) 
	im.imwrite(fn, diffuse_ibl_gt.astype(np.float32))

	print("Complete.")
