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

def Kfast(l, m):
	cAM = abs(m)
	uVal = 1.0
	k = l+cAM
	while k > (l - cAM):
		uVal *= k
		k-=1
	return np.sqrt( (2.0 * l + 1.0) / ( 4 * np.pi * uVal ) )

def SH(l, m, theta, phi):
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

def shEvaluate(theta, phi, lmax):
	if np.isscalar(theta):
		coeffsMatrix = np.zeros((1,1,shTerms(lmax)))
	else:
		coeffsMatrix = np.zeros((theta.shape[0],phi.shape[0],shTerms(lmax)))

	for l in range(0,lmax+1):
		for m in range(-l,l+1):
			index = shIndex(l, m)
			coeffsMatrix[:,:,index] = SH(l, m, theta, phi)
	return coeffsMatrix

def getCoefficientsMatrix(xres,lmax=2):
	yres = int(xres/2)
	# setup fast vectorisation
	x = np.arange(0,xres)
	y = np.arange(0,yres).reshape(yres,1)

	# Setup polar coordinates
	latLon = xy2ll(x,y,xres,yres)

	# Compute spherical harmonics. Apply thetaOffset due to EXR spherical coordiantes
	Ylm = shEvaluate(latLon[0], latLon[1], lmax)
	return Ylm

def getCoefficientsFromFile(ibl_filename, lmax=2, resize_width=None, filterAmount=None):
	ibl = im.imread(os.path.join(os.path.dirname(__file__), ibl_filename))
	return getCoefficientsFromImage(ibl, lmax=lmax, resize_width=resize_width, filterAmount=filterAmount)

def getCoefficientsFromImage(ibl, lmax=2, resize_width=None, filterAmount=None):
	# Resize if necessary (I recommend it for large images)
	if resize_width is not None:
		#ibl = cv2.resize(ibl, dsize=(resize_width,int(resize_width/2)), interpolation=cv2.INTER_CUBIC)
		ibl = resizeImage(ibl, resize_width, int(resize_width/2), cv2.INTER_CUBIC)
	elif ibl.shape[1] > 1000:
		#print("Input resolution is large, reducing for efficiency")
		#ibl = cv2.resize(ibl, dsize=(1000,500), interpolation=cv2.INTER_CUBIC)
		ibl = resizeImage(ibl, 1000, 500, cv2.INTER_CUBIC)
	xres = ibl.shape[1]
	yres = ibl.shape[0]

	# Pre-filtering, windowing
	if filterAmount is not None:
		ibl = blurIBL(ibl, amount=filterAmount)

	# Compute sh coefficients
	sh_basis_matrix = getCoefficientsMatrix(xres,lmax)

	# Sampling weights
	solidAngles = getSolidAngleMap(xres)

	# Project IBL into SH basis
	nCoeffs = shTerms(lmax)
	iblCoeffs = np.zeros((nCoeffs,3))
	for i in range(0,shTerms(lmax)):
		iblCoeffs[i,0] = np.sum(ibl[:,:,0]*sh_basis_matrix[:,:,i]*solidAngles)
		iblCoeffs[i,1] = np.sum(ibl[:,:,1]*sh_basis_matrix[:,:,i]*solidAngles)
		iblCoeffs[i,2] = np.sum(ibl[:,:,2]*sh_basis_matrix[:,:,i]*solidAngles)

	return iblCoeffs

def findWindowingFactor(coeffs, maxLaplacian=10.0):
	# http://www.ppsloan.org/publications/StupidSH36.pdf 
	# Based on probulator implementation, empirically chosen maxLaplacian
	lmax = sh_lmax_from_terms(coeffs.shape[0])
	tableL = np.zeros((lmax+1))
	tableB = np.zeros((lmax+1))

	def sqr(x):
		return x*x
	def cube(x):
		return x*x*x

	for l in range(1, lmax+1):
		tableL[l] = float(sqr(l) * sqr(l + 1))
		B = 0.0
		for m in range(-1, l+1):
			B += np.mean(coeffs[shIndex(l,m),:])
		tableB[l] = B;

	squaredLaplacian = 0.0
	for l in range(1, lmax+1):
		squaredLaplacian += tableL[l] * tableB[l]

	targetSquaredLaplacian = maxLaplacian * maxLaplacian;
	if (squaredLaplacian <= targetSquaredLaplacian): return 0.0

	windowingFactor = 0.0
	iterationLimit = 10000000;
	for i in range(0, iterationLimit):
		f = 0.0
		fd = 0.0
		for l in range(1, lmax+1):
			f += tableL[l] * tableB[l] / sqr(1.0 + windowingFactor * tableL[l]);
			fd += (2.0 * sqr(tableL[l]) * tableB[l]) / cube(1.0 + windowingFactor * tableL[l]);

		f = targetSquaredLaplacian - f;

		delta = -f / fd;
		windowingFactor += delta;
		if (abs(delta) < 0.0000001):
			break
	return windowingFactor

def applyWindowing(coeffs, windowingFactor=None, verbose=False):
	# http://www.ppsloan.org/publications/StupidSH36.pdf 
	lmax = sh_lmax_from_terms(coeffs.shape[0])
	if windowingFactor is None:
		windowingFactor = findWindowingFactor(coeffs)
	if windowingFactor <= 0: 
		if verbose: print("No windowing applied")
		return coeffs
	if verbose: print("Using windowingFactor: %s" % (windowingFactor))
	for l in range(0, lmax+1):
		s = 1.0 / (1.0 + windowingFactor * l * l * (l + 1.0) * (l + 1.0))
		for m in range(-l, l+1):
			coeffs[shIndex(l,m),:] *= s;
	return coeffs

# Misc functions
def resizeImage(img, width, height, interpolation=cv2.INTER_CUBIC):
	if img.shape[1]<width: # up res
		if interpolation=='max_pooling':
			return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
		else:
			return cv2.resize(img, (width, height), interpolation=interpolation)
	if interpolation=='max_pooling': # down res, max pooling
		try:
			import skimage.measure
			scaleFactor = int(float(img.shape[1])/width)
			factoredWidth = width*scaleFactor
			img = cv2.resize(img, (factoredWidth, int(factoredWidth/2)), interpolation=cv2.INTER_CUBIC)
			blockSize = scaleFactor
			r = skimage.measure.block_reduce(img[:,:,0], (blockSize,blockSize), np.max)
			g = skimage.measure.block_reduce(img[:,:,1], (blockSize,blockSize), np.max)
			b = skimage.measure.block_reduce(img[:,:,2], (blockSize,blockSize), np.max)
			img = np.dstack((np.dstack((r,g)),b)).astype(np.float32)
			return img
		except:
			print("Failed to do max_pooling, using default")
			return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
	else: # down res, using interpolation
		return cv2.resize(img, (width, height), interpolation=interpolation)

def grey2colour(greyImg):
	return (np.repeat(greyImg[:,:][:, :, np.newaxis], 3, axis=2)).astype(np.float32)

def colour2grey(colImg):
	return ((colImg[:,:,0]+colImg[:,:,1]+colImg[:,:,2])/3).astype(np.float32)

def poleScale(y, width, relative=True):
	"""
	y = y pixel position (cast as a float)
	Scaling pixels lower toward the poles
	Sample scaling in latitude-longitude maps:
	http://webstaff.itn.liu.se/~jonun/web/teaching/2012-TNM089/Labs/IBL/scalefactors.pdf
	"""
	height = int(width/2)
	piHalf = np.pi/2
	pi4 = np.pi*4
	pi2OverWidth = (np.pi*2)/width
	piOverHeight = np.pi/height
	theta = (1.0 - ((y + 0.5) / height)) * np.pi
	scaleFactor = (1.0 / pi4) * pi2OverWidth * (np.cos(theta - (piOverHeight / 2.0)) - np.cos(theta + (piOverHeight / 2.0)))
	if relative:
		scaleFactor /= (1.0 / pi4) * pi2OverWidth * (np.cos(piHalf - (piOverHeight / 2.0)) - np.cos(piHalf + (piOverHeight / 2.0)))
	return scaleFactor

def getSolidAngle(y, width, is3D=False):
	"""
	y = y pixel position (cast as a float)
	Solid angles in latitude-longitude maps:
	http://webstaff.itn.liu.se/~jonun/web/teaching/2012-TNM089/Labs/IBL/scalefactors.pdf
	"""
	height = int(width/2)
	pi2OverWidth = (np.pi*2)/width
	piOverHeight = np.pi/height
	theta = (1.0 - ((y + 0.5) / height)) * np.pi
	return pi2OverWidth * (np.cos(theta - (piOverHeight / 2.0)) - np.cos(theta + (piOverHeight / 2.0)))

def getSolidAngleMap(width):
	height = int(width/2)
	return np.repeat(getSolidAngle(np.arange(0,height), width)[:, np.newaxis], width, axis=1)

def getDiffuseMap(ibl_name, width=600, width_low_res=32, output_width=None):
	if output_width is None:
		output_width = width
	height = int(width/2)
	height_low_res = int(width_low_res/2)

	img = im.imread(ibl_name, 'EXR-FI')[:,:,0:3]
	img = resizeImage(img, width, height)

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

	solidAngles = getSolidAngleMap(width)

	print("Convolving", (width_low_res,height_low_res))
	outputDiffuseMap = np.zeros((height_low_res,width_low_res,3))
	def compute(x_i, y_i):
		x_i_s = int((float(x_i)/width_low_res)*width)
		y_i_s = int((float(y_i)/height_low_res)*height)
		dot = np.maximum(0, d_x[y_i_s, x_i_s]*d_x + d_y[y_i_s, x_i_s]*d_y + d_z[y_i_s, x_i_s]*d_z)
		for c_i in range(0,3):
			outputDiffuseMap[y_i, x_i, c_i] = np.sum(dot * img[:,:,c_i] * solidAngles) / np.pi

	start = time.time()
	for x_i in range(0,outputDiffuseMap.shape[1]):
		#print(float(x_i)/outputDiffuseMap.shape[1])
		for y_i in range(0,outputDiffuseMap.shape[0]):
			compute(x_i, y_i)
	end = time.time()
	print("Elapsed time: %.4f seconds" % (end - start))

	if width_low_res < output_width:
		outputDiffuseMap = resizeImage(outputDiffuseMap, output_width, int(output_width/2), cv2.INTER_LANCZOS4)

	return outputDiffuseMap.astype(np.float32)

# Spherical harmonics reconstruction
def getDiffuseCoefficients(lmax):
	# From "An Efficient Representation for Irradiance Environment Maps" (2001), Ramamoorthi & Hanrahan
	diffuseCoeffs = [np.pi, (2*np.pi)/3]
	for l in range(2,lmax+1):
		if l%2==0:
			a = (-1.)**((l/2.)-1.)
			b = (l+2.)*(l-1.)
			#c = float(np.math.factorial(l)) / (2**l * np.math.factorial(l/2)**2)
			c = math.factorial(int(l)) / (2**l * math.factorial(int(l//2))**2)
			#s = ((2*l+1)/(4*np.pi))**0.5
			diffuseCoeffs.append(2*np.pi*(a/b)*c)
		else:
			diffuseCoeffs.append(0)
	return np.asarray(diffuseCoeffs) / np.pi

def shReconstructSignal(coeffs, sh_basis_matrix=None, width=600):
	if sh_basis_matrix is None:
		lmax = sh_lmax_from_terms(coeffs.shape[0])
		sh_basis_matrix = getCoefficientsMatrix(width,lmax)
	return np.dot(sh_basis_matrix,coeffs).astype(np.float32)

def shRender(iblCoeffs, width=600):
	lmax = sh_lmax_from_terms(iblCoeffs.shape[0])
	diffuseCoeffs =getDiffuseCoefficients(lmax)
	sh_basis_matrix = getCoefficientsMatrix(width,lmax)
	renderedImage = np.zeros((int(width/2),width,3))
	for idx in range(0,iblCoeffs.shape[0]):
		l = l_from_idx(idx)
		coeff_rgb = diffuseCoeffs[l] * iblCoeffs[idx,:]
		renderedImage[:,:,0] += sh_basis_matrix[:,:,idx] * coeff_rgb[0]
		renderedImage[:,:,1] += sh_basis_matrix[:,:,idx] * coeff_rgb[1]
		renderedImage[:,:,2] += sh_basis_matrix[:,:,idx] * coeff_rgb[2]
	return renderedImage

def getNormalMapAxesDuplicateRGB(normalMap):
	# Make normal for each axis, but in 3D so we can multiply against RGB
	N3Dx = np.repeat(normalMap[:,:,0][:, :, np.newaxis], 3, axis=2)
	N3Dy = np.repeat(normalMap[:,:,1][:, :, np.newaxis], 3, axis=2)
	N3Dz = np.repeat(normalMap[:,:,2][:, :, np.newaxis], 3, axis=2)
	return N3Dx, N3Dy, N3Dz

def shRenderL2(iblCoeffs, normalMap):
	# From "An Efficient Representation for Irradiance Environment Maps" (2001), Ramamoorthi & Hanrahan
	C1 = 0.429043
	C2 = 0.511664
	C3 = 0.743125
	C4 = 0.886227
	C5 = 0.247708
	N3Dx, N3Dy, N3Dz = getNormalMapAxesDuplicateRGB(normalMap)
	return	(C4 * iblCoeffs[0,:] + \
			2.0 * C2 * iblCoeffs[3,:] * N3Dx + \
			2.0 * C2 * iblCoeffs[1,:] * N3Dy + \
			2.0 * C2 * iblCoeffs[2,:] * N3Dz + \
			C1 * iblCoeffs[8,:] * (N3Dx * N3Dx - N3Dy * N3Dy) + \
			C3 * iblCoeffs[6,:] * N3Dz * N3Dz - C5 * iblCoeffs[6] + \
			2.0 * C1 * iblCoeffs[4,:] * N3Dx * N3Dy + \
			2.0 * C1 * iblCoeffs[7,:] * N3Dx * N3Dz + \
			2.0 * C1 * iblCoeffs[5,:] * N3Dy * N3Dz ) / np.pi

def getNormalMap(width):
	height = int(width/2)
	x = np.arange(0,width)
	y = np.arange(0,height).reshape(height,1)
	latLon = xy2ll(x,y,width,height)
	return spherical2Cartesian2(latLon[0], latLon[1])

def shReconstructDiffuseMap(iblCoeffs, width=600):
	# Rendering 
	if iblCoeffs.shape[0] == 9: # L2
		# setup fast vectorisation
		xyz = getNormalMap(width)
		renderedImage = shRenderL2(iblCoeffs, xyz)
	else: # !L2
		renderedImage = shRender(iblCoeffs, width)

	return renderedImage.astype(np.float32)

def writeReconstruction(c, lmax, fn='', width=600, output_dir='./output/'):
	im.imwrite(output_dir+'_sh_light_l'+str(lmax)+fn+'.exr',shReconstructSignal(c, width=width))
	im.imwrite(output_dir+'_sh_render_l'+str(lmax)+fn+'.exr',shReconstructDiffuseMap(c, width=width))

# Utility functions for SPH
def shPrint(coeffs, precision=3):
	nCoeffs = coeffs.shape[0]
	lmax = sh_lmax_from_terms(coeffs.shape[0])
	currentBand = -1
	for idx in range(0,nCoeffs):
		band = l_from_idx(idx)
		if currentBand!=band:
			currentBand = band
			print('L'+str(currentBand)+":")
		print(np.around(coeffs[idx,:],precision))
	print('')

def shPrintToFile(coeffs, precision=3, output_file_path="./output/_coefficients.txt"):
	nCoeffs = coeffs.shape[0]
	lmax = sh_lmax_from_terms(coeffs.shape[0])
	currentBand = -1

	with open(output_file_path, 'w') as file:		
		for idx in range(0,nCoeffs):

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


def shTermsWithinBand(l):
	return (l*2)+1

def shTerms(lmax):
	return (lmax + 1) * (lmax + 1)

def sh_lmax_from_terms(terms):
	return int(np.sqrt(terms)-1)

def shIndex(l, m):
	return l*l+l+m

def l_from_idx(idx):
	return int(np.sqrt(idx))

def paintNegatives(img):
	indices = [img[:,:,0] < 0 or img[:,:,1] < 0 or img[:,:,2] < 0]
	img[indices[0],0] = abs((img[indices[0],0]+img[indices[0],1]+img[indices[0],2])/3)*10
	img[indices[0],1] = 0
	img[indices[0],2] = 0

def blurIBL(ibl, amount=5):
	x = ibl.copy()
	x[:,:,0] = ndimage.gaussian_filter(ibl[:,:,0], sigma=amount)
	x[:,:,1] = ndimage.gaussian_filter(ibl[:,:,1], sigma=amount)
	x[:,:,2] = ndimage.gaussian_filter(ibl[:,:,2], sigma=amount)
	return x

def spherical2Cartesian2(theta, phi):
	phi = phi + np.pi
	x = np.sin(theta)*np.cos(phi)
	y = np.cos(theta)
	z = np.sin(theta)*np.sin(phi)
	if not np.isscalar(x):
		y = np.repeat(y, x.shape[1], axis=1)
	return np.moveaxis(np.asarray([x,z,y]), 0,2)

def spherical2Cartesian(theta, phi):
	x = np.sin(theta)*np.cos(phi)
	y = np.sin(theta)*np.sin(phi)
	z = np.cos(theta)
	if not np.isscalar(x):
		z = np.repeat(z, x.shape[1], axis=1)
	return np.moveaxis(np.asarray([x,z,y]), 0,2)

def xy2ll(x,y,width,height):
	def yLocToLat(yLoc, height):
		return (yLoc / (float(height)/np.pi))
	def xLocToLon(xLoc, width):
		return (xLoc / (float(width)/(np.pi * 2)))
	return np.asarray([yLocToLat(y, height), xLocToLon(x, width)], dtype=object)

def getCartesianMap(width):
	height = int(width/2)
	image = np.zeros((height,width))
	x = np.arange(0,width)
	y = np.arange(0,height).reshape(height,1)
	latLon = xy2ll(x,y,width,height)
	return spherical2Cartesian(latLon[0], latLon[1])

# Example functions
def cosine_lobe_example(dir, width):
	# https://github.com/google/spherical-harmonics/blob/master/sh/spherical_harmonics.cc
	xyz = getCartesianMap(width)
	return grey2colour(np.clip(np.sum(dir*xyz, axis=2), 0.0, 1.0))
	#return grey2colour(1.0 * (np.exp(-(np.arccos(np.sum(dir*xyz, axis=2))/(0.5**2)))))

def robin_green_example(width):
	# "The Gritty Details" by Robin Green
	dir1 = np.asarray([0,1,0])
	theta = np.repeat(latLon[0][:, np.newaxis], width, axis=1).reshape((height,width))
	phi = np.repeat(latLon[1][np.newaxis,:], height, axis=0).reshape((height,width))
	return grey2colour(	np.maximum(0.0, 5*np.cos(theta)-4) + 
						np.maximum(0.0, -4*np.sin(theta-np.pi) * np.cos(phi-2.5)-3) )

# Visualisations
def sh_visualise(lmax=2, sh_basis_matrix=None, showIt=False, output_dir='./output/'):
	cdict =	{'red':	((0.0, 1.0, 1.0),
					(0.5, 0.0, 0.0),
					(1.0, 0.0, 0.0)),
			'green':((0.0, 0.0, 0.0),
					(0.5, 0.0, 0.0),
					(1.0, 1.0, 1.0)),
			'blue':	((0.0, 0.0, 0.0),
					(1.0, 0.0, 0.0))}

	if sh_basis_matrix is None:
		sh_basis_matrix = getCoefficientsMatrix(600,lmax)

	lmax = sh_lmax_from_terms(sh_basis_matrix.shape[2])

	rows = lmax+1
	cols = shTermsWithinBand(lmax)
	imgIndex = 0

	if lmax==0:
		plt.imshow(sh_basis_matrix[:,:,0], cmap=LinearSegmentedColormap('RedGreen', cdict), vmin=-1, vmax=1)
		plt.axis("off")
		plt.savefig(output_dir+'_fig_sh_l'+str(lmax)+'.jpg')
		if showIt:
			plt.show()
		return

	fig, axs = plt.subplots(nrows=rows, ncols=cols, gridspec_kw={'wspace':0.1, 'hspace':-0.4}, squeeze=True, figsize=(16, 8))
	for c in range(0,cols):
		for r in range(0,rows):
			axs[r,c].axis('off')

	for l in range(0,lmax+1):
		nInBand = shTermsWithinBand(l)
		colOffset = int(cols/2) - int(nInBand/2)
		rowOffset = (l*cols)+1
		index = rowOffset+colOffset
		for i in range(0,nInBand):
			axs[l, i+colOffset].axis("off")
			axs[l, i+colOffset].imshow(sh_basis_matrix[:,:,imgIndex], cmap=LinearSegmentedColormap('RedGreen', cdict), vmin=-1, vmax=1)
			imgIndex+=1

	#plt.tight_layout()
	plt.savefig(output_dir+'_fig_sh_l'+str(lmax)+'.jpg')
	if showIt:
		plt.show()

if __name__ == "__main__":
	print("Spherical Harmonics for latitude-longitude radiance maps")

	# Parsing input
	parser = argparse.ArgumentParser(description="Process IBL filename and number of bands.")
	parser.add_argument('--ibl_filename', type=str, default='./images/grace-new.exr',
	                    help="Path to the ibl filename (default: './images/grace-new.exr').")
	parser.add_argument('--lmax', type=int, default=3,
	                    help="Number of bands (must be an integer, default: 3).")
	parser.add_argument('--output_dir', type=str, default='./output/', 
	                    help="Output directory (default: './output/').")
	parser.add_argument('--resize_width', type=int, default=1000, 
	                    help="Width to resize the image (default: 1000).")
	args = parser.parse_args()

	print(f"Input File (IBL): {args.ibl_filename}")
	print(f"Number of Bands (lmax): {args.lmax}")
	print(f"Resize Width: {args.resize_width}")
	print(f"Output Directory: {args.output_dir}")
	print("\n")

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)
	resize_height = int(args.resize_width/2)

	# Visualise the SPH
	print("Plotting spherical harmonic functions...")
	sh_visualise(args.lmax, showIt=False, output_dir=args.output_dir)

	# Read image
	print("Reading in image...")
	ibl_GT = resizeImage(im.imread(args.ibl_filename, 'EXR-FI')[:,:,0:3], args.resize_width, resize_height, cv2.INTER_CUBIC)
	im.imwrite(args.output_dir+'_ibl_GT.exr', ibl_GT.astype(np.float32))

	# SPH projection
	print("Running spherical harmonics...")
	iblCoeffs = getCoefficientsFromFile(args.ibl_filename, args.lmax, resize_width=args.resize_width)
	#iblCoeffs = getCoefficientsFromImage(ibl_GT, args.lmax, resize_width=args.resize_width)
	writeReconstruction(iblCoeffs, args.lmax, '_SPH', width=args.resize_width, output_dir=args.output_dir)
	#shPrint(iblCoeffs)
	shPrintToFile(iblCoeffs)
	print("Spherical harmonics processing is complete.\n")

	# Diffuse convolution diffuse map (ground truth)
	print("Generating ground truth diffuse map for comparison...")
	diffuseLowResWidth = 32 # changing resolution is a trade off between processing time and ground truth quality
	output_width = args.resize_width
	fn = args.output_dir+'_diffuse_ibl_GT_'+str(args.resize_width)+'_'+str(diffuseLowResWidth)+'_'+str(output_width)+'.exr'
	diffuse_ibl_GT = getDiffuseMap(args.ibl_filename, width=args.resize_width, width_low_res=diffuseLowResWidth, output_width=output_width) 
	im.imwrite(fn, diffuse_ibl_GT.astype(np.float32))

	print("Complete")
