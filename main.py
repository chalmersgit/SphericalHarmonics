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
import argparse
import imageio.v3 as im
import cv2 # resize images with float support

# Custom
import spherical_harmonics as sh
import sh_utility
import utility

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

if __name__ == "__main__":
	# Parsing input
	parser = argparse.ArgumentParser(description="Process IBL filename and number of bands.")
	parser.add_argument('--ibl_filename', type=str, default='./images/grace-new.exr',
	                    help="Path to the ibl filename (default: './images/grace-new.exr').")
	parser.add_argument('--l_max', type=int, default=3,
	                    help="Number of bands (must be an integer, default: 3).")
	parser.add_argument('--output_dir', type=str, default='./output/', 
	                    help="Output directory (default: './output/').")
	parser.add_argument('--resize_width', type=int, default=1000, 
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
	sh_utility.sh_visualise(args.l_max, show=False, output_dir=args.output_dir)

	# Read image
	print("Reading in image...")
	ibl_GT = utility.resize_image(im.imread(args.ibl_filename, plugin='EXR-FI')[:,:,0:3], args.resize_width, resize_height, cv2.INTER_CUBIC)
	im.imwrite(args.output_dir+'_ibl_GT.exr', ibl_GT.astype(np.float32))

	# SPH projection
	print("Running spherical harmonics...")
	ibl_coeffs = sh_utility.get_coefficients_from_file(args.ibl_filename, args.l_max, resize_width=args.resize_width)
	#ibl_coeffs = get_coefficients_from_image(ibl_GT, args.l_max, resize_width=args.resize_width)
	sh_utility.write_reconstruction(ibl_coeffs, args.l_max, '_SPH', width=args.resize_width, output_dir=args.output_dir)
	#sh_print(ibl_coeffs)
	sh_utility.sh_print_to_file(ibl_coeffs)
	print("Spherical harmonics processing is complete.\n")

	# Diffuse convolution diffuse map (ground truth)
	print("Generating ground truth diffuse map for comparison...")
	diffuse_low_res_width = 32 # changing resolution is a trade off between processing time and ground truth quality
	output_width = args.resize_width
	fn = args.output_dir+'_diffuse_ibl_gt_'+str(args.resize_width)+'_'+str(diffuse_low_res_width)+'_'+str(output_width)+'.exr'
	diffuse_ibl_gt = utility.get_diffuse_map(args.ibl_filename, width=args.resize_width, width_low_res=diffuse_low_res_width, output_width=output_width) 
	im.imwrite(fn, diffuse_ibl_gt.astype(np.float32))

	print("Complete.")
