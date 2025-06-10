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
Spherical harmonics for radiance maps using numpy.

Assumes:
Equirectangular format.
theta: [0 to pi], from top to bottom row of pixels.
phi: [0 to 2*Pi], from left to right column of pixels.
'''

import os
import sys
import numpy as np
import argparse
import imageio.v3 as im
import cv2  # For resizing images with float support

# Custom
import sh_utilities
import custom_utilities

def google_example_data(direction, width):
	"""
	Example function for cosine lobe calculation. For further details, see Google's implementation at https://github.com/google/spherical-harmonics/blob/master/sh/spherical_harmonics.cc
	const std::vector<double> google = { 0.886227, 0.0, 1.02333, 0.0, 0.0, 0.0, 0.495416, 0.0, 0.0 };

	Args:
		direction (ndarray): Direction vector.
		width (int): Width of the image.

	Returns:
		ndarray: radiance map with pixel values following a cosine lobe

	Reference:
		
	"""
	xyz = sh_utilities.get_cartesian_map(width)
	return custom_utilities.grey_to_colour(np.clip(np.sum(direction * xyz, axis=2), 0.0, 1.0))

def gritty_details_example_data(width):
	"""
	Example function based on "The Gritty Details" by Robin Green, page 17 (note, the paper is missing the first coefficient in the fourth band)
	
	Args:
		width (int): Width of the image.

	Returns:
		ndarray: radiance map with pixel values following gritty details example
	"""

	x = np.arange(0,width)
	y = np.arange(0,width//2).reshape(width//2,1)
	lat_lon = sh_utilities.xy_to_ll(x,y,width,width//2)
	theta = np.repeat(lat_lon[0][:, np.newaxis], width, axis=1).reshape((width//2, width))
	phi = np.repeat(lat_lon[1][np.newaxis, :], width//2, axis=0).reshape((width//2, width))
	return custom_utilities.grey_to_colour(np.maximum(0.0, 5 * np.cos(theta) - 4) +
						  np.maximum(0.0, -4 * np.sin(theta - np.pi) * np.cos(phi - 2.5) - 3))

def run_gritty_details_example():
	"""
	Run the gritty details example and print the SH coefficients.
	"""
	l_max = 3
	width = 2048
	radiance_map_data = gritty_details_example_data(width)

	ibl_coeffs = sh_utilities.get_coefficients_from_image(radiance_map_data, l_max, resize_width=width)
	print("Google's example result:")
	sh_utilities.sh_print(ibl_coeffs)

def run_google_example():
	"""
	Run the Google cosine lobe example and print the SH coefficients.
	"""
	l_max = 2
	width = 2048
	radiance_map_data = google_example_data([0,1,0], width)

	ibl_coeffs = sh_utilities.get_coefficients_from_image(radiance_map_data, l_max, resize_width=width)
	print("Google's example result:")
	sh_utilities.sh_print(ibl_coeffs)


def parse_arguments():
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(description="Process IBL filename and number of bands.")
	parser.add_argument('--ibl_filename', type=str, default='./images/grace-new.exr',
						help="Path to the IBL filename (default: './images/grace-new.exr').")
	parser.add_argument('--l_max', type=int, default=3,
						help="Number of bands (must be an integer, default: 3).")
	parser.add_argument('--output_dir', type=str, default='./output/',
						help="Output directory (default: './output/').")
	parser.add_argument('--resize_width', type=int, default=1000,
						help="Width to resize the image (default: 1000).")
	return parser.parse_args()


def main():
	# Parsing input
	args = parse_arguments()

	print("Spherical Harmonics for latitude-longitude radiance maps")
	print(f"Input File (IBL): {args.ibl_filename}")
	print(f"Number of Bands (l_max): {args.l_max}")
	print(f"Resize Width: {args.resize_width}")
	print(f"Output Directory: {args.output_dir}\n")

	if not os.path.exists(args.output_dir):
		os.makedirs(args.output_dir)

	resize_height = args.resize_width // 2

	# Visualize the spherical harmonic functions
	print("Plotting spherical harmonic functions...")
	sh_utilities.sh_visualise(args.l_max, show=False, output_dir=args.output_dir)

	# Read image
	print("Reading image...")
	radiance_map_data = custom_utilities.resize_image(
		im.imread(args.ibl_filename, plugin='EXR-FI')[:, :, :3],
		args.resize_width,
		resize_height,
		cv2.INTER_CUBIC
	)

	im.imwrite(os.path.join(args.output_dir, '_radiance_map_data.exr'), radiance_map_data.astype(np.float32))
	im.imwrite(os.path.join(args.output_dir, '_radiance_map_data.jpg'), custom_utilities.linear2sRGB(radiance_map_data))

	# SPH projection
	print("Running spherical harmonics...")
	#ibl_coeffs = sh_utilities.get_coefficients_from_file(args.ibl_filename, args.l_max, resize_width=args.resize_width)
	ibl_coeffs = sh_utilities.get_coefficients_from_image(radiance_map_data, args.l_max, resize_width=args.resize_width)
	sh_utilities.write_reconstruction(ibl_coeffs, args.l_max, '_SPH', width=args.resize_width, output_dir=args.output_dir)
	#sh_utilities.sh_print(ibl_coeffs)
	sh_utilities.sh_print_to_file(ibl_coeffs)

	print("Spherical harmonics processing is complete.\n")

	# Generate ground truth diffuse map for comparison
	print("Generating ground truth diffuse map for comparison...")
	diffuse_low_res_width = 32  # Trade-off between processing time and ground truth quality
	output_width = args.resize_width
	diffuse_ibl_gt = custom_utilities.get_roughness_map(
		args.ibl_filename,
		width=args.resize_width,
		width_low_res=diffuse_low_res_width,
		output_width=output_width
	)
	im.imwrite(os.path.join(args.output_dir, f'_diffuse_ibl_gt_{args.resize_width}_{diffuse_low_res_width}_{output_width}.exr'), diffuse_ibl_gt.astype(np.float32))
	im.imwrite(os.path.join(args.output_dir, f'_diffuse_ibl_gt_{args.resize_width}_{diffuse_low_res_width}_{output_width}.jpg'), custom_utilities.linear2sRGB(diffuse_ibl_gt))

	print("Complete.")


if __name__ == "__main__":
	main()
	#run_google_example()
	#run_gritty_details_example()