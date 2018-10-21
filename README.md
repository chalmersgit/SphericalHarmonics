# SphericalHarmonics
Spherical harmonics for radiance maps in python (numpy). 

I wrote this code because I found it was difficult to obtain coefficients for a radiance map (especially in Python).

It supports an arbitrary number of band.

The operations are vectorised (numpy) so it's pretty efficient.

I've also added code for obtaining diffuse BRDF coefficients as well as rendering code which, given radiance map coefficients, will render out a diffuse map. 

I've also added code for computing a ground truth diffuse map, so you can compare the spherical harmonics reconstruction with the ground truth. The ground truth can be a little slow, so I've added the ability to render the diffuse values at a low resolution while sampling the high resolution source image. After rendering at a low resolution, it up samples using Lanczos interpolation. I found doing it this way was the most efficient while also producing high quality ground truth images.

There's also a visualise function which plots the spherical harmonics.

Usage:
python sphericalHarmonics.py [string filename.ext] [int nBands]

Example:
python sphericalHarmonics.py radianceMap.exr 2

See the main function to see examples of functions you can utilise in your own code.
