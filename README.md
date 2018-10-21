# SphericalHarmonics
Spherical harmonics for radiance maps in Python (numpy). 

I wrote this code because I found it was difficult to obtain coefficients for a radiance map (especially in Python).

Features:
- Numpy vectorised for efficiency
- Obtain coefficients for a radiance map (in equirectangular format)
- Windowing function to reduce negative values
- Obtain diffuse BRDF coefficients
- Render a diffuse map (given radiance map coefficients)
- Supports an arbitrary number of bands 
- Plot the spherical harmonics in a figure
- Render a ground truth radiance map to compare with. 

The ground truth can be a little slow to compute, so I've added the ability to render the diffuse values at a low resolution while sampling the high resolution source image. After rendering at a low resolution, I increase the resolution (so it's easier to see) using Lanczos interpolation. I found doing it this way was the most efficient while also producing high quality ground truth images.

Usage:
python sphericalHarmonics.py [string filename.ext] [int nBands]

Example:
python sphericalHarmonics.py radianceMap.exr 2

See the main function to see examples of functions you can utilise in your own code.

References:
- Ramamoorthi, Ravi, and Pat Hanrahan. "An efficient representation for irradiance environment maps", 2001.
- Sloan, Peter-Pike, Jan Kautz, and John Snyder. "Precomputed radiance transfer for real-time rendering in dynamic, low-frequency lighting environments", 2002.
- "Spherical Harmonic Lighting: The Gritty Details" by Robin Green
- "Stupid Spherical Harmonics (SH) Tricks" by Peter Pike Sloan
- PBRT source code: https://www.csie.ntu.edu.tw/~cyy/courses/rendering/pbrt-2.00/html/sh_8cpp_source.html
- Probulator source code (I based my windowing code on this): https://github.com/kayru/Probulator
- Some radiance maps can be downloaded here: http://gl.ict.usc.edu/Data/HighResProbes/

Things TODO if people want it (and if I can be bothered):
- Support other formats (e.g. cubemap, angular map, etc.)
- Change, remove, or support other modules (e.g. I use imageio for reading HDR images, cv2 for resizing, etc.)
- More optimisations
- Other windowing methods
- Other visualisations
- Restructure the code so it's more organised (e.g. use classes, have it possible to install via pip, etc.)
- Whatever else you can think of

