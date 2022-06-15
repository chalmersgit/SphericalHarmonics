# SphericalHarmonics
Spherical harmonics for radiance maps in Python (numpy). 

I wrote this code because I found it was difficult to obtain coefficients for a radiance map from existing libraries (especially in Python).

Features:
- Obtain coefficients for a radiance map (in equirectangular format)
- Numpy vectorised for efficiency
- Windowing function for reducing ringing artefacts
- Reconstruct radiance map from coefficients
- Obtain diffuse BRDF coefficients
- Render a diffuse map (given radiance map coefficients)
- Supports an arbitrary number of bands 
- Plot the spherical harmonics in a figure
- Render a ground truth diffuse map to compare with. 

The ground truth diffuse map can be a little slow to compute, so I've added the ability to render the diffuse values at a low resolution while sampling the high resolution source image. After rendering at a low resolution, I increase the resolution (so it's easier to see) using Lanczos interpolation. I found doing it this way was the most efficient while also producing high quality ground truth images.

# Usage
python sphericalHarmonics.py [string filename.ext] [int nBands]

Example:
python sphericalHarmonics.py radianceMap.exr 2

See the main function to see examples of functions you can utilise in your own code.

# References
- Ramamoorthi, Ravi, and Pat Hanrahan. "An efficient representation for irradiance environment maps", 2001.
- Sloan, Peter-Pike, Jan Kautz, and John Snyder. "Precomputed radiance transfer for real-time rendering in dynamic, low-frequency lighting environments", 2002.
- "Spherical Harmonic Lighting: The Gritty Details" by Robin Green
- "Stupid Spherical Harmonics (SH) Tricks" by Peter Pike Sloan
- PBRT source code: https://www.csie.ntu.edu.tw/~cyy/courses/rendering/pbrt-2.00/html/sh_8cpp_source.html
- Probulator source code (I based my windowing code on this): https://github.com/kayru/Probulator
- Some radiance maps can be downloaded here: http://gl.ict.usc.edu/Data/HighResProbes/

Things TODO if people want it:
- Support other formats (e.g. cubemap, angular map, etc.)
- Change, remove, or support other modules (e.g. I use imageio for reading HDR images, cv2 for resizing, etc.)
- More optimisations
- Other windowing methods
- Other visualisations
- Restructure code
- and more. Feel free to message me.

# Dependencies
You can use pip to install the modules I've used. The only gotcha is with imageio. By default it does not provide OpenEXR support.
To add OpenEXR support to imageio, see here:
https://imageio.readthedocs.io/en/stable/_autosummary/imageio.plugins.freeimage.html#module-imageio.plugins.freeimage

e.g., run the following in python (after installing imageio):
imageio.plugins.freeimage.download()

# License
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
