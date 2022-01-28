# Object recognition and vision - Topic B
## Self-supervised learning methods for low level vision - Super-Resolution and Deblurring

The codes for the project are available in the noise2noise folder. To run the codes the instructions are the same as the one provided by the noise2noise Readme from the official implementation. The new types of noise that we added are
- gaussian [modified]: instead of adding gaussian noise to the image this now multiplies the Fourier transform of the image with 1-mean complex gaussian numbers.
- super: Super-resolution implements the architecture described in the report (the code may be outdated)
- gblur: this applies the convolve2convolve method described in the report.
- fftblur: this applies the DC-removal method described in the report.
