# Robust Adaptive Denoising of Color Images with Mixed Gaussian and Impulsive Noise

This software implements the Robust Adaptive Denoising algoritm. Code is written in CUDA.

# Requirements
- CUDA SDK
- libpng

# Installation
Run:

```
make lib CUDA=1
make main CUDA=1
```

# Usage
Sample usage:
`./main_rob  <variant> <reference image { rgb }> <noisy image {rgb}> <block_radius> <patch_radius> <alpha> <h> <sigma> 
where:

variant = Algorihm variant
Reference image - original, not noisy image
Noisy image - image which will be filtered
block_radius - radius of the processing block B
patch_radius - radius of the patch
alpha - number of pixels taked into account { positive }
sigma - h prameter { positive }
sigmai - sigma prameter { positive }

Running:
`./main_rob  <variant> <reference image { rgb }> <noisy image {rgb}> 

will use the Self-tunning approach.


# Acknowledgment

This code uses parts of the Fourier 0.8 library by Emre Celebi licensed on GPL avalaible here:

http://sourceforge.net/projects/fourier-ipal

http://www.lsus.edu/faculty/~ecelebi/fourier.htm

