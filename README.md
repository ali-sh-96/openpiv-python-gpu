# OpenPIV-Python-GPU
Algorithms for particle image velocimetry (PIV) on a GPU.

[![DOI](https://zenodo.org/badge/774692247.svg)](https://zenodo.org/doi/10.5281/zenodo.10846418)

[openpiv-python](https://github.com/OpenPIV/openpiv-python) consists of Python modules for performing PIV analysis on a set of image pairs. [openpiv-python-gpu](https://github.com/ali-sh-96/openpiv-python-gpu) is a GPU implementation of the same algorithms as [openpiv-python-cpu](https://github.com/ali-sh-96/openpiv-python-cpu), depending only on CuPy for GPU acceleration. The objective of this project was to reduce the PIV computation time and maintain compatibility with the CPU-based version.

## Related projects
- [openpiv-python-cpu](https://github.com/ali-sh-96/openpiv-python-cpu)
  An alternative implementation of the same algorithms for use with a central processing unit (CPU).

- [openptv-python-gpu](https://github.com/ali-sh-96/openptv-python-gpu)
  GPU-accelerated two-dimensional particle tracking velocimetry (PTV), based on a relaxation scheme.

## Warning
OpenPIV-Python is currently under active development, which means it might contain some bugs, and its API is subject to change. The algorithms have been tested on both Windows (work station and laptops) and Linux (Google Colab).

## Installation
First, install CuPy based on your CUDA Toolkit version. For CUDA Toolkit versions 11.2 ~ 11.8 use:

    pip install cupy-cuda11x

For CUDA Toolkit versions 12.x use:

    pip install cupy-cuda12x

Then, use the following command to clone the repository:

    git clone https://github.com/ali-sh-96/openpiv-python-gpu

Finally, add the directory of the cloned repository to your PYTHONPATH:

    import sys
    openpiv_gpu_path = "Path to where you cloned the repository"
    sys.path.append(openpiv_gpu_path)

## Documentation
The OpenPIV documentation is readily accessible on the project's webpage at https://openpiv.readthedocs.org. For information on how to use the modules, see the tutorial notebooks below. Also see [openpiv-python-cpu](https://github.com/ali-sh-96/openpiv-python-cpu) for more tutorials on stitching and masking in PIV. 

## Tutorials
- [Basic tutorial](https://colab.research.google.com/github/ali-sh-96/openpiv-python-gpu/blob/main/openpiv_gpu/tutorials/openpiv_python_gpu_tutorial.ipynb)
- [Batching tutorial](https://colab.research.google.com/github/ali-sh-96/openpiv-python-gpu/blob/main/openpiv_gpu/tutorials/openpiv_python_gpu_batching_tutorial.ipynb)
- [Advanced tutorial](https://colab.research.google.com/github/ali-sh-96/openpiv-python-gpu/blob/main/openpiv_gpu/tutorials/openpiv_python_gpu_advanced_tutorial.ipynb)

## Contributors
1. [OpenPIV team](https://groups.google.com/forum/#!forum/openpiv-users)
2. [Alex Liberzon](https://github.com/alexlib)
3. [Ali Shirinzad](https://github.com/ali-sh-96)
4. [Pierre E. Sullivan](https://turbulence.mie.utoronto.ca/members/sullivan/)

Copyright statement: `gpu_smoothn.py` is the GPU accelerated version of `cpu_smoothn.py`, which it self is a Python version of `smoothn.m` originally created by
[D. Garcia](https://de.mathworks.com/matlabcentral/fileexchange/25634-smoothn), written by Prof. Lewis, and available on
[GitHub](https://github.com/profLewis/geogg122/blob/master/Chapter5_Interpolation/python/smoothn.py). We are thankful to the original authors for
releasing their work as an open source. OpenPIV license does not relate to this code. Please communicate with the
authors regarding their license.

## CUDA Toolkit installation on Windows
Installing CuPy requires CUDA Toolkit. First, make sure to get the latest supported NVIDIA drivers (https://www.nvidia.com/Download/index.aspx) and install Visual Studio (https://visualstudio.microsoft.com/). After Visual Studio is installed, you may install CUDA Toolkit (https://developer.nvidia.com/cuda-downloads).

## How to cite this work
Shirinzad, A., Jaber, K., Xu, K., & Sullivan, P. E. (2023). An Enhanced Python-Based Open-Source Particle Image Velocimetry Software for Use with Central Processing Units. Fluids, 8(11), 285. https://doi.org/10.3390/fluids8110285