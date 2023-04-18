# README

This program is used to verify Google's Sycamore quantum 18-cycle and 20-cycle circuits. As the largest intermediate tensor in this project is 32GB, a GPU with at least 80GB of memory, such as an A100-80GB, is required.

## Directory/File

1. demo.py: main program file
2. run.sh: execution script
3. bashrc: environment variables
4. data: 

    a. new_equations_m18.txt: new equations after reordering  for 18-cycle

    b. new_equations_m20.txt: new equations after reordering  for 20-cycle
    
5. cutensor_python: the code is taken from [cudaLibrarySamples/cuTensor](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuTENSOR), and has been modified to adapt to the calculation of einsum transferred to GEMM.
6. results: directory for storing result tensors

## Dependencies

1. PyTorch, you could use the container: [nvidia pytorch:22.12-py3](http://nvcr.io/nvidia/pytorch:22.12-py3) or manually install it following the [offical guidance](https://pytorch.org/).
2. cuTensor v1.6.1: [cuTENSOR Download Page](https://developer.nvidia.com/cutensor-archive)

## Download path file

Download the contraction path files for the 18-cycle and 20-cycle, and place them in the `data` directory.
1. 18-cycle contraction path file: [Google Drive link](https://drive.google.com/file/d/1J1k9bwMo2X_lRhM3v2eOEFJpc8LmRvX-/view?usp=sharing), md5sum check: `14e82ef441f8fa799cb807b0b75b3591`
2. 20-cycle contraction path file: [Google Drive link](https://drive.google.com/file/d/1izZySGP9INMwpzWkv7jM_3ZdKuF_FzTN/view?usp=sharing), md5sum check: `5870cdfeddd3026a5a6ba344c98dc294`

## Install

1. Fill in the relevant paths in bashrc and then `source bashrc`
2. Install cutensor_python: `sh run.sh -i`

## Run

1. 18-cycle: modify `N_CYCLE` to `18` in `run.sh`
2. 20-cycle: modify `N_CYCLE` to `20` in `run.sh`
    

Single A100: `sh run.sh`

DGX-A100: `sh run.sh -mg`