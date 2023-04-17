# README

This program is used to verify Google's Sycamore quantum 18 and 20 cycle circuits. As the largest intermediate tensor in this project is 32GB, a GPU with at least 80GB of memory, such as an A100-80GB, is required.

## Directory/File

1. demo.py: main program file
2. run.sh: execution script
3. bashrc: environment variables
4. data: 

    a. scheme_n53_m18.pt: 18-cycle contraction path file

    b. scheme_n53_m20.pt: 20-cycle contraction path file

    c. new_equations_m18.txt: new equations after reordering  for 18-cycle

    d. new_equations_m20.txt: new equations after reordering  for 20-cycle
    
5. cutensor_python: the code comes from cudaLibrarySamples/cuTensor, and the code has been modified to adapt to the calculation of einsum transferred to GEMM
6. results: directory for storing result tensors

## Dependencies

1. PyTorch image [nvidia pytorch:22.12-py3](http://nvcr.io/nvidia/pytorch:22.12-py3)
2. cuTensor v1.6.1: [cuTENSOR Download Page](https://developer.nvidia.com/cutensor-archive)

## Install

1. Fill in the relevant paths in bashrc and then `source bashrc`
2. Install cutensor_python: `sh run.sh -i`

## Run

1. 18-cycle circuit verification
    
    Modify `N_CYCLE` to `18` in `run.sh`
    
2. 20-cycle circuit verification
    
    Modify `N_CYCLE` to `20` in `run.sh`
    

Single A100: `sh run.sh`

DGX-A100: `sh run.sh -mg`