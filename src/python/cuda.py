# Unfinished, due to performance issues, moved directly to C++ CUDA implementation
import numpy as np
from numba import jit, cuda
import time
import cv2 as cv
import argparse
import numba
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

def f1(x,y):
    return np.sin(x+y)

def f2(x,y):
    return y-x

def f3(x,y):
    return np.sin(5*(x+y))

def cuda_option():
    sizex = 50
    sizey = 50
    steps = 10
    array = np.zeros((sizex,sizey,1),dtype=float)
    boundaries = np.zeros((sizex,sizey,1),dtype=bool)
    d_arr = cuda.to_device(np.zeros((sizex*sizey),dtype=float))

    # get necessary dimension for GPU blocks and grids
    blockdim = 32 # 32 beacause that is the maximum amount of threads that can be given to a core
    griddim = np.ceil(array.shape[0]*array.shape[1] / blockdim).astype(int)

    for i in np.arange(steps):
        rng_states = create_xoroshiro128p_states(sizex*sizey, seed=1)
        # goes through the array as 1d and resizes it afterwards
        monte_carlo_solve[griddim,blockdim](rng_states, array.flatten(),boundaries.flatten(), d_arr)
    result = d_arr.copy_to_host().reshape((sizex,sizey))

    #cv.waitKey(1)
    #cv.imshow("img",result.astype(float))
    frame = cv.cvtColor((np.floor(result.astype(float)*255).astype(dtype='uint8')), cv.COLOR_BGRA2BGR)
    cv.imwrite('result_cuda.jpg', frame)

@cuda.jit
def monte_carlo_solve(states, array_in:np.ndarray, boundaries:np.ndarray, array_out:np.ndarray):
    i = cuda.grid(1)
    if i < array_in.shape[0]:
        z = 0
        index = i
        total_value = 0.
        while not boundaries[index]:
            randx = xoroshiro128p_uniform_float32(states, i)
            move = 1 if randx >= 0.5 else -1
        total_value += array_in[index]
        z+=1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    #cuda_option()