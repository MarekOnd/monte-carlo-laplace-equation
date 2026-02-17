# Solving 2D Laplace's equation using Monte Carlo method

## Prerequisites

Depending on which implementations are used:
- Make
- Python + packages in requirements.txt
- C++
- CUDA library

## Implementations

### Python

In `src/python/main.py`. Used just as an overview for the actual algorithm in C++.

### C++ sequential

In `src/sequential/main.cpp`.

### CUDA parallel

In `src/cuda/main.cu`. Experimental speedup which moves the boundary (and introduces more errors) is in `src/cuda-experiment/main.cu`.

## Verification

Verification can be performed using `src/verification/main.cpp`. For example, checking the example image at `fig/example-result.png` can be performed with

```
src/verification/main fig/example-result.png
```

## Example result

Result for zero right hand side with sinusoidal boundary condition (`sin(x+y)` where `(x,y)` are the coordinates).

<img src="fig/example-result.png">
