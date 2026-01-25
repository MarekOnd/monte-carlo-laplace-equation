import numpy as np
import time
import cv2 as cv
import argparse

def f1(x,y):
    return np.sin(x+y)

def f2(x,y):
    return y-x

def f3(x,y):
    return np.sin(5*(x+y))

def naive_approach(f = f3):
    sizex = 50
    sizey = 50
    steps = 10
    array = np.zeros((sizex,sizey,1),dtype=float)
    boundaries = np.zeros((sizex,sizey,1),dtype=bool)

    for x in np.arange(sizex):
        boundaries[x,sizey-1] = True
        array[x,sizey-1] = f(x/sizex,1)
        boundaries[x,0] = True
        array[x,0] = f(x/sizex,0)
    for y in np.arange(sizey):
        boundaries[sizex-1,y] = True
        array[sizex-1,y] = f(1,y/sizey)
        boundaries[0,y] = True
        array[0,y] = f(0,y/sizey)

    for x in np.arange(sizex):
        for y in np.arange(sizey):
            # Skip if it is a boundary already
            if boundaries[x,y]:
                continue
            for i in np.arange(steps):
                posx, posy = x, y
                while not boundaries[posx,posy]:
                    if np.random.randint(0,2) == 0:
                        posx += 2*np.random.randint(0,2) - 1
                    else:
                        posy += 2*np.random.randint(0,2) - 1
                array[x,y] += array[posx,posy]
            array[x,y] = array[x,y]/steps
    # Can show an output window but sometimes breaks with fonts
    #cv.namedWindow('img',cv.WINDOW_NORMAL)
    #cv.imshow("img",array.astype(float))
    frame = cv.cvtColor((np.floor((array- np.min(array)).astype(float)*200 ).astype(dtype='uint8')), cv.COLOR_BGRA2BGR)
    cv.imwrite('result.jpg', frame)

def perform_sampling(array : np.ndarray, boundaries : np.ndarray, steps : int):
    for x in np.arange(array.shape[0]):
        for y in np.arange(array.shape[1]):
            # Skip if it is a boundary already
            if boundaries[x,y]:
                continue
            for i in np.arange(steps):
                posx, posy = x, y
                while not boundaries[posx,posy]:
                    np.random.BitGenerator
                    if np.random.randint(0,2) == 0:
                        posx += 2*np.random.randint(0,2) - 1
                    else:
                        posy += 2*np.random.randint(0,2) - 1
                array[x,y] += array[posx,posy]
            array[x,y] = array[x,y]/steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    naive_approach()
    

    