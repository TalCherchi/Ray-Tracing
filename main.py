
import ray_tracer
import argparse
import sys
import numpy as np
import os
from typing import Dict, Any
from PIL import Image
import time
NDArray = Any


def main(arr):
    if(len(sys.argv)<3 or len(sys.argv)>5 ):
        print("invalid input")
        return 0
    args=[]
    args.append(sys.argv[1])
    args.append(sys.argv[2])
    if(len(sys.argv)==3):
        args.append(500)
        args.append(500)
    elif(len(sys.argv)==4):
        args.append(sys.argv[3])
        args.append(500)
    else:
        args.append(sys.argv[3])
        args.append(sys.argv[4])


    path_name = sys.argv[2]

    if len(path_name) < 4:
        path_name = path_name + ".png"
    elif (path_name[len(path_name)-4:] != ".png"):
        path_name = path_name + ".png"
    im=ray_tracer.parseScene(args)
    #call ray tarcer
    pic=Image.fromarray(im.astype('uint8'))
    pic.save(path_name)
    
    


if __name__ == '__main__': 
    main(sys.argv)
    
