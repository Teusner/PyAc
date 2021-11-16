import sys
import numpy as np
import matplotlib.pyplot as plt

import glob
import os

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        raise ValueError("A valid path should be given as parameter !")
        
    folder = os.path.join(os.getcwd(), sys.argv[1], "*.txt")
    A = np.zeros((101, 1000))

    for filename in glob.glob(folder):
        file = open(filename, 'r')
        for line in file:
            if line[0] != '#':
                l = line.split("\t")
                A[int(l[1]), int(l[0])] = float(l[2])

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.set_xlabel(r"x ($m$)")
    ax.set_ylabel(r"y ($m$)")
    ax.set_title(r"Signal module")
    im = ax.imshow(A, aspect="auto")
    fig.colorbar(im, orientation="horizontal")
    plt.show()