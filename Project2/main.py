import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from pandas.plotting import table
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import os
import math
import itertools
import pandas as pd
import sys

from data_perparation import read_landmarks, read_odometry

np.random.seed(19)

if __name__ == "__main__":
    # -------------------- Question 1-------------------- #
    # Read the data
    print("Reading ground truth landmarks")
    trueLandmarks = np.array(read_landmarks("D:\Masters Degree\Courses\Sensors In Autonomous Systems 0510-7951\Homework\HW_3\Landmarks\LastID_1.csv"))

    print("Reading ground truth odometry")
    trueOdometry = read_odometry("D:\Masters Degree\Courses\Sensors In Autonomous Systems 0510-7951\Homework\HW_3\odometry.dat")
    print("Finished")

    # --------------------------------------------------- #
