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

import data_preparation
import graphs
from ParticlesFilter import ParticlesFilter

np.random.seed(19)

def Q1():
    # Read the data
    print("Reading ground truth landmarks")
    trueLandmarks = np.array(data_preparation.read_landmarks("D:\Masters Degree\Courses\Sensors In Autonomous Systems 0510-7951\Homework\HW_3\Landmarks\LastID_1.csv"))

    print("Reading ground truth odometry")
    trueOdometry = data_preparation.read_odometry("D:\Masters Degree\Courses\Sensors In Autonomous Systems 0510-7951\Homework\HW_3\odometry.dat")

    # Caculate true trajectory
    trueTrajectory = data_preparation.calculate_trajectory(trueOdometry)

    # Plot trajectory
    graphs.plot_trajectory(trueTrajectory, trueLandmarks)

    # Generate measurement odometry
    sigma_r1 = 0.01
    sigma_t = 0.2
    sigma_r2 = 0.01
    measured_trajectory = data_preparation.generate_measurement_odometry(trueOdometry, sigma_r1, sigma_t, sigma_r2)
    graphs.plot_trajectory_and_measured_trajectory(trueTrajectory, measured_trajectory, trueLandmarks)

    # Run Particle Filter
    # Initalize
    sigma_range = 0.1
    sigma_bearing = 0.01
    numberOfParticles = 1000
    pf = ParticlesFilter(trueLandmarks, sigma_r1, sigma_t, sigma_r2, sigma_range, sigma_bearing, numberOfParticles)

    # Run particle filter
    pf.run(trueLandmarks, trueOdometry, trueTrajectory)

    # Plot final frame
    graphs.draw_pf_frame(trueTrajectory, pf.history, trueLandmarks, pf.particles, "Final Frame")

    # Build animation
    plt.close('all')
    pf_animation = graphs.build_animation(trueTrajectory[:, 0:2], pf.history[:, 0:2])
    plt.show()
    print("Question 1 is done")

def Q2():
    print("Question 2 is done")

def Q3():
    print("Question 3 is done")

if __name__ == "__main__":
    Q1()

