import numpy as np
import pykitti
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

import ICP
import time

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
    sigma_bearing = 0.1
    numberOfParticles = 1000
    pf = ParticlesFilter(trueLandmarks, sigma_r1, sigma_t, sigma_r2, sigma_range, sigma_bearing, numberOfParticles)

    # Run particle filter
    pf.run(trueLandmarks, trueOdometry, trueTrajectory)

    # Plot final frame
    graphs.draw_pf_frame(trueTrajectory, pf.history, trueLandmarks, pf.particles, "Final Frame")

    # Build animation
    plt.close('all')
    num_frames = len(trueOdometry)
    pf_animation = graphs.build_animation(trueTrajectory[:, 0:2], pf.history[:, 0:2], pf.particles_history[:, :, 0:2], num_frames)
    plt.show()

    # Save animation
    is_save_animation = False
    if is_save_animation:
        save_path = "../results/Q1/"
        graphs.save_animation(pf_animation, save_path, "Particle_filter_animation")

    print("Question 1 is done")

def Q2():
    # ------------------------------- #
    # Load Data
    basedir = r'D:\\Masters Degree\\Courses\\Sensors In Autonomous Systems 0510-7951\\Homework\\course_ex1\\raw_data\\'
    date = '2011_09_26'
    drive = '0117'
    frame_A = 5
    frame_B = 9
    data = pykitti.raw(basedir, date, drive)
    pc_A = data_preparation.get_pc(data, frame_A)
    pc_B = data_preparation.get_pc(data, frame_B)
    Image_A = data_preparation.load_image(data, frame_A)
    Image_B = data_preparation.load_image(data, frame_B)
    # ------------------------------- #
    # Visualize the point clouds
    # clouds = [pc_A, pc_B]  # TODO
    # graphs.visualize_clouds(clouds, 'FullP- Before_ICPC')
    # graphs.show_frame(Image_A, pc_A, 'A')
    # graphs.show_frame(Image_B, pc_B, 'B')
    # ------------------------------- #
    # KDTree Analysis
    # indices, _, _ = ICP.assign_closest_pairs_kdtree(pc_A, pc_B)  # TODO
    # pc_target = pc_A
    # pc_source = pc_B[indices, :]
    # # choose the point clouds to visualize, by inserting to a list
    # clouds = [pc_B, pc_source]
    # graphs.visualize_clouds(clouds, 'Full_PC_NearestNeighbors')
    # ------------------------------- #
    # Run vanilla ICP on full point cloud
    # num_iters_kd, errors_kd, pc_B_kd, final_R, final_t = ICP.icp(pc_A, pc_B, ICP.assign_closest_pairs_kdtree)
    # graphs.show_results(num_iters_kd, errors_kd, pc_A, pc_B_kd, final_R, final_t, 'FullPC')
    # # show final results
    # clouds = [pc_A, pc_B_kd[-1]]
    # graphs.visualize_clouds(clouds, 'ICP-Results')
    # ------------------------------- #
    # Filter the point cloud above minimal height
    pc_f_A = ICP.filter_pc(pc_A)
    pc_f_B = ICP.filter_pc(pc_B)
    # choose the point clouds to visualize, by inserting to a list
    clouds = [pc_f_A, pc_f_B]
    graphs.visualize_clouds(clouds, 'FilteredPC- before_icp')
    # ------------------------------- #
    # Run ICP with the filtered point clouds
    max_dist = 10
    indices, tree, dist = ICP.assign_closest_pairs_kdtree(pc_f_A, pc_f_B)
    dist = dist.squeeze()
    pc_target_f = pc_f_A
    pc_source_f = pc_f_B[indices, :]
    clouds = [pc_f_B, pc_source_f]
    graphs.visualize_clouds(clouds, 'FilterPC-NearestNeighbors')
    # ------------------------------- #
    # Run icp with K-D Tree function as the data associator
    start_time = time.time()
    # Run ICP
    num_iters_kd_f, errors_kd_f, pc_B_kd_f, final_R_f, final_t_f = ICP.icp(pc_f_A, pc_f_B, ICP.assign_closest_pairs_kdtree)
    # Stop the timer
    end_time = time.time()
    # Calculate the elapsed time
    elapsed_time_kdree = end_time - start_time
    print("Elapsed time: {:.4f} seconds".format(elapsed_time_kdree))
    graphs.show_results(num_iters_kd_f, errors_kd_f, pc_f_A, pc_B_kd_f, final_R_f, final_t_f, 'with K-D Tree, filtered Point Cloud')
    # show final results
    clouds = [pc_A, pc_B_kd_f[-1]]
    graphs.visualize_clouds(clouds, 'Filter_PC_ICP-Results')
    # ------------------------------- #

    # ------------------------------- #

    print("Question 2 is done")

def Q3():
    print("Question 3 is done")

if __name__ == "__main__":
    # Q1()
    Q2()
