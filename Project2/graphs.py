import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

def plot_trajectory(trueTrajectory, trueLandmarks):
    plt.figure(figsize=(8, 8))
    plt.plot(trueTrajectory[:, 0], trueTrajectory[:, 1])
    plt.scatter(trueLandmarks[:, 0], trueLandmarks[:, 1], s=10, facecolors='none', edgecolors='b')
    plt.grid()
    plt.axis('equal')
    plt.xlabel("X [m]", fontsize=10)
    plt.ylabel("Y [m]", fontsize=10)
    plt.legend(['Ground Truth', 'Landmarks'], prop={"size": 8}, loc="best")
    plt.title('Ground trues trajectory and landmarks', fontsize=10)

def plot_trajectory_and_measured_trajectory(trueTrajectory, measured_trajectory, trueLandmarks):
    plt.figure(figsize=(8, 8))
    plt.plot(trueTrajectory[:, 0], trueTrajectory[:, 1])
    plt.plot(measured_trajectory[:, 0], measured_trajectory[:, 1], color='r')
    plt.scatter(trueLandmarks[:, 0], trueLandmarks[:, 1], s=10, facecolors='none', edgecolors='b')
    plt.grid()
    plt.xlabel("X [m]", fontsize=10)
    plt.ylabel("Y [m]", fontsize=10)
    plt.legend(['Ground truth trajectory', 'Trajectory with gaussian noise in the odometry data', 'Landmarks'],
               prop={"size": 10}, loc="best")
    plt.title('Ground trues trajectory, landmarks and noisy trajectory', fontsize=10)

def draw_pf_frame(trueTrajectory, measured_trajectory, trueLandmarks, particles, title):
    """
    Plots the ground truth and estimated trajectories as well as the landmarks, the particles and their heading
    Parameters:
        trueTrajectory - dim is [num_frames x 3] or [num_frames x 2] (the heading is not used)
        measured_trajectory - dim is [num_frames x 3] or [num_frames x 2] (the heading is not used)
        trueLandmarks - dim is [num_landmarks, 2]
        particles - dim is [number_of_particles, 3]
        title - the title of the graph
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(trueTrajectory[:, 0], trueTrajectory[:, 1])
    ax.scatter(trueLandmarks[:, 0], trueLandmarks[:, 1], s=10, facecolors='none', edgecolors='b')
    line_segments = []
    for particle in particles:
        x = particle[0]
        y = particle[1]
        heading_line_len = 0.5
        endx = x + heading_line_len * np.cos(particle[2])
        endy = y + heading_line_len * np.sin(particle[2])
        line_segments.append(np.array([[x, y], [endx, endy]]))
    line_collection = LineCollection(line_segments, color='c', alpha=0.08)
    ax.scatter(particles[:, 0], particles[:, 1], s=8, facecolors='none', edgecolors='g', alpha=0.7)
    ax.add_collection(line_collection)
    ax.plot(measured_trajectory[:, 0], measured_trajectory[:, 1], color='r')

    ax.grid()
    ax.set_title(title, fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)
    ax.legend(['Ground Truth', 'Particle filter estimated trajectory', 'Landmarks', 'Particles and their heading'], prop={"size": 10}, loc="best")
