import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.animation as animation
import os

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
    ax.legend(['Ground Truth', 'Landmarks', 'Particles and their heading', 'Particle filter estimated trajectory'], prop={"size": 10}, loc="best")


def build_animation(true_trajectory, estimated_trajectory, particles_history, num_frames, label1="True Trajectory", label2="Estimated Trajectory", label3="Particles"):
    # Create figure and axis
    fig, ax = plt.subplots()

    # Set axis limits based on the arrays' data
    # ax.set_xlim(np.min([np.min(true_trajectory[:, 0]), np.min(estimated_trajectory[:, 0]), np.min(particles_history[:, :, 0])]),
    #             np.max([np.max(true_trajectory[:, 0]), np.max(estimated_trajectory[:, 0]), np.max(particles_history[:, :, 0])]))
    # ax.set_ylim(np.min([np.min(true_trajectory[:, 1]), np.min(estimated_trajectory[:, 1]), np.min(particles_history[:, :, 1])]),
    #             np.max([np.max(true_trajectory[:, 1]), np.max(estimated_trajectory[:, 1]), np.max(particles_history[:, :, 1])]))

    ax.set_xlim(-5, 11)
    ax.set_ylim(-5, 12)

    # Initialize empty scatter plots
    line1, = ax.plot([], [], linewidth=2, label=label1, color='b')
    line2, = ax.plot([], [], label=label2, color='r')
    scatter3 = ax.scatter([], [], label=label1, color='g')
    x1, y1, x2, y2 = [], [], [], []

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    # Animation update function
    def update(frame):
        # Get x, y locations for the current frame
        x1.append(true_trajectory[frame, 0])
        y1.append(true_trajectory[frame, 1])
        x2.append(estimated_trajectory[frame, 0])
        y2.append(estimated_trajectory[frame, 1])
        x3 = particles_history[frame, :, 0]
        y3 = particles_history[frame, :, 1]

        # Update scatter plot data
        line1.set_data(x1, y1)
        line2.set_data(x2, y2)
        scatter3.set_offsets(np.column_stack((x3, y3)))

        # Set title and frame number
        ax.set_title(f"Frame: {frame + 1}/{num_frames}")

        return line1, line2, scatter3

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100)
    ax.legend()
    ax.grid(True)
    return ani

def save_animation(ani, basedir, file_name):
    print("Saving animation")
    ani.save(os.path.join(basedir, f'{file_name}.gif'), writer='pillow', fps=50)
    print("Animation saved")