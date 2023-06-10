import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.animation as animation
import os
import plotly.graph_objects as go
# ------------------- PART 1 - PARTICLE FILTER ------------------- #
def plot_trajectory(trueTrajectory, trueLandmarks):
    plt.figure(figsize=(8, 8))
    plt.plot(trueTrajectory[:, 0], trueTrajectory[:, 1])
    plt.scatter(trueLandmarks[:, 0], trueLandmarks[:, 1], s=10, facecolors='none', edgecolors='b')
    plt.grid()
    plt.axis('equal')
    plt.xlabel("X [m]", fontsize=10)
    plt.ylabel("Y [m]", fontsize=10)
    plt.legend(['Ground Truth', 'Landmarks'], prop={"size": 8}, loc="best")
    plt.title('Ground truth trajectory and landmarks', fontsize=10)

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
    plt.title('Ground truth trajectory, landmarks and noisy trajectory', fontsize=10)

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


def build_animation(true_trajectory, estimated_trajectory, particles_history, landmarks, num_frames,
                    label1="True Trajectory", label2="Estimated Trajectory", label3="Particles", label4="Landmarks"):
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
    scatter3 = ax.scatter([], [], label=label3, color='g')
    scatter4 = ax.scatter([], [], label=label4, color='blue', facecolors='none')
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
        x4 = landmarks[:, 0]
        y4 = landmarks[:, 1]

        # Update scatter plot data
        line1.set_data(x1, y1)
        line2.set_data(x2, y2)
        scatter3.set_offsets(np.column_stack((x3, y3)))
        scatter4.set_offsets(np.column_stack((x4, y4)))

        # Set title and frame number
        ax.set_title(f"Frame: {frame + 1}/{num_frames}")

        return line1, line2, scatter3, scatter4

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100)
    ax.legend()
    ax.grid(True)
    return ani

def save_animation(ani, basedir, file_name):
    print("Saving animation")
    ani.save(os.path.join(basedir, f'{file_name}.gif'), writer='pillow', fps=10)
    print("Animation saved")

# ------------------- PART 2 - ICP ------------------- #
palette = ['#FF1493', '#afd6fa']
def plot_errors(errors, title):
    plt.figure()
    plt.title("ICP Error of {} vs Iterations".format(title))
    plt.plot(range(1, 1+len(errors)), errors)
    plt.xlabel("Iteration [a.u.]")
    plt.ylabel("Error [m]")


def visualize_clouds(clouds, fname):

    data = []
    for i, cloud in enumerate(clouds):
        xdata = cloud[:, 0]
        ydata = cloud[:, 1]
        zdata = cloud[:, 2]

        data.append(go.Scatter3d(x=xdata, y=ydata, z=zdata,
            mode='markers',
            marker=dict(
              size=1,
              color=palette[i],
              opacity=1.0
          ))
        )

      # Decide bounds
    mega_cloud = np.vstack(clouds)
    mega_centroid = np.average(mega_cloud, axis=0)
    mega_max = np.amax(mega_cloud, axis=0)
    mega_min = np.amin(mega_cloud, axis=0)
    lower_bound = mega_centroid - (np.amax(mega_max - mega_min) / 2)
    upper_bound = mega_centroid + (np.amax(mega_max - mega_min) / 2)

    # Setup layout
    grid_lines_color = 'rgb(127, 127, 127)'
    layout = go.Layout(
      scene=dict(
          xaxis=dict(nticks=8,
                  range=[lower_bound[0], upper_bound[0]],
                  showbackground=True,
                  backgroundcolor='rgb(30, 30, 30)',
                  gridcolor=grid_lines_color,
                  zerolinecolor=grid_lines_color),
          yaxis=dict(nticks=8,
                  range=[lower_bound[1], upper_bound[1]],
                  showbackground=True,
                  backgroundcolor='rgb(30, 30, 30)',
                  gridcolor=grid_lines_color,
                  zerolinecolor=grid_lines_color),
          zaxis=dict(nticks=8,
                  range=[lower_bound[2], upper_bound[2]],
                  showbackground=True,
                  backgroundcolor='rgb(30, 30, 30)',
                  gridcolor=grid_lines_color,
                  zerolinecolor=grid_lines_color),
          xaxis_title="x (meters)",
          yaxis_title="y (meters)",
          zaxis_title="z (meters)"
      ),
      margin=dict(r=10, l=10, b=10, t=10),
      paper_bgcolor='rgb(30, 30, 30)',
      font=dict(
          family="Courier New, monospace",
          color=grid_lines_color
      ),
      legend=dict(
          font=dict(
              family="Courier New, monospace",
              color='rgb(127, 127, 127)'
          )
      )
    )

    fig = go.Figure(data=data, layout=layout)
    fig.show()

def visualize_clouds_animation(pc_A, pc_B_list, fname, speed=100):
    # Setup data
    iteration_labels = [str(i) for i in range(len(pc_B_list))]
    all_clouds = []
    frames = []
    num_samples = min(pc_A.shape[0], pc_B_list[0].shape[0])
    subset_indices = np.random.choice(num_samples,
                                int(num_samples / (len(pc_B_list) * 0.5)),
                                replace=False)
    for i, (label, pc_B) in enumerate(zip(iteration_labels, pc_B_list)):
        data = []
        clouds = [pc_A, pc_B]
        for j, pc in enumerate(clouds):
            xdata = pc[subset_indices, 0]
            ydata = pc[subset_indices, 1]
            zdata = pc[subset_indices, 2]

            data.append(go.Scatter3d(x=xdata, y=ydata, z=zdata,
                mode='markers',
                marker=dict(
                    size=1,
                    color=palette[j],
                    opacity=1.0
                ))
            )
        frames.append(go.Frame(name=label, data=data))
        all_clouds += clouds
        if i == 0:
            first_data = data

    # Decide bounds
    mega_cloud = np.vstack(all_clouds)
    mega_centroid = np.average(mega_cloud, axis=1)
    mega_max = np.amax(mega_cloud, axis=1)
    mega_min = np.amin(mega_cloud, axis=1)
    lower_bound = mega_centroid - (np.amax(mega_max - mega_min) / 2)
    upper_bound = mega_centroid + (np.amax(mega_max - mega_min) / 2)

    # Setup layout
    grid_lines_color = 'rgb(127, 127, 127)'
    layout = go.Layout(
      scene=dict(
          xaxis=dict(nticks=8,
                  range=[lower_bound[0], upper_bound[0]],
                  showbackground=True,
                  backgroundcolor='rgb(30, 30, 30)',
                  gridcolor=grid_lines_color,
                  zerolinecolor=grid_lines_color),
          yaxis=dict(nticks=8,
                  range=[lower_bound[1], upper_bound[1]],
                  showbackground=True,
                  backgroundcolor='rgb(30, 30, 30)',
                  gridcolor=grid_lines_color,
                  zerolinecolor=grid_lines_color),
          zaxis=dict(nticks=8,
                  range=[lower_bound[2], upper_bound[2]],
                  showbackground=True,
                  backgroundcolor='rgb(30, 30, 30)',
                  gridcolor=grid_lines_color,
                  zerolinecolor=grid_lines_color),
          xaxis_title="x (meters)",
          yaxis_title="y (meters)",
          zaxis_title="z (meters)"
      ),
      hovermode=False,
      margin=dict(r=10, l=10, b=10, t=10),
      paper_bgcolor='rgb(30, 30, 30)',
      font=dict(
          family="Courier New, monospace",
          color=grid_lines_color
      ),
      sliders=[dict(
          active=0,
          yanchor="top",
          xanchor="left",
          currentvalue=dict(
              font=dict(
                  family="Courier New, monospace",
                  color='rgb(30, 30, 30)'
              ),
              prefix="Step ",
              visible=True,
              xanchor="right"
          ),
          pad=dict(b=10, t=0),
          len=0.95,
          x=0.05,
          y=0,
          font=dict(
              family="Courier New, monospace",
              color='rgb(127, 127, 127)'
          ),
          steps=[dict(
                  args=[[iteration_label], dict(
                      frame=dict(duration=speed),
                      mode="immediate"
                  )],
                  label=iteration_label,
                  method="animate") for iteration_label in iteration_labels]
      )],
      updatemenus=[dict(
          buttons=[dict(
              args=[None, dict(
                  frame=dict(duration=speed),
                  fromcurrent=True
              )],
              label="Play",
              method="animate"
          )],
          direction="left",
          pad=dict(r=10, t=30),
          showactive=False,
          type="buttons",
          x=0.05,
          xanchor="right",
          y=0,
          yanchor="top",
          font=dict(
              family="Courier New, monospace",
              color='rgb(127, 127, 127)'
          )
      )],
      legend=dict(
          font=dict(
              family="Courier New, monospace",
              color='rgb(127, 127, 127)'
          )
      )
    )

    fig = go.Figure(data=first_data, layout=layout, frames=frames)
    fig.show()

def show_results(i, errors, pc_A, pc_B_list,R,T, title):
    # Print results
    print("ICP of {} run for {} iterations".format(title, i))
    print("Final Error is: {:.3f} [m]".format(errors[-1]))
    # show R+T
    print("ICP results: R:{} , T: {}".format(R,T))

    # Plot errors
    plot_errors(errors, title)

    # show animation
    visualize_clouds_animation(pc_A, pc_B_list, title)

def show_frame(Image, pc, frame_name):
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].set_title("Frame " + frame_name)
    axs[0].imshow(Image)
    axs[1].scatter(pc[:, 0], pc[:, 1], c=-pc[:, 2], marker='.', s=0.5)
    axs[1].scatter(0, 0, c='r', marker='x')
    axs[1].set_title("Frame " + frame_name + " -BEV")
    axs[1].set_xlim(-30, 30)
    axs[1].set_ylim(-30, 30)
    axs[1].axis('scaled')
    axs[1].grid()
    axs[1].set_xlabel('x[m]')
    axs[1].set_ylabel('y[m]')
    plt.show()

def icp_analysis(errors_kd, errors_kd_f, errors_knn, elapsed_time_kdree, elapsed_time_knn):
    # print final errors
    print("Final error of K-D Tree is: {} [m^2]".format(errors_kd[-1]))
    print("Final error of K-D Tree with filtered PC is: {} [m^2]".format(errors_kd_f[-1]))
    print("Final error of K-Nearest Neighbors with filtered PC is: {} [m^2]".format(errors_knn[-1]))

    # Time consuming
    print("Elapsed time kdree: {:.4f} seconds".format(elapsed_time_kdree))
    print("Elapsed time knn: {:.4f} seconds".format(elapsed_time_knn))

    # compare errors on a graph
    plt.figure()
    plt.title("ICP Errors of different methods vs Iterations")
    plt.plot(range(1, 1 + len(errors_kd)), errors_kd, '-rx', markevery=[len(errors_kd) - 1], label='Full PC, K-D Tree')
    plt.plot(range(1, 1 + len(errors_kd_f)), errors_kd_f, '-bx', markevery=[len(errors_kd_f) - 1],
             label='Filtered PC, K-D Tree')
    plt.plot(range(1, 1 + len(errors_knn)), errors_knn, '-kx', markevery=[len(errors_knn) - 1],
             label='Filtered PC, K-Nearest Neighbors')
    plt.xlabel("Iterations [a.u.]")
    plt.ylabel("Error [m]")
    plt.grid()
    plt.legend()

    # sqrt of errors
    plt.figure()
    plt.title("ICP Errors of different methods Vs Iterations")
    plt.plot(range(1, 1 + len(errors_kd)), np.sqrt(errors_kd), '-rx', markevery=[len(errors_kd) - 1],
             label='Full PC, K-D Tree')
    plt.plot(range(1, 1 + len(errors_kd_f)), np.sqrt(errors_kd_f), '-bx', markevery=[len(errors_kd_f) - 1],
             label='Filtered PC, K-D Tree')
    plt.plot(range(1, 1 + len(errors_knn)), np.sqrt(errors_knn), '-kx', markevery=[len(errors_knn) - 1],
             label='Filtered PC, K-Nearest Neighbors')
    plt.xlabel("Iterations [a.u.]")
    plt.ylabel("Error [m]")
    plt.grid()
    plt.legend()

