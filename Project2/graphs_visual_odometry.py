import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Ellipse
import pymap3d as pm
import matplotlib as mpl
import cv2


def plot_odometry_gt_and_noisy(sensor_data_gt, sensor_data_noised, save_fig_path="results\\", fig_name="fig.png"):
    sensor_data_count = int(len(sensor_data_noised))
    mu_arr_gt = np.array([[0, 0, 0]])
    mu_arr_noisy = np.array([[0, 0, 0]])
    for idx in range(sensor_data_count):
        # ----- ground-truth odometry -------
        delta_r1_gt = sensor_data_gt[idx]["r1"]
        delta_r2_gt = sensor_data_gt[idx]["r2"]
        delta_trans_gt = sensor_data_gt[idx]["t"]
        calc_x_gt = lambda theta_p: delta_trans_gt * np.cos(theta_p + delta_r1_gt)
        calc_y_gt = lambda theta_p: delta_trans_gt * np.sin(theta_p + delta_r1_gt)
        theta_gt = delta_r1_gt + delta_r2_gt
        theta_prev_gt = mu_arr_gt[-1, 2]
        mu_arr_gt = np.vstack(
            (mu_arr_gt, mu_arr_gt[-1] + np.array([calc_x_gt(theta_prev_gt), calc_y_gt(theta_prev_gt), theta_gt])))

        # ---- noised odometry ------
        delta_r1_noised = sensor_data_noised[idx]["r1"]
        delta_r2_noised = sensor_data_noised[idx]["r2"]
        delta_trans_noised = sensor_data_noised[idx]["t"]

        calc_x = lambda theta_p: delta_trans_noised * np.cos(theta_p + delta_r1_noised)
        calc_y = lambda theta_p: delta_trans_noised * np.sin(theta_p + delta_r1_noised)
        theta = delta_r1_noised + delta_r2_noised

        theta_prev = mu_arr_noisy[-1, 2]
        mu_arr_noisy = np.vstack(
            (mu_arr_noisy, mu_arr_noisy[-1] + np.array([calc_x(theta_prev), calc_y(theta_prev), theta])))

    fig, ax = plt.subplots()
    fig.suptitle('Odometry Trajectory Ground-Truth vs Noised')
    ax.plot(mu_arr_gt[:, 0], mu_arr_gt[:, 1])
    ax.plot(mu_arr_noisy[:, 0], mu_arr_noisy[:, 1])
    ax.set_xlabel('X[m]')
    ax.set_ylabel('Y[m]')
    ax.set_aspect('equal')
    ax.legend(['ground-truth', 'noised'], loc='right')
    fig.savefig(os.path.join(save_fig_path, fig_name), transparent=True)


def plot_single_graph(X_Y, title, xlabel, ylabel, label, is_scatter=False, sigma=None, save_fig=False,
                      save_fig_path="results\\", fig_name="fig.png"):
    """
    That function plots a single graph

    Args:
        X_Y (np.ndarray): array of values X and Y, array shape [N, 2]
        title (str): sets figure title
        xlabel (str): sets xlabel value
        ylabel (str): sets ylabel value
        label (str): sets legend's label value
    """
    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if is_scatter:
        ax.scatter(np.arange(X_Y.shape[0]), X_Y, s=1, label=label, c="b")
        if sigma is not None:
            ax.plot(np.arange(X_Y.shape[0]), sigma, c="orange")
            ax.plot(np.arange(X_Y.shape[0]), -sigma, c="orange")
    elif len(X_Y.shape) == 1:
        ax.plot(np.arange(X_Y.shape[0]), X_Y, label=label)
    else:
        ax.plot(X_Y[:, 0], X_Y[:, 1], label=label)

    ax.legend()

    if (save_fig):
        fig.savefig(os.path.join(save_fig_path, fig_name), transparent=True)


def plot_graph_and_scatter(X_Y0, X_Y1, title, xlabel, ylabel, label0, label1, color0='b', color1='r', point_size=1,
                           save_fig=False, save_fig_path="results\\", fig_name="fig.png"):
    """
    That function plots two graphs, plot and scatter

    Args:
        X_Y0 (np.ndarray): array of values X and Y, array shape [N, 2] of graph 0
        X_Y1 (np.ndarray): array of values X and Y, array shape [N, 2] of graph 1
        title (str): sets figure title
        xlabel (str): sets xlabel value
        ylabel (str): sets ylabel value
        label0 (str): sets legend's label value of graph 0
        label1 (str): sets legend's label value of graph 1
        color0 (str): color of graph0
        color1 (str): color of graph1
        point_size(float): size of scatter points
    """
    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(X_Y0[:, 0], X_Y0[:, 1], label=label0, c=color0)
    ax.scatter(X_Y1[:, 0], X_Y1[:, 1], label=label1, s=point_size, c=color1)
    ax.legend()
    if (save_fig):
        fig.savefig(os.path.join(save_fig_path, fig_name), transparent=True)


def plot_four_graphs(X_values, Y0, Y1, Y2, Y3, title, xlabel, ylabel, label0, label1, label2, label3):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(X_values, Y0, label=label0)
    plt.plot(X_values, Y1, label=label1)
    plt.plot(X_values, Y2, label=label2)
    plt.plot(X_values, Y3, label=label3)
    plt.legend()


def plot_three_graphs(X_Y0, X_Y1, X_Y2, title, xlabel, ylabel, label0, label1, label2, save_fig=False,
                      save_fig_path="results\\", fig_name="fig.png"):
    fig, ax = plt.subplots()
    fig.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(X_Y0[:, 0], X_Y0[:, 1], 'g:', label=label0)
    ax.plot(X_Y1[:, 0], X_Y1[:, 1], 'b-', label=label1)
    ax.plot(X_Y2[:, 0], X_Y2[:, 1], 'r--', label=label2)
    ax.legend()
    if (save_fig):
        fig.savefig(os.path.join(save_fig_path, fig_name), transparent=True)


def build_animation(X_Y0, X_Y1, images, title, xlabel, ylabel, label0, label1):
    frames = []
    images_iter = np.nditer(images)
    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 2)
    imageAx = fig.add_subplot(2, 1, 1)
    print("Creating animation")

    x0, y0, x1, y1 = [], [], [], []
    val0, = plt.plot([], [], 'b:', animated=True, label=label0)
    val1, = plt.plot([], [], 'r-', animated=True, label=label1)
    img = imageAx.imshow(images[0, :, :, :])

    plt.legend()

    values = np.hstack((X_Y0, X_Y1))

    def init():
        # ax.set_xlim(-50, 300)
        # ax.set_ylim(-10, 500)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        val0.set_data([], [])
        val1.set_data([], [])
        # img.set_data(data)
        return val0, val1

    def update(frame, *fargs):
        images_itr = fargs[0]
        x0.append(frame[0])
        y0.append(frame[1])
        x1.append(frame[2])
        y1.append(frame[3])

        val0.set_data(x0, y0)
        val1.set_data(x1, y1)
        img.set_data(images_itr[0])
        images_itr.iternext()
        return val0, val1, img

    anim = animation.FuncAnimation(fig, update, frames=values, init_func=init, interval=1, repeat=False, blit=True,
                                   fargs=(images_iter))
    return anim


def save_animation(ani, basedir, file_name):
    print("Saving animation")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(os.path.join(basedir, f'{file_name}.mp4'), writer=writer)
    print("Animation saved")


def show_graphs():
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib import animation

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    e1 = Ellipse(xy=(0.5, 0.5), width=0.5, height=0.2, angle=60, animated=True)
    e2 = Ellipse(xy=(0.8, 0.8), width=0.5, height=0.2, angle=100, animated=True)
    ax.add_patch(e1)
    ax.add_patch(e2)


    def init():
        return [e1, e2]


    def animate(i):
        e1.angle = e1.angle + 0.5
        e2.angle = e2.angle + 0.5
        return e1, e2


    anim = animation.FuncAnimation(fig, animate, init_func=init, interval=1, blit=True)
    plt.show()