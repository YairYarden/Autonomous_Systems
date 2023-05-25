import matplotlib.pyplot as plt

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