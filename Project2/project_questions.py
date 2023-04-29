import os
import numpy as np

import data_loader
import graphs
from data_preparation import *
import random

# from utils.misc_tools import error_ellipse
# from utils.ellipse import draw_ellipse
# import matplotlib.pyplot as plt
# from matplotlib import animation
# import graphs
from kalman_filter import KalmanFilter #, ExtendedKalmanFilter, ExtendedKalmanFilterSLAM


random.seed(123)
np.random.seed(123)

# font = {'size': 20}
# plt.rc('font', **font)


class ProjectQuestions:
    def __init__(self, dataset):
        """
        Given a Loaded Kitti data set with the following ground truth values: tti dataset and adds noise to GT-gps values
        - lat: latitude [deg]
        - lon: longitude [deg]
        - yaw: heading [rad]
        - vf: forward velocity parallel to earth-surface [m/s]
        - wz: angular rate around z axis [rad/s]
        Builds the following np arrays:
        - enu - lla converted to enu data
        - times - for each frame, how much time has elapsed from the previous frame
        - yaw_vf_wz - yaw, forward velocity and angular change rate
        - enu_noise - enu with Gaussian noise (sigma_xy=3 meters)
        - yaw_vf_wz_noise - yaw_vf_wz with Gaussian noise in vf (sigma 2.0) and wz (sigma 0.2)
        """
        self.dataset = dataset

        data_oxts = data_loader.DataLoader.get_gps_imu(dataset)
        lats = [i.packet.lat for i in data_oxts]
        lons = [i.packet.lon for i in data_oxts]
        alts = [i.packet.alt for i in data_oxts]
        self.lla = np.array([lats, lons, alts]).T

        self.enu = extract_ENU_from_LLA(lats, lons, alts)

        _, self.times, self.yaw_vf_wz = build_GPS_trajectory(dataset) #TODO (hint- use build_GPS_trajectory)
        self.delta_t = self.times[1] - self.times[0]

        # add noise to the trajectory
        self.sigma_xy = 3 #TODO
        self.sigma_vf = 2 #TODO
        self.sigma_wz = 0.2 #TODO

        num_examples = self.enu.shape[0]

        self.enu_noise = self.enu + np.concatenate([np.random.normal(0, self.sigma_xy, size=(num_examples, 2)), np.zeros((num_examples, 1))], axis=1)

        self.yaw_vf_wz_noise = self.yaw_vf_wz + np.concatenate([np.zeros((num_examples, 1)),
                                                np.random.normal(0, self.sigma_vf, size=(num_examples, 1)),
                                                np.random.normal(0, self.sigma_wz, size=(num_examples, 1))], axis=1)


    def Q1(self):
        """
        That function runs the code of question 1 of the project.
        Load data from the KITTI dataset, add noise to the ground truth GPS values, and apply a Kalman filter to the noisy data (enu).
        """

        # Plot Global coordinates (LLA)
        graphs.plot_trajectory_and_height(self.lla, title1="Latitude - Longtitude GT", xlabel1="Latitude[deg]", ylabel1="Longtitude[deg]", title2="Ältitude GT", xlabel2="Frame number", ylabel2="Altitude[m]")
        # Plot ENU coordinates
        graphs.plot_trajectory_and_height(self.enu, title1="East - North GT", xlabel1="East[m]", ylabel1="North[m]", title2="Up GT", xlabel2="Frame number", ylabel2="Up[m]")

        # Plot ENU coordinates with noise
        graphs.plot_trajectory_with_noise(self.enu, self.enu_noise, title="ENU - GT & Noised", xlabel="x[m]", ylabel="y[m]", legend_gt="GT", legend_noise="Noised")

        # Kalman Filter regular
        sigma_n = 0.8
        kf = KalmanFilter(self.enu_noise[:, 0:2], self.times, self.sigma_xy, sigma_n, self.delta_t, is_dead_reckoning=False)
        enu_kf, cov_mat = kf.run(variance_amp=3)

        # calc_RMSE_maxE(locations_GT, locations_kf)
        RMSE, maxE = kf.calc_RMSE_maxE(self.enu[:, 0:2], enu_kf)
        print('RMSE: ', RMSE, 'maxE: ', maxE)

        # Plot ENU comparison between GT and KalmanFilter
        graphs.plot_trajectory_comparison(self.enu[:, 0:2], enu_kf)

        # Kalman Filter with dead reckoning
        kf_dead_reckon = KalmanFilter(self.enu_noise[:, 0:2], self.times, self.sigma_xy, sigma_n, self.delta_t, is_dead_reckoning=True)
        enu_kf_dead_reckon, cov_mat_dead_reckon = kf_dead_reckon.run(variance_amp=3)
        graphs.plot_trajectory_comparison_dead_reckoning(self.enu[:, 0:2], enu_kf, enu_kf_dead_reckon)

        # animation that shows the covariances of the EKF estimated path
        trajectory_animation = graphs.build_animation(self.enu[:, 0:2], enu_kf_dead_reckon, enu_kf, cov_mat, title="Trajectory Animation", xlabel="East [meters]", ylabel="North [meters]",
                                                      label1="Predicted Trajectory", label2="Dead Reckoning", label0="GT Trajectory")

        # this animation that shows the covariances of the dead reckoning estimated path
        ani_dead_reckon_cov = graphs.build_animation(self.enu[:, 0:2], enu_kf, enu_kf_dead_reckon, cov_mat_dead_reckon, title="Trajectory Animation", xlabel="East [meters]", ylabel="North [meters]",
                                                      label1="Predicted Trajectory", label2="Dead Reckoning", label0="GT Trajectory")

        # Plot the estimated error of x-y values separately and corresponded sigma value along the trajectory
        err_x = self.enu[:, 0] - enu_kf[:, 0]
        err_cov_x = (err_x, np.sqrt(cov_mat[:, 0]))
        err_y = self.enu[:, 1] - enu_kf[:, 1]
        err_cov_y = (err_y, np.sqrt(cov_mat[:, 3]))
        graphs.plot_error(err_cov_x, err_cov_y)

        # Save animation
        is_save_animation = False
        if is_save_animation:
            save_path = "../results/Q1/"
            graphs.save_animation(trajectory_animation, save_path, "Kalman_Filter_Trajectory_Animation_zoomed2")
            graphs.save_animation(ani_dead_reckon_cov, save_path, "dead_reckon_cov_animation")



    def Q2(self):

        """
        That function runs the code of question 2 of the project.
        Load data from the KITTI dataset, add noise to the ground truth GPS values, yaw rate, and velocities, and apply a Kalman filter to the noisy data.
        """



        # plot yaw, yaw rate and forward velocity
        yaw_vec = self.yaw_vf_wz[:, 0]
        fv_vec = self.yaw_vf_wz[:, 1]
        yaw_rate_vec = self.yaw_vf_wz[:, 2]
        graphs.plot_yaw_yaw_rate_fv(yaw_vec, yaw_rate_vec, fv_vec)

        # sigma_samples =

        # sigma_vf, sigma_omega =

        # build_LLA_GPS_trajectory

        # add_gaussian_noise to u and measurments (locations_gt[:,i], sigma_samples[i])

         # plot vf and wz with and without noise

        # ekf = ExtendedKalmanFilter(sigma_samples, sigma_vf, sigma_omega)
        # locations_ekf, sigma_x_xy_yx_y_t = ekf.run(locations_noised, times, yaw_vf_wz_noised, do_only_predict=False)

        # RMSE, maxE = ekf.calc_RMSE_maxE(locations_gt, locations_ekf)

        # print the maxE and RMSE

        # draw the trajectories

        # draw the error

        #v.	Plot the estimated error of x-y-θ values separately and corresponded sigma value along the trajectory



        # build_animation

            # animation that shows the covariances of the EKF estimated path

            # this animation that shows the covariances of the dead reckoning estimated path

        # save_animation(ani, os.path.dirname(__file__), "ekf_predict")


    def get_odometry(self, sensor_data):
        """
        Args:
            sensor_data: map from a tuple (frame number, type) where type is either ‘odometry’ or ‘sensor’.
            Odometry data is given as a map containing values for ‘r1’, ‘t’ and ‘r2’ – the first angle, the translation and the second angle in the odometry model respectively.
            Sensor data is given as a map containing:
              - ‘id’ – a list of landmark ids (starting at 1, like in the landmarks structure)
              - ‘range’ – list of ranges, in order corresponding to the ids
              - ‘bearing’ – list of bearing angles in radians, in order corresponding to the ids

        Returns:
            numpy array of of dim [num of frames X 3]
            first two components in each row are the x and y in meters
            the third component is the heading in radians
        """
        num_frames = len(sensor_data) // 2
        state = np.array([[0, 0, 0]], dtype=float).reshape(1, 3)
        for i in range(num_frames):
            curr_odometry = sensor_data[i, 'odometry']
            t = np.array([
                curr_odometry['t'] * np.cos(state[-1, 2] + curr_odometry['r1']),
                curr_odometry['t'] * np.sin(state[-1, 2] + curr_odometry['r1']),
                curr_odometry['r1'] + curr_odometry['r2']
            ]).reshape(3, 1)
            new_pos = state[-1, :].reshape(3, 1) + t
            state = np.concatenate([state, new_pos.reshape(1, 3)], axis=0)
        return state


    # def Q3(self):
    #
    #     """
    #     Runs the code for question 3 of the project
    #     Loads the odometry (robot motion) and sensor (landmarks) data supplied with the exercise
    #     Adds noise to the odometry data r1, trans and r2
    #     Uses the extended Kalman filter SLAM algorithm with the noisy odometry data to predict the path of the robot and
    #     the landmarks positions
    #     """
    #
    #     #Pre-processing
    #     landmarks = self.dataset.load_landmarks()
    #     sensor_data_gt = self.dataset.load_sensor_data()
    #     state = self.get_odometry(sensor_data_gt)
    #     sigma_x_y_theta = #TODO
    #     variance_r1_t_r2 = #TODO
    #     variance_r_phi = #TODO
    #
    #     sensor_data_noised = add_gaussian_noise_dict(sensor_data_gt, list(np.sqrt(np.array(variance_r1_t_r2))))
    #     # plot trajectory
    #       #TODO
    #     # plot trajectory + noise
    #       #TODO
    #
    #     import matplotlib.pyplot as plt
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111)
    #
    #     # KalmanFilter
    #
    #     ekf_slam = ExtendedKalmanFilterSLAM(sigma_x_y_theta, variance_r1_t_r2, variance_r_phi)
    #
    #     frames, mu_arr, mu_arr_gt, sigma_x_y_t_px1_py1_px2_py2 = ekf_slam.run(sensor_data_gt, sensor_data_noised, landmarks, ax)
    #
    #     #RMSE, maxE =calc_RMSE_maxE
    #
    #
    #     # draw the error for x, y and theta
    #
    #     # Plot the estimated error ofof x-y-θ -#landmark values separately and corresponded sigma value along the trajectory
    #
    #     # draw the error
    #
    #
    #     graphs.plot_single_graph(mu_arr_gt[:,0] - mu_arr[:,0], "x-$x_n$", "frame", "error", "x-$x_n$",
    #                              is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,0]))
    #     graphs.plot_single_graph(mu_arr_gt[:,1] - mu_arr[:,1], "y-$y_n$", "frame", "error", "y-$y_n$",
    #                              is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,1]))
    #     graphs.plot_single_graph(normalize_angles_array(mu_arr_gt[:,2] - mu_arr[:,2]), "$\\theta-\\theta_n$",
    #                              "frame", "error", "$\\theta-\\theta_n$",
    #                              is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,2]))
    #
    #     graphs.plot_single_graph((np.tile(landmarks[1][0], mu_arr.shape[0]) - mu_arr[:,3]),
    #                              "landmark 1: x-$x_n$", "frame", "error [m]", "x-$x_n$",
    #                              is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,3]))
    #     graphs.plot_single_graph((np.tile(landmarks[1][1], mu_arr.shape[0]) - mu_arr[:,4]),
    #                              "landmark 1: y-$y_n$", "frame", "error [m]", "y-$y_n$",
    #                              is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,4]))
    #
    #     graphs.plot_single_graph((np.tile(landmarks[2][0], mu_arr.shape[0]) - mu_arr[:,5]),
    #                              "landmark 2: x-$x_n$", "frame", "error [m]", "x-$x_n$",
    #                              is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,5]))
    #     graphs.plot_single_graph((np.tile(landmarks[2][1], mu_arr.shape[0]) - mu_arr[:,6]),
    #                              "landmark 2: y-$y_n$", "frame", "error [m]", "y-$y_n$",
    #                              is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,6]))
    #
    #     ax.set_xlim([-2, 12])
    #     ax.set_ylim([-2, 12])
    #
    #     ani = animation.ArtistAnimation(fig, frames, repeat=False)
    #     graphs.show_graphs()
    #     # ani.save('im.mp4', metadata={'artist':'me'})

    def run(self):
        # self.Q1()
        self.Q2()
        # self.Q3()


