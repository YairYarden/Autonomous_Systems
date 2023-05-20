import data_loader
import graphs
from data_preparation import *
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from kalman_filter import KalmanFilter, ExtendedKalmanFilter, ExtendedKalmanFilterSLAM
from kalman_filter import KalmanFilterConstAcc

random.seed(123)
np.random.seed(123)

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
        self.sigma_xy = 3
        self.sigma_vf = 2 # 2
        self.sigma_wz = 0.2 # 0.2
        self.sigma_theta = 1

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

        is_const_acc = False
        if(is_const_acc):
            # Kalman filter Const Acc
            sigma_n = 0.35
            kf = KalmanFilterConstAcc(self.enu_noise[:, 0:2], self.times, self.sigma_xy, sigma_n, self.delta_t, is_dead_reckoning=False)
            enu_kf, cov_mat = kf.run(variance_amp=3)
        else:
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

        # animation that shows the covariances of the KF estimated path
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

        # Compute containment percent
        containment_percent_x, containment_percent_y = kf.compute_containment_percent(err_cov_x, err_cov_y)
        print('Containment percent X : ', containment_percent_x)
        print('Containment percent Y : ', containment_percent_y)

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
        vf_vec = self.yaw_vf_wz[:, 1]
        yaw_rate_vec = self.yaw_vf_wz[:, 2]
        graphs.plot_yaw_yaw_rate_fv(yaw_vec, yaw_rate_vec, vf_vec)

        # plot vf and wz with and without noise
        graphs.plot_vf_wz_with_and_without_noise(self.yaw_vf_wz, self.yaw_vf_wz_noise)

        ekf = ExtendedKalmanFilter(self.enu_noise, self.yaw_vf_wz_noise, self.times, self.sigma_xy, self.sigma_theta, self.sigma_vf, self.sigma_wz, 3, False)
        enu_ekf, yaw_ekf, cov_mat = ekf.run()

        # RMSE, maxE
        RMSE, maxE = ekf.calc_RMSE_maxE(self.enu[:, 0:2], enu_ekf)
        print('RMSE: ', RMSE, 'maxE: ', maxE)

        # Compare to KF
        kf = KalmanFilter(self.enu_noise[:, 0:2], self.times, self.sigma_xy, 0.8, self.delta_t, is_dead_reckoning=False)
        enu_kf, _ = kf.run(variance_amp=3)
        graphs.plot_three_graphs(self.enu[:, 0:2], enu_ekf, enu_kf, 'Trajectory Comparison KF vs EKF', 'East[meters]', 'North[meters]','GT Trajectory', 'EKF Trajectory', 'KF Trajectory')

        # draw the trajectories
        graphs.plot_trajectory_comparison(self.enu[:, 0:2], enu_ekf)

        # dead reckoning
        ekf_dead_reckon = ExtendedKalmanFilter(self.enu_noise, self.yaw_vf_wz_noise, self.times, self.sigma_xy, self.sigma_theta,
                                               self.sigma_vf, self.sigma_wz, 3, True)
        enu_ekf_dead_reckon, yaw_ekf_dead_reckon, cov_mat_dead_reckon = ekf_dead_reckon.run()
        graphs.plot_trajectory_comparison_dead_reckoning(self.enu[:, 0:2], enu_ekf, enu_ekf_dead_reckon)

        #v.	Plot the estimated error of x-y-θ values separately and corresponded sigma value along the trajectory
        err_x = self.enu[:, 0] - enu_ekf[:, 0]
        err_cov_x = (err_x, np.sqrt(cov_mat[:, 0]))
        err_y = self.enu[:, 1] - enu_ekf[:, 1]
        err_cov_y = (err_y, np.sqrt(cov_mat[:, 3]))
        err_yaw = self.yaw_vf_wz[:, 0] - yaw_ekf
        err_cov_yaw = (err_yaw, np.sqrt(cov_mat[:, 4]))
        graphs.plot_error(err_cov_x, err_cov_y, err_cov_yaw)

        # Compute containment percent
        containment_percent_x, containment_percent_y, containment_percent_yaw = ekf.compute_containment_percent(err_cov_x, err_cov_y, err_cov_yaw)
        print('Containment percent X : ', containment_percent_x)
        print('Containment percent Y : ', containment_percent_y)
        print('Containment percent Yaw : ', containment_percent_yaw)

        # EKF estimated path animation
        trajectory_animation = graphs.build_animation(self.enu[:, 0:2], enu_ekf_dead_reckon, enu_ekf, cov_mat[:, 0:4], title="Trajectory Animation", xlabel="East [meters]",
                                                      ylabel="North [meters]", label1="Dead Reckoning", label2="Predicted Trajectory", label0="GT Trajectory")

        # EKF estimated path animation with reckon dead
        ani_dead_reckon_cov = graphs.build_animation(self.enu[:, 0:2], enu_ekf, enu_ekf_dead_reckon, cov_mat_dead_reckon[:, 0:4], title="Trajectory Animation",
                                                     xlabel="East [meters]", ylabel="North [meters]", label1="Predicted Trajectory", label2="Dead Reckoning", label0="GT Trajectory")

        # Save animation
        is_save_animation = False
        if is_save_animation:
            save_path = "../results/Q2/"
            graphs.save_animation(trajectory_animation, save_path, "EKF_trajectory_animation")
            graphs.save_animation(ani_dead_reckon_cov, save_path, "EKF_trajectory_animation_dead_reckon_cov")

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


    def Q3(self):

        """
        Runs the code for question 3 of the project
        Loads the odometry (robot motion) and sensor (landmarks) data supplied with the exercise
        Adds noise to the odometry data r1, trans and r2
        Uses the extended Kalman filter SLAM algorithm with the noisy odometry data to predict the path of the robot and
        the landmarks positions
        """

        #Pre-processing
        landmarks = self.dataset.load_landmarks()
        sensor_data_gt = self.dataset.load_sensor_data()
        state = self.get_odometry(sensor_data_gt)

        variance_r1_t_r2 = [0.01**2, 0.1**2, 0.01**2]
        variance_r_phi = [0.118**2, 0.118**2]
        # variance_r_phi = [0.3**2, 0.0035**2]
        # sigma_x_y_theta = np.array([variance_r1_t_r2[1], variance_r1_t_r2[1], variance_r1_t_r2[0] + variance_r1_t_r2[2]])
        sigma_x_y_theta = np.array([0, 0, 0])

        # Add noise
        sensor_data_noised = add_gaussian_noise_dict(sensor_data_gt, list(np.sqrt(np.array(variance_r1_t_r2))))
        state_noised = self.get_odometry(sensor_data_noised)

        # plot trajectory + noise
        graphs.plot_trajectory_with_noise(state, state_noised, 'Noised Trajectory', 'x', 'y', 'Trajectory GT', 'Noised Trajectory')

        # KalmanFilter
        ekf_slam = ExtendedKalmanFilterSLAM(sigma_x_y_theta, variance_r1_t_r2, variance_r_phi)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        frames, mu_arr, mu_arr_gt, sigma_x_y_t_px1_py1_px2_py2 = ekf_slam.run(sensor_data_gt, sensor_data_noised, landmarks, ax)

        RMSE, maxE = ekf_slam.calc_RMSE_maxE(mu_arr_gt[:, 0:2], mu_arr[:, 0:2])
        print('RMSE: ', RMSE, 'maxE: ', maxE)

        # draw the error for x, y and theta

        # Plot the estimated error ofof x-y-θ -#landmark values separately and corresponded sigma value along the trajectory

        # draw the error

        graphs.plot_single_graph(mu_arr_gt[:,0] - mu_arr[:,0], "x-$x_n$", "frame", "error", "x-$x_n$",
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,0]))
        graphs.plot_single_graph(mu_arr_gt[:,1] - mu_arr[:,1], "y-$y_n$", "frame", "error", "y-$y_n$",
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,1]))
        graphs.plot_single_graph(normalize_angles_array(mu_arr_gt[:,2] - mu_arr[:,2]), "$\\theta-\\theta_n$",
                                 "frame", "error", "$\\theta-\\theta_n$",
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,2]))

        graphs.plot_single_graph((np.tile(landmarks[1][0], mu_arr.shape[0]) - mu_arr[:,3]),
                                 "landmark 1: x-$x_n$", "frame", "error [m]", "x-$x_n$",
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,3]))
        graphs.plot_single_graph((np.tile(landmarks[1][1], mu_arr.shape[0]) - mu_arr[:,4]),
                                 "landmark 1: y-$y_n$", "frame", "error [m]", "y-$y_n$",
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,4]))

        graphs.plot_single_graph((np.tile(landmarks[2][0], mu_arr.shape[0]) - mu_arr[:,5]),
                                 "landmark 2: x-$x_n$", "frame", "error [m]", "x-$x_n$",
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,5]))
        graphs.plot_single_graph((np.tile(landmarks[2][1], mu_arr.shape[0]) - mu_arr[:,6]),
                                 "landmark 2: y-$y_n$", "frame", "error [m]", "y-$y_n$",
                                 is_scatter=True, sigma=np.sqrt(sigma_x_y_t_px1_py1_px2_py2[:,6]))

        ax.set_xlim([-2, 12])
        ax.set_ylim([-2, 12])

        # Show Animation
        ani = animation.ArtistAnimation(fig, frames, repeat=False)
        graphs.show_graphs()

        # Save animation
        is_save_animation = False
        if is_save_animation:
            save_path = "../results/Q3/"
            graphs.save_animation(ani, save_path, "SLAM_animation2")

        # ani.save('SLAM_animation.mp4', metadata={'artist':'me'})

    def run(self):
        # self.Q1()
        self.Q2()
        # self.Q3()


