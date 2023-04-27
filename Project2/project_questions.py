import os
import numpy as np
from data_preparation import *
from utils.misc_tools import error_ellipse
from utils.ellipse import draw_ellipse
import matplotlib.pyplot as plt
from matplotlib import animation
import graphs
import random

from kalman_filter import KalmanFilter, ExtendedKalmanFilter, ExtendedKalmanFilterSLAM
import graphs
import random
from utils.misc_tools import error_ellipse
from utils.ellipse import draw_ellipse
import matplotlib.pyplot as plt
from matplotlib import animation


random.seed(#TODO)
np.random.seed(#TODO)

font = {'size': 20}
plt.rc('font', **font)


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
        self.enu, self.times, self.yaw_vf_wz = #TODO (hint- use build_GPS_trajectory)

        # add noise to the trajectory
        self.sigma_xy = #TODO
        self.sigma_vf = #TODO
        self.sigma_wz = #TODO

        self.enu_noise = #TODO
        self.yaw_vf_wz_noise=#TODO
    
    def Q1(self):
        """
        That function runs the code of question 1 of the project.
        Load data from the KITTI dataset, add noise to the ground truth GPS values, and apply a Kalman filter to the noisy data (enu).
        """
	
	    # "TODO":	        

        # sigma_x_y, sigma_Qn = 
        
        # build_ENU_from_GPS_trajectory
        
        # KalmanFilter
     
        # calc_RMSE_maxE(locations_GT, locations_kf)

        # build_animation (hint- graphs.build_animation)

            # animation that shows the covariances of the EKF estimated path

            # this animation that shows the covariances of the dead reckoning estimated path

        # Plot the estimated error of x-y values separately and corresponded sigma value along the trajectory

        # save_animation(ani, os.path.dirname(__file__), "ekf_predict")

        

    def Q2(self):

        """
        That function runs the code of question 2 of the project.
        Load data from the KITTI dataset, add noise to the ground truth GPS values, yaw rate, and velocities, and apply a Kalman filter to the noisy data.
        """

        # plot yaw, yaw rate and forward velocity
        
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
        sigma_x_y_theta = #TODO
        variance_r1_t_r2 = #TODO
        variance_r_phi = #TODO

        sensor_data_noised = add_gaussian_noise_dict(sensor_data_gt, list(np.sqrt(np.array(variance_r1_t_r2))))
        # plot trajectory 
          #TODO  
        # plot trajectory + noise
          #TODO
        
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        # KalmanFilter

        ekf_slam = ExtendedKalmanFilterSLAM(sigma_x_y_theta, variance_r1_t_r2, variance_r_phi)
        
        frames, mu_arr, mu_arr_gt, sigma_x_y_t_px1_py1_px2_py2 = ekf_slam.run(sensor_data_gt, sensor_data_noised, landmarks, ax)

        #RMSE, maxE =calc_RMSE_maxE
        
        
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
        
        ani = animation.ArtistAnimation(fig, frames, repeat=False)
        graphs.show_graphs()
        # ani.save('im.mp4', metadata={'artist':'me'})
    
    def run(self):
        self.Q3()
        
        
