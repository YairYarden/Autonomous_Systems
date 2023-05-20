import math

import numpy as np
import matplotlib.pyplot as plt
from utils.plot_state import plot_state
from data_preparation import normalize_angle, normalize_angles_array


class KalmanFilter:
    """
    class for the implementation of Kalman filter
    """

    def __init__(self, enu_noise, times, sigma_xy, sigma_n, delta_t=1, is_dead_reckoning=False):
        """
        Args:
            enu_noise: enu data with noise
            times: elapsed time in seconds from the first timestamp in the sequence
            sigma_xy: sigma in the x and y axis as provided in the question
            sigma_n: hyperparameter used to fine tune the filter
            is_dead_reckoning: should dead reckoning be applied after 5.0 seconds when applying the filter
        """
        self.enu_noise = enu_noise
        self.times = times
        self.sigma_xy = sigma_xy
        self.sigma_n = sigma_n
        self.is_dead_reckoning = is_dead_reckoning

        # Constant velocity model
        # Matrices notation as in the lecture
        # Coordinates : [x,y, vx, vy] : 4x1
        self.A = np.eye(4, dtype=float)
        self.A[0, 2] = delta_t
        self.A[1, 3] = delta_t

        self.B = np.zeros([4, 4], dtype=float)

        # Estimating only x,y
        self.C = np.zeros([2, 4], dtype=float)
        self.C[0, 0] = 1
        self.C[1, 1] = 1

        self.Q = (sigma_xy ** 2) * np.eye(2, dtype=float)

        self.R = np.zeros([4, 4], dtype=float)
        self.R[2, 2] = delta_t * (sigma_n ** 2)
        self.R[3, 3] = delta_t * (sigma_n ** 2)

    @staticmethod
    def calc_RMSE_maxE(X_Y_GT, X_Y_est):
        """
        That function calculates RMSE and maxE

        Args:
            X_Y_GT (np.ndarray): ground truth values of x and y
            X_Y_est (np.ndarray): estimated values of x and y

        Returns:
            (float, float): RMSE, maxE
        """

        converge_delay = 100
        RMSE = np.sqrt(np.mean((X_Y_GT[converge_delay:] - X_Y_est[converge_delay:]) ** 2))
        maxE = np.max(np.sum(np.abs(X_Y_GT[converge_delay:] - X_Y_est[converge_delay:]), axis=1))

        return RMSE, maxE

    @staticmethod
    def compute_containment_percent(err_cov_x, err_cov_y):
        err_x, cov_x = err_cov_x
        err_y, cov_y = err_cov_y

        count_x = np.count_nonzero(np.abs(err_x) <= cov_x)
        count_y = np.count_nonzero(np.abs(err_y) <= cov_y)

        containment_percent_x = (count_x / len(err_x)) * 100
        containment_percent_y = (count_y / len(err_y)) * 100

        return containment_percent_x, containment_percent_y

    def kalman_filter_iter(self, prev_mu, prev_sigma, z, curr_time):
        """
        Args:
            prev_mu: previous mu
            prev_sigma: previous sigma
            z: measurement
            curr_time: current time

        Returns:
            mu, Sigma
        """

        # Prediction
        mu_pred = np.dot(self.A, prev_mu)  # B is zero cause of constant velocity
        sigma_pred = np.dot(self.A, np.dot(prev_sigma, self.A.T)) + self.R

        # Kalman Gain
        inv_mat = np.linalg.inv(np.dot(self.C, np.dot(sigma_pred, self.C.T)) + self.Q)
        if not (self.is_dead_reckoning) or curr_time < 5.0:
            K = np.dot(sigma_pred, np.dot(self.C.T, inv_mat))
        else:
            K = np.zeros([4, 2], dtype=float)

        # Correction
        z = z[0:2, :]
        mu = mu_pred + np.dot(K, (z - np.dot(self.C, mu_pred)))
        sigma = np.dot((np.eye(4, dtype=float) - np.dot(K, self.C)), sigma_pred)

        return mu, sigma

    def run(self, variance_amp=3):
        """
        Runs the Kalman filter

        outputs: enu_kf, cov_mat
        """
        # Initialize
        enu_kf = np.zeros(self.enu_noise[:, 0:2].shape)  # x,y
        prev_mu = np.zeros([4, 1], dtype=float)
        prev_sigma = np.zeros([4, 4], dtype=float)
        prev_sigma[0, 0] = variance_amp * (self.sigma_xy ** 2)
        prev_sigma[1, 1] = variance_amp * (self.sigma_xy ** 2)

        cov_mat = np.zeros([enu_kf.shape[0], 4])
        num_frames = enu_kf.shape[0]
        for iFrame in range(num_frames):
            curr_enu = self.enu_noise[iFrame, 0:2].T[np.newaxis]
            z = np.vstack((curr_enu.T, np.zeros([2, 1], dtype=float)))
            mu_t, sigma_t = self.kalman_filter_iter(prev_mu, prev_sigma, z, self.times[iFrame])
            enu_kf[iFrame, :] = mu_t[0:2, 0]
            cov_mat[iFrame, :] = sigma_t[0, 0], sigma_t[0, 1], sigma_t[1, 0], sigma_t[1, 1]

            # Update
            prev_mu = mu_t
            prev_sigma = sigma_t

        return enu_kf, cov_mat
# ------------------------------------------------------------------------------------------------- #
class KalmanFilterConstAcc:
    """
    class for the implementation of Kalman filter
    """

    def __init__(self, enu_noise, times, sigma_xy, sigma_n, delta_t=1, is_dead_reckoning=False):
        """
        Args:
            enu_noise: enu data with noise
            times: elapsed time in seconds from the first timestamp in the sequence
            sigma_xy: sigma in the x and y axis as provided in the question
            sigma_n: hyperparameter used to fine tune the filter
            is_dead_reckoning: should dead reckoning be applied after 5.0 seconds when applying the filter
        """
        self.enu_noise = enu_noise
        self.times = times
        self.sigma_xy = sigma_xy
        self.sigma_n = sigma_n
        self.is_dead_reckoning = is_dead_reckoning

        # Constant acceleration model
        # Coordinates : [x,y, vx, vy, ax, ay] : 6x1
        self.A = np.eye(6, dtype=float)
        self.A[0, 2] = delta_t
        self.A[0, 4] = 0.5 * (delta_t ** 2)
        self.A[1, 3] = delta_t
        self.A[1, 5] = 0.5 * (delta_t ** 2)
        self.A[2, 4] = delta_t
        self.A[3, 5] = delta_t

        # Constant velocity model
        self.B = np.zeros([6, 6], dtype=float)

        # Estimating only x,y
        self.C = np.zeros([2, 6], dtype=float)
        self.C[0, 0] = 1
        self.C[1, 1] = 1

        self.Q = (sigma_xy ** 2) * np.eye(2, dtype=float)

        self.R = np.zeros([6, 6], dtype=float)
        self.R[4, 4] = delta_t * (sigma_n ** 2)
        self.R[5, 5] = delta_t * (sigma_n ** 2)


    @staticmethod
    def calc_RMSE_maxE(X_Y_GT, X_Y_est):
        """
        That function calculates RMSE and maxE

        Args:
            X_Y_GT (np.ndarray): ground truth values of x and y
            X_Y_est (np.ndarray): estimated values of x and y

        Returns:
            (float, float): RMSE, maxE
        """

        converge_delay = 100
        RMSE = np.sqrt(np.mean((X_Y_GT[converge_delay:] - X_Y_est[converge_delay:]) ** 2))
        maxE = np.max(np.sum(np.abs(X_Y_GT[converge_delay:] - X_Y_est[converge_delay:]), axis=1))

        return RMSE, maxE

    @staticmethod
    def compute_containment_percent(err_cov_x, err_cov_y):
        err_x, cov_x = err_cov_x
        err_y, cov_y = err_cov_y

        count_x = np.count_nonzero(np.abs(err_x) <= cov_x)
        count_y = np.count_nonzero(np.abs(err_y) <= cov_y)

        containment_percent_x = (count_x / len(err_x)) * 100
        containment_percent_y = (count_y / len(err_y)) * 100

        return containment_percent_x, containment_percent_y

    def kalman_filter_iter(self, prev_mu, prev_sigma, z, curr_time):
        """
        Args:
            prev_mu: previous mu
            prev_sigma: previous sigma
            z: measurement
            curr_time: current time

        Returns:
            mu, Sigma
        """

        # Prediction
        mu_pred = np.dot(self.A, prev_mu)  # B is zero cause of constant velocity
        sigma_pred = np.dot(self.A, np.dot(prev_sigma, self.A.T)) + self.R

        # Kalman Gain
        inv_mat = np.linalg.inv(np.dot(self.C, np.dot(sigma_pred, self.C.T)) + self.Q)
        if not (self.is_dead_reckoning) or curr_time < 5.0:
            K = np.dot(sigma_pred, np.dot(self.C.T, inv_mat))
        else:
            K = np.zeros([6, 2], dtype=float)

        # Correction
        z = z[0:2, :]
        mu = mu_pred + np.dot(K, (z - np.dot(self.C, mu_pred)))
        sigma = np.dot((np.eye(6, dtype=float) - np.dot(K, self.C)), sigma_pred)

        return mu, sigma

    def run(self, variance_amp=3):
        """
        Runs the Kalman filter

        outputs: enu_kf, cov_mat
        """
        # Initialize
        enu_kf = np.zeros(self.enu_noise[:, 0:2].shape)  # x,y
        prev_mu = np.zeros([6, 1], dtype=float)
        prev_sigma = np.zeros([6, 6], dtype=float)
        prev_sigma[0, 0] = variance_amp * (self.sigma_xy ** 2)
        prev_sigma[1, 1] = variance_amp * (self.sigma_xy ** 2)

        cov_mat = np.zeros([enu_kf.shape[0], 4])
        num_frames = enu_kf.shape[0]
        for iFrame in range(num_frames):
            curr_enu = self.enu_noise[iFrame, 0:2].T[np.newaxis]
            z = np.vstack((curr_enu.T, np.zeros([4, 1], dtype=float)))
            mu_t, sigma_t = self.kalman_filter_iter(prev_mu, prev_sigma, z, self.times[iFrame])
            enu_kf[iFrame, :] = mu_t[0:2, 0]
            cov_mat[iFrame, :] = sigma_t[0, 0], sigma_t[0, 1], sigma_t[1, 0], sigma_t[1, 1]

            # Update
            prev_mu = mu_t
            prev_sigma = sigma_t

        return enu_kf, cov_mat


# ------------------------------------------------------------------------------------------------- #
class ExtendedKalmanFilter:
    """
    class for the implementation of the extended Kalman filter
    """
    def __init__(self, enu_noise, yaw_vf_wz, times, sigma_xy, sigma_theta, sigma_vf, sigma_wz, k, is_dead_reckoning, dead_reckoning_start_sec=5.0):
        """
        Args:
            enu_noise: enu data with noise
            times: elapsed time in seconds from the first timestamp in the sequence
            sigma_xy: sigma in the x and y axis as provided in the question
            sigma_n: hyperparameter used to fine tune the filter
            yaw_vf_wz: the yaw, forward velocity and angular change rate to be used (either non noisy or noisy, depending on the question)
            sigma_theta: sigma of the heading
            sigma_vf: sigma of the forward velocity
            sigma_wz: sigma of the angular change rate
            k: hyper parameter to fine tune the filter
            is_dead_reckoning: should dead reckoning be applied after 5.0 seconds when applying the filter
            dead_reckoning_start_sec: from what second do we start applying dead reckoning, used for experimentation only
        """
        self.enu_noise = enu_noise
        self.yaw_vf_wz = yaw_vf_wz
        self.times = times
        self.sigma_xy = sigma_xy
        self.sigma_theta = sigma_theta
        self.sigma_vf = sigma_vf
        self.sigma_wz = sigma_wz
        self.k = k
        self.is_dead_reckoning = is_dead_reckoning
        self.dead_reckoning_start_sec = dead_reckoning_start_sec
        self.delta_t = times[1] - times[0]

        # Matrices notation as in the lecture
        # Jacobian Matrix G

        # Jacobian Matrix H
        self.H = np.array([[1, 0, 0], [0, 1, 0]])

        # Covariance Matrix Q
        self.Q = np.diag(np.array([sigma_xy ** 2, sigma_xy ** 2]))

        # Covariance Matrix R
        self.R = np.zeros([3, 3], dtype=float)
        self.R_hat = np.diag(np.array([sigma_vf ** 2, sigma_wz ** 2]))

        # Init Sigma
        self.sigma_0 = np.diag(np.array([k*(sigma_xy**2), k*(sigma_xy**2), k*sigma_theta]))

    @staticmethod
    def g_func(vf, wz, prev_mu, delta_t):
        result = prev_mu
        yaw = prev_mu[2]
        result[0] += (-(vf/wz)*math.sin(yaw) + (vf/wz)*math.sin(yaw + wz*delta_t))
        result[1] += ((vf/wz)*math.cos(yaw) - (vf/wz)*math.cos(yaw + wz*delta_t))
        result[2] += wz*delta_t
        return result

    @staticmethod
    def calc_RMSE_maxE(X_Y_GT, X_Y_est):
        """
        That function calculates RMSE and maxE

        Args:
            X_Y_GT (np.ndarray): ground truth values of x and y
            X_Y_est (np.ndarray): estimated values of x and y

        Returns:
            (float, float): RMSE, maxE
        """

        converge_delay = 100
        RMSE = np.sqrt(np.mean((X_Y_GT[converge_delay:] - X_Y_est[converge_delay:]) ** 2))
        maxE = np.max(np.sum(np.abs(X_Y_GT[converge_delay:] - X_Y_est[converge_delay:]), axis=1))

        return RMSE, maxE

    @staticmethod
    def compute_containment_percent(err_cov_x, err_cov_y, err_cov_yaw):
        err_x, cov_x = err_cov_x
        err_y, cov_y = err_cov_y
        err_yaw, cov_yaw = err_cov_yaw

        converge_delay = 100
        err_x = err_x[converge_delay:]
        cov_x = cov_x[converge_delay:]
        err_y = err_y[converge_delay:]
        cov_y = cov_y[converge_delay:]
        err_yaw = err_yaw[converge_delay:]
        cov_yaw = cov_yaw[converge_delay:]

        count_x = np.count_nonzero(np.abs(err_x) <= cov_x)
        count_y = np.count_nonzero(np.abs(err_y) <= cov_y)
        count_yaw = np.count_nonzero(np.abs(err_yaw) <= cov_yaw)

        containment_percent_x = (count_x / len(err_x)) * 100
        containment_percent_y = (count_y / len(err_y)) * 100
        containment_percent_yaw = (count_yaw / len(err_yaw)) * 100

        return containment_percent_x, containment_percent_y, containment_percent_yaw

    def extended_kalman_filter_iter(self, prev_mu, prev_sigma, yaw, vf, wz, z, delta_t, curr_time):

        V = np.array([[-(1 / wz) * math.sin(yaw) + (1 / wz) * math.sin(yaw + wz * delta_t),
                      (vf / (wz ** 2)) * math.sin(yaw) - (vf / (wz ** 2)) * math.sin(yaw + wz * delta_t) + (vf / wz) * math.cos(yaw + wz * delta_t) * delta_t],
                      [(1 / wz) * math.cos(yaw) - (1 / wz) * math.cos(yaw + wz * delta_t),
                       -(vf / (wz ** 2)) * math.cos(yaw) + (vf / (wz ** 2)) * math.cos(yaw + wz * delta_t) + (vf / wz) * math.sin(yaw + wz * delta_t) * delta_t],
                      [0, delta_t]])

        G = np.array([[1, 0, -(vf / wz) * math.cos(yaw) + (vf / wz) * math.cos(yaw + wz * delta_t)],
                      [0, 1, -(vf / wz) * math.sin(yaw) + (vf / wz) * math.sin(yaw + wz * delta_t)],
                      [0, 0, 1]])

        R_t1 = np.diag(np.array([0.001, 0.0016, 0.00076]))
        R = np.dot(V, np.dot(self.R_hat, V.T))

        # Predicition step
        mu_pred = self.g_func(vf, wz, prev_mu, delta_t)
        sigma_pred = np.dot(G, np.dot(prev_sigma, G.T)) + R + R_t1

        # Kalman Gain
        inv_mat = np.linalg.inv(np.dot(self.H, np.dot(sigma_pred, self.H.T)) + self.Q)
        if not (self.is_dead_reckoning) or curr_time < 5.0:
            K = np.dot(sigma_pred, np.dot(self.H.T, inv_mat))
        else:
            K = np.zeros([3, 2], dtype=float)

        # Correction step
        mu = mu_pred + np.dot(K, (z - np.dot(self.H, mu_pred)))
        sigma = np.dot((np.eye(3, dtype=float) - np.dot(K, self.H)), sigma_pred)

        return mu, sigma

    def run(self):
        """
        Runs the extended Kalman filter
        outputs: enu_ekf, yaw_ekf, cov_mat
        """

        # Collect results
        enu_ekf = np.zeros(self.enu_noise[:, 0:2].shape)  # x,y
        yaw_ekf = np.zeros(self.enu_noise[:, 0].shape)
        cov_mat = np.zeros([enu_ekf.shape[0], 5])

        # Extract yaw, vf, wz
        yaw_vec = self.yaw_vf_wz[:, 0]
        vf_vec = self.yaw_vf_wz[:, 1]
        wz_vec = self.yaw_vf_wz[:, 2]

        # Init mu and sigma
        prev_mu = np.zeros([3, 1], dtype=float)
        prev_mu[2] = yaw_vec[0]
        prev_sigma = self.sigma_0

        num_frames = enu_ekf.shape[0]
        for iFrame in range(num_frames):
            z = self.enu_noise[iFrame, 0:2].T[np.newaxis]
            mu_t, sigma_t = self.extended_kalman_filter_iter(prev_mu, prev_sigma, yaw_vec[iFrame], vf_vec[iFrame], wz_vec[iFrame], z.T, self.delta_t, self.times[iFrame])
            enu_ekf[iFrame, :] = mu_t[0:2, 0]
            cov_mat[iFrame, :] = sigma_t[0, 0],  sigma_t[0, 1], sigma_t[1, 0], sigma_t[1, 1], sigma_t[2, 2]
            yaw_ekf[iFrame] = mu_t[2, 0]

            # Update
            prev_mu = mu_t
            prev_sigma = sigma_t

        return enu_ekf, yaw_ekf, cov_mat

# ---------------------------------------------------------------------------------------------------------------- #
class ExtendedKalmanFilterSLAM:


    def __init__(self, sigma_x_y_theta, variance_r1_t_r2, variance_r_phi):

        """
        Args:
            variance_x_y_theta: variance in x, y and theta respectively
            variance_r1_t_r2: variance in rotation1, translation and rotation2 respectively
            variance_r_phi: variance in the range and bearing
        """

        self.sigma_x_y_theta = sigma_x_y_theta
        self.variance_r1_t_r2 = variance_r1_t_r2
        self.variance_r_phi = variance_r_phi
        self.R_tilde = np.diag(np.sqrt(variance_r1_t_r2))

    def predict(self, mu_prev, sigma_prev, u, N):
        # Perform the prediction step of the EKF
        # u[0]=translation, u[1]=rotation1, u[2]=rotation2

        delta_trans, delta_rot1, delta_rot2 = u['t'], u['r1'], u['r2']
        theta_prev = mu_prev[2]

        F = np.hstack((np.eye(3,dtype=float), np.zeros([3,2*N])))
        G_x = np.array([[0,0, -1*delta_trans*math.sin(theta_prev + delta_rot1)], [0, 0, delta_trans*math.cos(theta_prev + delta_rot1)], [0,0,0]])
        G = np.eye(3+2*N,dtype=float) + np.dot(F.T, np.dot(G_x, F))

        V = np.array([[-delta_trans*math.sin(theta_prev+delta_rot1), delta_trans*math.cos(theta_prev+delta_rot1), 0],
                      [delta_trans*math.cos(theta_prev+delta_rot1), delta_trans*math.sin(theta_prev+delta_rot1), 0],
                      [1, 0, 1]])

        # V = np.array([[-delta_trans*math.sin(theta_prev+delta_rot1), math.cos(theta_prev+delta_rot1), 0],
        #               [delta_trans*math.cos(theta_prev+delta_rot1), math.sin(theta_prev+delta_rot1), 0],
        #               [1, 0, 1]])

        R_x = np.dot(V, np.dot(self.R_tilde, V.T))
        R = np.dot(F.T, np.dot(R_x, F))

        mu_est = mu_prev + np.squeeze(np.dot(F.T, np.array([[delta_trans*math.cos(theta_prev+delta_rot1)], [delta_trans*math.sin(theta_prev+delta_rot1)],[delta_rot1+delta_rot2]])))
        sigma_est = np.dot(G, np.dot(sigma_prev, G.T)) + R

        return mu_est, sigma_est

    def update(self, mu_pred, sigma_pred, z, observed_landmarks, N):
        # Perform filter update (correction) for each odometry-observation pair read from the data file.
        mu = mu_pred.copy()
        sigma = sigma_pred.copy()
        theta = mu[2]

        m = len(z["id"])
        Z = np.zeros(2 * m)
        z_hat = np.zeros(2 * m)
        H = None

        for idx in range(m):
            j = z["id"][idx] - 1
            r = z["range"][idx]
            phi = z["bearing"][idx]

            mu_j_x_idx = 3 + j*2
            mu_j_y_idx = 4 + j*2
            Z_j_x_idx = idx*2
            Z_j_y_idx = 1 + idx*2

            if observed_landmarks[j] == False:
                mu[mu_j_x_idx: mu_j_y_idx + 1] = mu[0:2] + np.array([r * np.cos(phi + theta), r * np.sin(phi + theta)])
                observed_landmarks[j] = True

            Z[Z_j_x_idx : Z_j_y_idx + 1] = np.array([r, phi])

            delta = mu[mu_j_x_idx : mu_j_y_idx + 1] - mu[0 : 2]
            q = delta.dot(delta)
            q_sqrt = np.sqrt(q)

            z_hat_range = q_sqrt
            z_hat_angle = normalize_angle(np.arctan2(delta[1], delta[0]) - theta)
            z_hat[Z_j_x_idx : Z_j_y_idx + 1] = np.array([z_hat_range, z_hat_angle])

            I = np.diag(5*[1])
            F_j = np.hstack((I[:,:3], np.zeros((5, 2*j)), I[:,3:], np.zeros((5, 2*N-2*(j+1)))))

            delta_x = delta[0]
            delta_y = delta[1]
            Hi_low = (1/q)*(np.array([[-q_sqrt*delta_x, -q_sqrt*delta_y, 0, q_sqrt*delta_x, q_sqrt*delta_y],
                                     [delta_y, -delta_x, -q, -delta_y, delta_x]], dtype=float))

            Hi = np.dot(Hi_low, F_j)

            if H is None:
                H = Hi.copy()
            else:
                H = np.vstack((H, Hi))

        # Q = np.diag(np.repeat(10*np.sqrt(self.variance_r_phi), int(H.shape[0]/2) ))

        # a = np.array([0.003, 0.003])
        # Q = np.diag(np.tile(a, m))
        #
        Q = np.diag(np.tile(self.variance_r_phi, m))

        S = np.linalg.inv(np.dot(H, np.dot(sigma_pred, H.T)) + Q)
        K = sigma_pred.dot((H.T).dot(S))

        diff = Z - z_hat
        diff[1::2] = normalize_angles_array(diff[1::2])

        mu = mu + K.dot(diff)
        sigma = (np.eye(sigma_pred.shape[0]) - K.dot(H)).dot(sigma_pred)

        mu[2] = normalize_angle(mu[2])

        # Remember to normalize the bearings after subtracting!
        # (hint: use the normalize_all_bearings function available in tools)

        # Finish the correction step by computing the new mu and sigma.
        # Normalize theta in the robot pose.


        return mu, sigma, observed_landmarks

    def run(self, sensor_data_gt, sensor_data_noised, landmarks, ax):
        # Get the number of landmarks in the map
        N = len(landmarks)

        # Initialize belief:
        # mu: 2N+3x1 vector representing the mean of the normal distribution
        # The first 3 components of mu correspond to the pose of the robot,
        # and the landmark poses (xi, yi) are stacked in ascending id order.
        # sigma: (2N+3)x(2N+3) covariance matrix of the normal distribution


        init_inf_val = 100

        mu_arr = np.zeros((1, 3 + 2*N))
        sigma_prev = np.diag(np.hstack((np.array(self.sigma_x_y_theta), init_inf_val*np.ones(2*N))))

        # sigma for analysis graph sigma_x_y_t + select 2 landmarks
        landmark1_ind = 3
        landmark2_ind = 5

        Index = [0, 1, 2, landmark1_ind, landmark1_ind+1, landmark2_ind, landmark2_ind+1]
        sigma_x_y_t_px1_py1_px2_py2 = sigma_prev[Index,Index].copy()

        observed_landmarks = np.zeros(N, dtype=bool)

        sensor_data_count = int(len(sensor_data_noised) / 2)
        frames = []

        mu_arr_gt = np.array([[0, 0, 0]])

        for idx in range(sensor_data_count):
            mu_prev = mu_arr[-1]

            u = sensor_data_noised[(idx, "odometry")]
            # predict
            mu_pred, sigma_pred = self.predict(mu_prev, sigma_prev, u, N)

            # update (correct)
            mu, sigma, observed_landmarks = self.update(mu_pred, sigma_pred, sensor_data_noised[(idx, "sensor")], observed_landmarks, N)

            mu_arr = np.vstack((mu_arr, mu))
            sigma_prev = sigma.copy()
            sigma_x_y_t_px1_py1_px2_py2 = np.vstack((sigma_x_y_t_px1_py1_px2_py2, sigma_prev[Index, Index].copy()))

            delta_r1_gt = sensor_data_gt[(idx, "odometry")]["r1"]
            delta_r2_gt = sensor_data_gt[(idx, "odometry")]["r2"]
            delta_trans_gt = sensor_data_gt[(idx, "odometry")]["t"]

            calc_x = lambda theta_p: delta_trans_gt * np.cos(theta_p + delta_r1_gt)
            calc_y = lambda theta_p: delta_trans_gt * np.sin(theta_p + delta_r1_gt)

            theta = delta_r1_gt + delta_r2_gt

            theta_prev = mu_arr_gt[-1,2]
            mu_arr_gt = np.vstack((mu_arr_gt, mu_arr_gt[-1] + np.array([calc_x(theta_prev), calc_y(theta_prev), theta])))

            frame = plot_state(ax, mu_arr_gt, mu_arr, sigma, landmarks, observed_landmarks, sensor_data_noised[(idx, "sensor")])

            frames.append(frame)

        return frames, mu_arr, mu_arr_gt, sigma_x_y_t_px1_py1_px2_py2

    @staticmethod
    def calc_RMSE_maxE(X_Y_GT, X_Y_est):
        """
        That function calculates RMSE and maxE

        Args:
            X_Y_GT (np.ndarray): ground truth values of x and y
            X_Y_est (np.ndarray): estimated values of x and y

        Returns:
            (float, float): RMSE, maxE
        """

        converge_delay = 20
        diff_xy = (X_Y_est[converge_delay:, :] - X_Y_GT[converge_delay:, :])
        RMSE = np.linalg.norm(diff_xy) / np.sqrt(len(diff_xy[:, 0]))
        maxE = np.max(np.sum(np.abs(X_Y_GT[converge_delay:] - X_Y_est[converge_delay:]), axis=1))

        return RMSE, maxE