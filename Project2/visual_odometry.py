import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

import graphs
import matplotlib.animation as animation
import itertools
import os
class VisualOdometry:

    def __init__(self, vo_data):
        self.vo_data = vo_data

        # initial camera pose
        self.camera_rotation = np.eye(3)
        self.camera_translation = np.zeros([3, 1])
        self.camera_intrinsic_mat = vo_data.get_intrinsics()

    def calc_trajectory(self):
        """
         apply the visual odometry algorithm
         """

        # Init params
        gt_trajectory = np.zeros([1, 2])
        measured_trajectory = np.zeros([1, 2])

        # Feature Extraction
        sift = cv2.SIFT_create()

        # Const
        MAX_FRAMES = 50
        frame_idx = -1

        prev_img = None
        prev_gt_pose = None

        for curr_img, curr_gt_pose in zip(self.vo_data.images, self.vo_data.gt_poses):

            # 1 - Capture new frame (curr_img) and pose (curr_gt_pose)
            if prev_img is None:
                prev_img = curr_img
                prev_gt_pose = curr_gt_pose
                img_list = np.zeros([MAX_FRAMES + 2, prev_img.shape[0], prev_img.shape[1], 3], dtype='uint8')
                continue

            frame_idx += 1
            if frame_idx % 10 == 0:
                print("Finished : ", 100 * (frame_idx / MAX_FRAMES), "% of frames")

            if frame_idx > MAX_FRAMES:
                break

            # 2.a - Extract features from the previous and current frames
            kp1, des1 = sift.detectAndCompute(prev_img, None)
            kp2, des2 = sift.detectAndCompute(curr_img, None)

            # 2.b - Match features between the previous and current frames
            pts1, pts2 = self.match_points_between_frames(kp1, des1, kp2, des2)

            # 3 - Compute essential matrix
            E, _ = cv2.findEssentialMat(pts1, pts2, self.camera_intrinsic_mat, cv2.RANSAC)

            # 4 - Decompose essential matrix to R, t
            _, R_curr, t_curr, _ = cv2.recoverPose(E, pts1, pts2, self.camera_intrinsic_mat)

            # 5 - Compute relative scale and rescale translation
            curr_pose = np.array([curr_gt_pose[0, 3], curr_gt_pose[2, 3]])
            prev_pose = np.array([prev_gt_pose[0, 3], prev_gt_pose[2, 3]])
            scale = np.linalg.norm(curr_pose - prev_pose)

            # 6 - Concatenate transformation
            self.camera_translation = self.camera_translation + (scale * self.camera_rotation.dot(t_curr))
            self.camera_rotation = self.camera_rotation.dot(R_curr)

            gt_trajectory = np.vstack((gt_trajectory, curr_pose))
            measured_trajectory = np.vstack((measured_trajectory, np.array([float(self.camera_translation[0]), float(-self.camera_translation[2])])))

            # Update frame & Pose
            prev_img = curr_img.copy()
            prev_gt_pose = curr_gt_pose.copy()

            # Save for visualization
            if frame_idx == 0:
                img_for_anim = cv2.drawKeypoints(prev_img, kp1, prev_img)
                img_list[0, :, :, :] = img_for_anim

            img_for_anim = cv2.drawKeypoints(curr_img, kp2, curr_img)
            img_list[frame_idx + 1, :, :, :] = img_for_anim

        # ------------------------------------------------------------------------------ #
        # Plot
        err_vec = np.linalg.norm(measured_trajectory - gt_trajectory, axis=1)
        fig2, ax2 = plt.subplots()
        ax2.plot(err_vec)
        ax2.set_xlabel('frame number')
        ax2.set_ylabel('error [m]')
        ax2.set_title('error vs frame number')

        fig3, ax3 = plt.subplots()
        fig3.suptitle('GT vs estimated trajectory')
        ax3.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b--', label='ground-truth')
        ax3.plot(measured_trajectory[:, 0], measured_trajectory[:, 1], 'r-', label='estimated')
        ax3.set_xlabel('X[m]')
        ax3.set_ylabel('Y[m]')
        ax3.legend()

        # calc euclidian distance between GT and Estimated trajectories
        distance = math.sqrt(np.linalg.norm(measured_trajectory - gt_trajectory))
        print('distance (estimated-gt) = ', distance)

        # Animation
        ani = self.build_animation(gt_trajectory[:, 0:2], measured_trajectory[:, 0:2], img_list,
                                    'VO estimated vs ground-truth trajectory ', 'X[m]', 'Y[m]', 'ground-truth',
                                    'estimated trajectory')
        fig_save_path = r"D:\Masters Degree\Courses\Sensors In Autonomous Systems 0510-7951\Homework\Autonomous_Systems\results\Q3"
        graphs.save_animation(ani, fig_save_path, 'Q3_GT_vs_estimated_trajectory_animation')

        # show plots
        plt.show()
        cv2.destroyAllWindows()

        return True
        # return gt_trajectory, measured_trajectory, key_points_history

    @staticmethod
    def match_points_between_frames(kp1, des1, kp2, des2):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # Apply ratio test
        pts1, pts2 = [], []
        matches_mask = [[0, 0] for i in range(len(matches))]

        for idx, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matches_mask[idx] = [1, 0]
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        return pts1, pts2

    def build_animation(self, X_Y0, X_Y1, img_array, title, xlabel, ylabel, label0, label1):
        self.img_array = img_array
        self.frame_idx = 0
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 2)
        imageAx = fig.add_subplot(2, 1, 1)
        print("Creating animation")

        x0, y0, x1, y1 = [], [], [], []
        val0, = ax.plot([], [], 'b:', animated=True, label=label0)
        val1, = ax.plot([], [], 'r-', animated=True, label=label1)
        val2 = imageAx.imshow(next(itertools.islice(self.vo_data.images, 0, None)), cmap=plt.get_cmap('gray'),
                              animated=True)

        ax.legend()

        values = np.hstack((X_Y0, X_Y1))

        def init():
            self.frame_idx = 0
            ax.set_xlim(-10, 700)
            ax.set_ylim(-30, 1000)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            val0.set_data([], [])
            val1.set_data([], [])

            val2.set_data(next(itertools.islice(self.vo_data.images, 0, None)))
            # img.set_data(data)
            return val0, val1

        def update(frame):
            x0.append(frame[0])
            y0.append(frame[1])
            x1.append(frame[2])
            y1.append(frame[3])

            val0.set_data(x0, y0)
            val1.set_data(x1, y1)
            val2.set_array(next(itertools.islice(self.vo_data.images, self.frame_idx, None)))
            val2.set_data(self.img_array[self.frame_idx, :, :, :])
            self.frame_idx += 1
            return val0, val1, val2

        anim = animation.FuncAnimation(fig, update, frames=values, init_func=init, interval=1, repeat=False, blit=True)
        return anim
