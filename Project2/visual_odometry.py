import numpy as np
import math
import cv2
from data_loader import DataLoader
from camera import Camera
import matplotlib.pyplot as plt
import os
from pathlib import Path
import graphs
import matplotlib.animation as animation
import itertools

np.random.seed(333)


class VisualOdometry:

    def __init__(self, vo_data):
        self.vo_data = vo_data
        P0 = vo_data.get_intrinsics()
        self.camera_intrinsic_mat = P0

    def calc_trajectory(self):
        fig_save_path = "..\\..\\results\\ex2\\"
        if (not os.path.exists(fig_save_path)):
            fig_save_path = os.path.join("..\\", fig_save_path)
        Path(fig_save_path).mkdir(parents=True, exist_ok=True)  # create fig_save_path directory if doesn't exsit!

        gt_pose = np.array([0.0, 0.0]).reshape(2, 1)
        gt_trajectory = np.zeros([1, 2])
        est_trajectory = np.zeros([1, 2])

        T = np.identity(4)
        images_gen = self.vo_data.images
        poses_gen = self.vo_data.gt_poses

        # Create some random colors
        color = np.random.randint(0, 255, (300, 3))
        # create SIFT
        sift = cv2.SIFT_create()
        t_gt = np.zeros([2, 1])
        prev_t_gt = np.zeros([2, 1])
        # Take first frame and find corners in it
        old_frame = next(images_gen)
        prev_pose = next(poses_gen)
        prev_t_gt[0] = prev_pose[0, -1]
        prev_t_gt[1] = prev_pose[2, -1]
        # create image list for animation
        max_frames = 5000
        img_list = np.zeros([max_frames + 2, old_frame.shape[0], old_frame.shape[1], 3], dtype='uint8')
        # first_frame_flag = True
        t_est = np.zeros([3, 1])
        R_est = np.eye(3)
        pose_est = np.zeros([2, 1])

        fig3, ax3 = plt.subplots()
        fig3.suptitle('GT vs estimated trajectory')

        prev_pose = np.zeros([3, 4])
        # kp_array = np.zeros([max_frames, ])
        for i, (image, pose) in enumerate(zip(images_gen, poses_gen)):
            if i > max_frames:
                break

            t_gt[0] = pose[0, -1]
            t_gt[1] = pose[2, -1]
            # compute relative scale
            scale = np.linalg.norm(t_gt - prev_t_gt)  # scale factor
            # scale = np.linalg.norm(prev_pose[:,-1] - pose[:,-1], axis=1) # scale factor
            prev_pose = pose.copy()
            prev_t_gt = t_gt.copy()
            gt_pose = t_gt

            gt_trajectory = np.vstack((gt_trajectory, gt_pose[0:2].T))
            # print(gt_pose)
            # print(gt_pose.shape)
            # ax.scatter(gt_pose[0],gt_pose[1])
            # p0 = cv2.goodFeaturesToTrack(old_frame, mask = None, **feature_params)
            kp1, des1 = sift.detectAndCompute(old_frame, None)
            if (i == 0):  # first iteration add to image list
                img_for_anim = cv2.drawKeypoints(old_frame, kp1, old_frame)
                img_list[0, :, :, :] = img_for_anim

            # Create a mask image for drawing purposes
            mask = np.zeros_like(old_frame)

            frame = image
            # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            # p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, p0, None, **lk_params)
            kp2, des2 = sift.detectAndCompute(frame, None)

            # match points using BFMatcher()
            bf = cv2.BFMatcher()
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1, des2, k=2)
            # matches = bf.knnMatch(des1,des2, k=2)
            # matches = sorted(matches, key = lambda x:x.distance)
            # Apply ratio test
            pts1 = []
            pts2 = []
            good = []

            matchesMask = [[0, 0] for i in range(len(matches))]
            # ratio test as per Lowe's paper
            for idx, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[idx] = [1, 0]
                    good.append([m])
                    pts2.append(kp2[m.trainIdx].pt)
                    pts1.append(kp1[m.queryIdx].pt)

            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=cv2.DrawMatchesFlags_DEFAULT)
            pts1 = np.int32(pts1)
            pts2 = np.int32(pts2)
            # find Essential Matrix based on matches (5-points)
            E, mask_e = cv2.findEssentialMat(pts1, pts2, self.camera_intrinsic_mat, cv2.RANSAC)
            points, R_curr, t_curr, mask_pose = cv2.recoverPose(E, pts1, pts2, self.camera_intrinsic_mat)

            # rescale traslation_vec (t_est)
            t_est = t_est + (scale * R_est.dot(t_curr))

            # R_est = R_curr.dot(R_est) # culamative R matrix
            R_est = R_est.dot(R_curr)  # culamative R matrix

            pose_est[0] = t_est[0]
            pose_est[1] = -t_est[2]
            est_trajectory = np.vstack((est_trajectory, pose_est[0:2].T))

            img3 = cv2.drawMatchesKnn(old_frame, kp1, frame, kp2, matches, None, **draw_params)
            img_for_anim = cv2.drawKeypoints(frame, kp2, frame)
            img_list[i + 1, :, :, :] = img_for_anim
            # img3 = cv2.drawMatchesKnn(old_frame,kp1,frame,kp2,good,old_frame,flags=2)
            # cv2.imshow('frame',img_for_anim)
            # k = cv2.waitKey(30) & 0xff
            # if k == 27:
            #     break

            # Now update the previous frame and previous points
            old_frame = frame.copy()

        ax3.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b--', label='ground-truth')
        ax3.plot(est_trajectory[:, 0], est_trajectory[:, 1], 'r-', label='estimated')
        ax3.set_xlabel('X[m]')
        ax3.set_ylabel('Y[m]')
        ax3.legend()
        # ax4.plot(est_trajectory[:,0],est_trajectory[:,1])
        fig3.savefig(os.path.join(fig_save_path, 'Q2_GT_vs_estimated_trajectory.png'), transparent=True)
        # calc euclidian distance between GT and Estimated trajectories
        distance = math.sqrt(np.linalg.norm(est_trajectory - gt_trajectory))
        print('distance (estimated-gt) = ', distance)
        kp_array = np.zeros([1])
        anim = self.build_animation(gt_trajectory[:, 0:2], est_trajectory[:, 0:2], img_list, kp_array,
                                    'VO estimated vs ground-truth trajectory ', 'X[m]', 'Y[m]', 'ground-truth',
                                    'estimated trajectory')
        graphs.save_animation(anim, fig_save_path, 'Q2_GT_vs_estimated_trajectory_animation')

        # show plots
        # plt.show()
        cv2.destroyAllWindows()

        return True

    def build_animation(self, X_Y0, X_Y1, img_array, kp_array, title, xlabel, ylabel, label0, label1):
        frames = []
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
            # val2.set_data(next(itertools.islice(self.vo_data.images, self.frame_idx, None)))
            val2.set_data(self.img_array[self.frame_idx, :, :, :])
            self.frame_idx += 1
            return val0, val1, val2

        anim = animation.FuncAnimation(fig, update, frames=values, init_func=init, interval=1, repeat=False, blit=True)
        return anim
