#
#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function, unicode_literals

import argparse
import glob
import sys

import tensorflow as tf
import numpy as np
import scipy.misc
import os
import os.path
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d


def run_model_on_image(image_raw):
    image_raw = scipy.misc.imresize(image_raw, (240, 320))
    image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

    hand_scoremap_v, image_crop_v, scale_v, center_v, \
    keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
                                                         keypoints_scoremap_tf, keypoint_coord3d_tf],
                                                        feed_dict={image_tf: image_v})

    keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

    return hand_scoremap_v, image_crop_v, scale_v, center_v, \
           keypoints_scoremap_v, keypoint_coord3d_v


def get_coords_and_figure_from_name(img_name):
    image_raw = scipy.misc.imread(img_name)

    hand_scoremap_v, image_crop_v, scale_v, center_v, \
    keypoints_scoremap_v, keypoint_coord3d_v = run_model_on_image(image_raw)

    # from here on: saving stuff
    basename = os.path.splitext(os.path.basename(img_name))[0]

    # post processing
    hand_scoremap_v = np.squeeze(hand_scoremap_v)
    image_crop_v = np.squeeze(image_crop_v)
    keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
    image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
    coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
    coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)
    # save keypoint coordinates
    keypoint_save_filename = "{:s}/{:s}_coords.pkl".format(output_coords_dir, basename)
    with open(keypoint_save_filename, 'wb') as f:
        pickle.dump((keypoint_coord3d_v, scale_v, center_v), f, protocol=-1)
    print("Saved keypoint coordinates to {:s}".format(keypoint_save_filename))

    # save image
    image_save_filename = "{:s}/{:s}_figures.png".format(output_figures_dir, basename)

    fig = plt.figure(1)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection='3d')
    ax1.imshow(image_raw)
    plot_hand(coord_hw, ax1)
    ax2.imshow(image_crop_v)
    plot_hand(coord_hw_crop, ax2)
    ax3.imshow(np.argmax(hand_scoremap_v, 2))
    plot_hand_3d(keypoint_coord3d_v, ax4)
    ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
    ax4.set_xlim([-3, 3])
    ax4.set_ylim([-3, 1])
    ax4.set_zlim([-3, 3])
    plt.savefig(image_save_filename)
    plt.close()

    print("Saved figure to {:s}".format(image_save_filename))


if __name__ == '__main__':
    sys.setrecursionlimit(3000)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, type=str)
    parser.add_argument("--output_figures_dir", required=True, type=str)
    parser.add_argument("--output_coords_dir", required=True, type=str)

    args = parser.parse_args()

    image_dir = args.image_dir
    output_figures_dir = args.output_figures_dir
    output_coords_dir = args.output_coords_dir

    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
    hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
    evaluation = tf.placeholder_with_default(True, shape=())

    # build network
    net = ColorHandPose3DNetwork()
    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, \
    keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # initialize network
    net.init(sess)

    # images to be shown
    image_list = sorted(glob.glob("{:s}/*.jpg".format(image_dir)))

    # Feed image list through network
    # for img_name in image_list:
    #     image_raw = scipy.misc.imread(img_name)
    #     image_raw = scipy.misc.imresize(image_raw, (240, 320))
    #     image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)
    #
    #     hand_scoremap_v, image_crop_v, scale_v, center_v, \
    #     keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
    #                                                          keypoints_scoremap_tf, keypoint_coord3d_tf],
    #                                                         feed_dict={image_tf: image_v})
    #
    #     hand_scoremap_v = np.squeeze(hand_scoremap_v)
    #     image_crop_v = np.squeeze(image_crop_v)
    #     keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
    #     keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)
    #
    #     # from here on: saving stuff
    #     basename = os.path.splitext(os.path.basename(img_name))[0]
    #
    #     # save keypoint coordinates
    #     keypoint_save_filename = "{:s}/{:s}_coords.pkl".format(output_coords_dir, basename)
    #     with open(keypoint_save_filename, 'wb') as f:
    #         pickle.dump(keypoint_coord3d_v, f)
    #     print("Saved keypoint coordinates to {:s}".format(keypoint_save_filename))
    #
    #     # post processing
    #     image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
    #     coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
    #     coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)
    #
    #     # save image
    #     image_save_filename = "{:s}/{:s}_figures.png".format(output_figures_dir, basename)
    #
    #     fig = plt.figure(1)
    #     ax1 = fig.add_subplot(221)
    #     ax2 = fig.add_subplot(222)
    #     ax3 = fig.add_subplot(223)
    #     ax4 = fig.add_subplot(224, projection='3d')
    #     ax1.imshow(image_raw)
    #     plot_hand(coord_hw, ax1)
    #     ax2.imshow(image_crop_v)
    #     plot_hand(coord_hw_crop, ax2)
    #     ax3.imshow(np.argmax(hand_scoremap_v, 2))
    #     plot_hand_3d(keypoint_coord3d_v, ax4)
    #     ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
    #     ax4.set_xlim([-3, 3])
    #     ax4.set_ylim([-3, 1])
    #     ax4.set_zlim([-3, 3])
    #     plt.savefig(image_save_filename)
    #     plt.close()
    #
    #     print("Saved figure to {:s}".format(image_save_filename))

    Parallel(n_jobs=1, prefer="threads")(
        delayed(get_coords_and_figure_from_name)(img_name) for img_name in tqdm(image_list))
