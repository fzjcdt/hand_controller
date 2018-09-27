import math
import time

import cv2
import numpy as np
import tensorflow as tf
from pynput.keyboard import Key, Controller

from classify_inference import ClassifyInference
from config import FLAGS
from cpm_inference import CPMInference
from utils import tracking_module, utils

joint_detections = np.zeros(shape=(21, 2))


def main(argv):
    global joint_detections

    # Create webcam instance
    cam = cv2.VideoCapture(FLAGS.cam_id)

    # Create kalman filters
    kalman_filter_array = [cv2.KalmanFilter(4, 2) for _ in range(FLAGS.num_of_joints)]
    for _, joint_kalman_filter in enumerate(kalman_filter_array):
        joint_kalman_filter.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32)
        joint_kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        joint_kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                       np.float32) * FLAGS.kalman_noise

    cpm_inference = CPMInference()
    classify_inference = ClassifyInference()
    keyboard = Controller()
    tracker = tracking_module.SelfTracker([FLAGS.webcam_height, FLAGS.webcam_width], FLAGS.input_size)

    count = 0
    action_count = [5, 10, 20, 40]
    last_gesture = -1
    action = ['up', 'down', 'left', 'right', 'larger', 'smaller']
    labels2key = {0: Key.down, 1: Key.up, 2: Key.right, 3: Key.left, 4: '+', 5: '-'}

    while True:
        # Prepare input image
        _, full_img = cam.read()

        test_img = tracker.tracking_by_joints(full_img, joint_detections=joint_detections)
        crop_full_scale = tracker.input_crop_ratio
        test_img_copy = test_img.copy()

        # White balance
        test_img_wb = utils.img_white_balance(test_img, 5)
        test_img_input = normalize_and_centralize_img(test_img_wb)

        # Inference
        t1 = time.time()
        stage_heatmap_np = cpm_inference.predict(test_img_input)
        # print('FPS: %.2f' % (1 / (time.time() - t1)))

        local_img, response = visualize_result(full_img, stage_heatmap_np, kalman_filter_array, tracker,
                                               crop_full_scale, test_img_copy)

        cv2.imshow('local_img', local_img.astype(np.uint8))  # 训练用图
        cv2.imshow('globalq_img', full_img.astype(np.uint8))  # 单人大框

        if response > 5.0:
            rst = classify_inference.predict([cv2.resize(local_img, (100, 100))])
            if rst == last_gesture:
                count += 1
            else:
                count = 1
                last_gesture = rst

            if count in action_count:
                keyboard.press(labels2key[last_gesture])
                keyboard.release(labels2key[last_gesture])
                print(action[last_gesture])
        else:
            last_gesture = -1

        if cv2.waitKey(1) == ord('q'):
            cpm_inference.shutdown()
            classify_inference.shutdown()
            break


def normalize_and_centralize_img(img):
    if FLAGS.color_channel == 'GRAY':
        img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).reshape((FLAGS.input_size, FLAGS.input_size, 1))

    if FLAGS.normalize_img:
        test_img_input = img / 256.0 - 0.5
        test_img_input = np.expand_dims(test_img_input, axis=0)
    else:
        test_img_input = img - 128.0
        test_img_input = np.expand_dims(test_img_input, axis=0)
    return test_img_input


def visualize_result(test_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img):
    demo_stage_heatmaps = []
    if FLAGS.DEMO_TYPE == 'MULTI':
        for stage in range(len(stage_heatmap_np)):
            demo_stage_heatmap = stage_heatmap_np[stage][0, :, :, 0:FLAGS.num_of_joints].reshape(
                (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
            demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size))
            demo_stage_heatmap = np.amax(demo_stage_heatmap, axis=2)
            demo_stage_heatmap = np.reshape(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size, 1))
            demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
            demo_stage_heatmap *= 255
            demo_stage_heatmaps.append(demo_stage_heatmap)

        last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.num_of_joints].reshape(
            (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
        last_heatmap = cv2.resize(last_heatmap, (FLAGS.input_size, FLAGS.input_size))
    else:
        last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.num_of_joints].reshape(
            (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
        last_heatmap = cv2.resize(last_heatmap, (FLAGS.input_size, FLAGS.input_size))

    response = correct_and_draw_hand(test_img, last_heatmap, kalman_filter_array, tracker, crop_full_scale, crop_img)

    if FLAGS.DEMO_TYPE == 'MULTI':
        if len(demo_stage_heatmaps) > 3:
            upper_img = np.concatenate((demo_stage_heatmaps[0], demo_stage_heatmaps[1], demo_stage_heatmaps[2]), axis=1)
            lower_img = np.concatenate(
                (demo_stage_heatmaps[3], demo_stage_heatmaps[len(stage_heatmap_np) - 1], crop_img),
                axis=1)
            demo_img = np.concatenate((upper_img, lower_img), axis=0)
            return demo_img, response
        else:
            # return np.concatenate((demo_stage_heatmaps[0], demo_stage_heatmaps[len(stage_heatmap_np) - 1], crop_img),
            #                       axis=1)

            return demo_stage_heatmaps[0], response
            # np.concatenate 合并array

    else:
        return crop_img


def correct_and_draw_hand(full_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img):
    global joint_detections
    joint_coord_set = np.zeros((FLAGS.num_of_joints, 2))
    local_joint_coord_set = np.zeros((FLAGS.num_of_joints, 2))

    mean_response_val = 0.0

    # Plot joint colors
    if kalman_filter_array is not None:
        for joint_num in range(FLAGS.num_of_joints):
            tmp_heatmap = stage_heatmap_np[:, :, joint_num]
            joint_coord = np.unravel_index(np.argmax(tmp_heatmap),
                                           (FLAGS.input_size, FLAGS.input_size))
            mean_response_val += tmp_heatmap[joint_coord[0], joint_coord[1]]
            joint_coord = np.array(joint_coord).reshape((2, 1)).astype(np.float32)
            kalman_filter_array[joint_num].correct(joint_coord)
            kalman_pred = kalman_filter_array[joint_num].predict()
            correct_coord = np.array([kalman_pred[0], kalman_pred[1]]).reshape((2))
            local_joint_coord_set[joint_num, :] = correct_coord

            # Resize back
            correct_coord /= crop_full_scale

            # Substract padding border
            correct_coord[0] -= (tracker.pad_boundary[0] / crop_full_scale)
            correct_coord[1] -= (tracker.pad_boundary[2] / crop_full_scale)
            correct_coord[0] += tracker.bbox[0]
            correct_coord[1] += tracker.bbox[2]
            joint_coord_set[joint_num, :] = correct_coord

    else:
        for joint_num in range(FLAGS.num_of_joints):
            tmp_heatmap = stage_heatmap_np[:, :, joint_num]
            joint_coord = np.unravel_index(np.argmax(tmp_heatmap),
                                           (FLAGS.input_size, FLAGS.input_size))
            mean_response_val += tmp_heatmap[joint_coord[0], joint_coord[1]]
            joint_coord = np.array(joint_coord).astype(np.float32)

            local_joint_coord_set[joint_num, :] = joint_coord

            # Resize back
            joint_coord /= crop_full_scale

            # Substract padding border
            joint_coord[0] -= (tracker.pad_boundary[2] / crop_full_scale)
            joint_coord[1] -= (tracker.pad_boundary[0] / crop_full_scale)
            joint_coord[0] += tracker.bbox[0]
            joint_coord[1] += tracker.bbox[2]
            joint_coord_set[joint_num, :] = joint_coord

    draw_hand(full_img, joint_coord_set, tracker.loss_track)
    draw_hand(crop_img, local_joint_coord_set, tracker.loss_track)
    joint_detections = joint_coord_set

    if mean_response_val >= 1:
        tracker.loss_track = False
    else:
        tracker.loss_track = True

    cv2.putText(full_img, 'Response: {:<.3f}'.format(mean_response_val),
                org=(20, 20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0))

    return mean_response_val


def draw_hand(full_img, joint_coords, is_loss_track):
    if is_loss_track:
        joint_coords = FLAGS.default_hand

    # Plot joints
    for joint_num in range(FLAGS.num_of_joints):
        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
            cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=3,
                       color=joint_color, thickness=-1)
        else:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
            cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])), radius=3,
                       color=joint_color, thickness=-1)

    # Plot limbs
    for limb_num in range(len(FLAGS.limbs)):
        x1 = int(joint_coords[int(FLAGS.limbs[limb_num][0])][0])
        y1 = int(joint_coords[int(FLAGS.limbs[limb_num][0])][1])
        x2 = int(joint_coords[int(FLAGS.limbs[limb_num][1])][0])
        y2 = int(joint_coords[int(FLAGS.limbs[limb_num][1])][1])
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if 150 > length > 5:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 3),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = limb_num // 4
            limb_color = list(map(lambda x: x + 35 * (limb_num % 4), FLAGS.joint_color_code[color_code_num]))
            cv2.fillConvexPoly(full_img, polygon, color=limb_color)


if __name__ == '__main__':
    tf.app.run()
