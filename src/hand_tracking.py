from typing import Union, List
from pathlib import Path
import re
from ipywidgets import interact, IntSlider, Layout
from epic_kitchens.hoa import load_detections, DetectionRenderer
import PIL.Image
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import pickle
import cv2
import sys, os
import json
import ast
import argparse


WIDTH = -1
HEIGHT = -1


def random_color():
    return tuple(np.random.choice(range(256), size=3))


def get_score_matrix(tracking_list, detection_list):
    score_matrix = np.zeros((len(tracking_list), len(detection_list))).astype(np.float32)

    assert (WIDTH > 0 and HEIGHT > 0)

    for i in range(len(tracking_list)):
        for j in range(len(detection_list)):
            track_center_x = (tracking_list[i][0] + tracking_list[i][2]) / 2.0
            track_center_y = (tracking_list[i][1] + tracking_list[i][3]) / 2.0
            detect_center_x = (detection_list[j][0] + detection_list[j][2]) / 2.0
            detect_center_y = (detection_list[j][1] + detection_list[j][3]) / 2.0
            tracking = np.array([track_center_x / WIDTH, track_center_y / HEIGHT])
            detecting = np.array([detect_center_x / WIDTH, detect_center_y / HEIGHT])

            score_matrix[i, j] = np.sqrt(((tracking - detecting) ** 2).sum())

    return score_matrix


def assign_detections_to_trackers(frame_idx, hand_history,
                                  tracker_list_idx, detections):
    new_track_list_idx = []
    trackers = []
    detect_bbox = []

    for hand_id in tracker_list_idx:
        trackers.append(hand_history[hand_id][2][-1])

    for detect_info in detections:
        detect_bbox.append(detect_info[:4])

    score_mat = get_score_matrix(trackers, detect_bbox)

    matched_row_idx, matched_col_idx = linear_sum_assignment(score_mat)

    for t, trk in enumerate(trackers):
        if t not in matched_row_idx:
            if hand_history[tracker_list_idx[t]][1] > 0:
                new_track_list_idx.append(tracker_list_idx[t])
                hand_history[tracker_list_idx[t]][1] -= 1

    for d, det in enumerate(detections):
        if d not in matched_col_idx:
            hand_history.append([frame_idx, 15, [det]])
            new_track_list_idx.append(len(hand_history) - 1)

    for i in range(len(matched_row_idx)):
        track_id, detect_id = matched_row_idx[i], matched_col_idx[i]

        if score_mat[track_id, detect_id] > 0.4:
            hand_history.append([frame_idx, 15, [detections[detect_id]]])
            new_track_list_idx.append(len(hand_history) - 1)

            if hand_history[tracker_list_idx[track_id]][1] > 0:
                new_track_list_idx.append(tracker_list_idx[track_id])
                hand_history[tracker_list_idx[track_id]][1] -= 1

        else:
            missing = frame_idx - (hand_history[tracker_list_idx[track_id]][0] +
                                   len(hand_history[tracker_list_idx[track_id]][2]))
            if missing > 0:
                filled_x1 = int(hand_history[tracker_list_idx[track_id]][2][-1][0])
                filled_y1 = int(hand_history[tracker_list_idx[track_id]][2][-1][1])
                filled_x2 = int(hand_history[tracker_list_idx[track_id]][2][-1][2])
                filled_y2 = int(hand_history[tracker_list_idx[track_id]][2][-1][3])
                filled_score = 0.9

                for i in range(missing):
                    hand_history[tracker_list_idx[track_id]][2].append([filled_x1, filled_y1, filled_x2, filled_y2,
                                                                        filled_score])

            hand_history[tracker_list_idx[track_id]][2].append(detections[detect_id])
            new_track_list_idx.append(tracker_list_idx[track_id])
            hand_history[tracker_list_idx[track_id]][1] = 15

    return hand_history, new_track_list_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default='handobj_detections/handobj_detections')
    args = parser.parse_args()
    data_base_path = args.root_dir

    hand_colors = [random_color() for i in range(20)]

    for video in os.listdir(data_base_path):
        frame_base = os.path.join(data_base_path, video, "frames")
        det_base = os.path.join(data_base_path, video, "frames_det_meta")
        frames = os.listdir(frame_base)
        frames.sort()
        first_img = cv2.imread(os.path.join(frame_base, frames[0]))
        global WIDTH, HEIGHT
        HEIGHT, WIDTH, _ = first_img.shape

        obj_history = []
        tracker_list_idx = []
        output_frame_dict = {}

        for frame_idx in range(0, len(frames)):
            output_frame_dict[frames[frame_idx]] = {}
            detection_list = []

            detection_file = frames[frame_idx].replace(".jpg", "_det.npz")
            det_path = os.path.join(det_base, detection_file)

            data = np.load(det_path, allow_pickle=True)
            if data['hand_dets'].shape:
                for hand in data['hand_dets']:
                    x1, y1, x2, y2, score = hand[:5]
                    detect_info = [x1, y1, x2, y2, score, frames[frame_idx]]
                    detection_list.append(detect_info)

            if len(obj_history) == 0:
                if len(detection_list) > 0:
                    for detect_info in detection_list:
                        obj_history.append([frame_idx, 15, [detect_info]])
                    tracker_list_idx = list(range(len(detection_list)))
            else:
                obj_history, tracker_list_idx = assign_detections_to_trackers(frame_idx,
                                                                              obj_history,
                                                                              tracker_list_idx,
                                                                              detection_list)
        tmp_obj_history = []
        for obj_idx, obj_info in enumerate(obj_history):
            if len(obj_info[2]) > 10:
                tmp_obj_history.append(obj_history[obj_idx])

        obj_history = tmp_obj_history

        trans_mat = np.eye(8)
        trans_mat[0, 4] = 3
        trans_mat[1, 5] = 3
        trans_mat[2, 6] = 3
        trans_mat[3, 7] = 3

        observation_mat = np.zeros((4, 8))
        observation_mat[0, 0] = 1
        observation_mat[1, 1] = 1
        observation_mat[2, 2] = 1
        observation_mat[3, 3] = 1

        for obj_idx, obj_info in enumerate(obj_history):
            start_frame = obj_info[0]
            measurements = []

            for i in range(len(obj_info[2])):
                x1, y1, x2, y2, score = obj_info[2][i][:5]
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                width, height = x2 - x1, y2 - y1
                measurements.append((center_x, center_y, width, height, score))

            first_measurement = measurements[0]
            init_x1, init_y1, init_x2, init_y2 = first_measurement[:4]
            kf = KalmanFilter(initial_state_mean=[init_x1, init_y1, init_x2, init_y2, 0, 0, 0, 0],
                              transition_matrices=trans_mat,
                              observation_matrices=observation_mat)

            measurements = np.asarray(measurements)

            kf = kf.em(measurements[:, :4], n_iter=5)
            (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements[:, :4])

            output = smoothed_state_means[:, :4]
            for idx, prediction in enumerate(output):
                center_x, center_y, width, height = prediction
                x1 = center_x - (width / 2)
                x2 = center_x + (width / 2)
                y1 = center_y - (height / 2)
                y2 = center_y + (height / 2)

                path = frames[start_frame + idx]
                output_frame_dict[path][obj_idx] = [x1, y1, x2, y2, measurements[idx, 4]]

        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        out = cv2.VideoWriter(video+"_h.mp4", fourcc, 5, (WIDTH, HEIGHT))

        for frame in frames[:300]:
            frame_path = os.path.join(frame_base, frame)
            img = cv2.imread(frame_path)
            # height, width, channels = img.shape
            for hand_idx in output_frame_dict[frame]:
                hand_bbox = output_frame_dict[frame][hand_idx]
                x1, y1, x2, y2 = hand_bbox[:4]
                r, g, b = hand_colors[hand_idx]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (int(r), int(g), int(b)), 2)
                cv2.putText(img, "hand" + str(hand_idx), (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (int(r), int(g), int(b)), 2)
            out.write(img)

        cv2.destroyAllWindows()
        out.release()


if __name__ == '__main__':
    main()

