import configparser
import csv
import os
import os.path as osp

import numpy as np
from PIL import Image

import json

from shapely.geometry import box, Polygon, MultiPolygon

def ignore_regions():
    # remove boxes from in ignore regions
    ignore_box_candidates = set()
    ##########################
    ignore_iou_thres = 0.1
    ##########################

    if len(ignore_x) > 0:
        if not isinstance(ignore_x[0], list):
            ignore_x = [ignore_x]
            ignore_y = [ignore_y]

        # build ignore regions:
        ignore_regions = []
        for r_idx in range(len(ignore_x)):
            region = []

            for x, y in zip(ignore_x[r_idx], ignore_y[r_idx]):
                region.append([x, y])

            ignore_region = Polygon(region)
            ignore_regions.append(ignore_region)

        region_ious = np.zeros((len(ignore_regions), num_det), dtype=np.float32)
        det_boxes = []
        for j in rrange(num_det):
            x1 = det[j, 0]
            y1 = det[j, 1]
            x2 = det[j, 2]
            y2 = det[j, 3]

            box_poly = Polygon([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2],
                [x1, y1],
            ])

            det_boxes.append(box_poly)

        for i in xrange(len(ignore_regions)):
            for j in xrange(num_det):
                if ignore_regions[i].is_valid:
                    poly_intersection = ignore_regions[i].intersection(det_boxes[j]).area
                    poly_union = ignore_regions[i].union(det_boxes[j]).area
                else:
                    multi_poly = ignore_regions[i].buffer(0)
                    poly_intersection = 0
                    poly_union = 0

                    if isinstance(multi_poly, Polygon):
                        poly_intersection = multi_poly.intersection(det_boxes[j]).area
                        poly_union = multi_poly.union(det_boxes[j]).area
                    else:
                        multi_poly = ignore_regions[i].buffer(0)
                        poly_intersection = 0
                        poly_union = 0

                        if isinstance(multi_poly, Polygon):
                            poly_intersection = multi_poly.intersection(det_boxes[j]).area
                            poly_union = multi_poly.union(det_boxes[j]).area
                        else:
                            for poly in multi_poly:
                                poly_intersection += poly.intersection(det_boxes[j]).area
                                poly_union += poly.union(det_boxes[j]).area

                    region_ious[i, j] = poly_intersection / poly_union

                    candidates = np.argwhere(region_ious[i] > ignore_iou_thres)
                    if len(candidates) > 0:
                        candidates = candidates[:, 0].tolist()
                        ignore_box_candidates.update(candidates)

                ious = np.zeros((num_gt, num_det), dtype=np.float32)
                for i in xrange(num_gt):
                    for j in xrange(num_det):
                        ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
                tfmat = (ious >= iou_thresh)
                # for each det, keep only the largest iou of all the gt
                for j in xrange(num_det):
                    largest_ind = np.argmax(ious[:, j])
                    for i in xrange(num_gt):
                        if i != largest_ind:
                            tfmat[i, j] = False
                # for each gt, keep only the largest iou of all the det
                for i in xrange(num_gt):
                    largest_ind = np.argmax(ious[i, :])
                    for j in xrange(num_det):
                        if j != largest_ind:
                            tfmat[i, j] = False
                for j in xrange(num_det):
                    if j in ignore_box_candidates and not tfmat[:, j].any():
                        # we have a detection in ignore region
                        detections_to_ignore[gallery_idx].append(j)
                        continue

                    y_score.append(det[j, -1])
                    if tfmat[:, j].any():
                        y_true.append(True)
                    else:
                        y_true.append(False)
                count_tp += tfmat.sum()
                count_gt += num_gt


class PTSequence():
    """Multiple Object Tracking Dataset.
    """
    def __init__(self, seq_name, mot_dir, dataset_path, vis_threshold=0.0):
        self._seq_name = seq_name
        self._vis_threshold = vis_threshold

        self._mot_dir = mot_dir
        self.dataset_path = dataset_path
        self._folders = os.listdir(self._mot_dir)

        if seq_name is not None:
            assert seq_name in self._folders, \
                'Image set does not exist: {}'.format(seq_name)

            self.data, self.no_gt, self.ignore = self._sequence()
        else:
            self.data = []
            self.no_gt = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        data = self.data[idx]
        img = Image.open(data['im_path']).convert("RGB")
        img = np.asarray(img)

        sample = {}
        sample['img'] = img
        sample['dets'] = np.array([det[:4] for det in data['dets']])
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']

        return sample

    def _sequence(self):
        seq_name = self._seq_name
        seq_path = osp.join(self._mot_dir, seq_name)

        config_file = os.path.join(seq_path, 'image_info.json')

        assert osp.exists(config_file), \
            'Config file does not exist: {}'.format(config_file)

        with open(config_file, 'r') as file:
            image_info = json.load(file)

        seqLength = len(image_info)
        gt_file = osp.join(seq_path, 'gt', 'gt.txt')

        total = []

        visibility = {}
        boxes = {}
        dets = {}
        ignore_x = {}
        ignore_y = {}

        for i in range(seqLength):
            frame_index = image_info[i]['frame_index']
            boxes[frame_index] = {}
            visibility[frame_index] = {}
            dets[frame_index] = []
            ignore_x[frame_index] = image_info[i]['ignore_regions_x']
            ignore_y[frame_index] = image_info[i]['ignore_regions_y']

        no_gt = False
        if osp.exists(gt_file):
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    # class person, certainity 1, visibility >= 0.25
                    row[7] = '1'
                    row[8] = '1'
                    if int(row[6]) == 1 and int(row[7]) == 1 and float(row[8]) >= self._vis_threshold:
                        # Make pixel indexes 0-based, should already be 0-based (or not)
                        x1 = int(float(row[2])) # - 1
                        y1 = int(float(row[3])) # - 1
                        # This -1 accounts for the width (width of 1 x1=x2)
                        x2 = x1 + int(float(row[4])) #- 1
                        y2 = y1 + int(float(row[5])) # - 1
                        bb = np.array([x1,y1,x2,y2], dtype=np.float32)

                        boxes[int(row[0])][int(row[1])] = bb
                        visibility[int(row[0])][int(row[1])] = float(row[8])
        else:
            no_gt = True

        det_file = osp.join(seq_path, 'det', 'det.txt')

        if osp.exists(det_file):
            with open(det_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    x1 = float(row[2]) - 1
                    y1 = float(row[3]) - 1
                    # This -1 accounts for the width (width of 1 x1=x2)
                    x2 = x1 + float(row[4]) - 1
                    y2 = y1 + float(row[5]) - 1
                    score = float(row[6])
                    bb = np.array([x1,y1,x2,y2, score], dtype=np.float32)
                    dets[int(float(row[0]))].append(bb)

        dataset_path = self.dataset_path
        for i in range(seqLength):
            im_path = os.path.join(dataset_path, image_info[i]['file_name'])
            frame_index = image_info[i]['frame_index']

            sample = {'gt':boxes[frame_index],
                      'im_path':im_path,
                      'vis':visibility[frame_index],
                      'dets':dets[frame_index]}

            total.append(sample)

        return total, no_gt, {'ignore_x': ignore_x, 'ignore_y': ignore_y}

    def __str__(self):
        return self._seq_name

    def write_results(self, all_tracks, output_dir):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        """

        #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(osp.join(output_dir, self._seq_name), "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow(
                        [frame + 1,
                         i + 1,
                         x1 + 1,
                         y1 + 1,
                         x2 - x1 + 1,
                         y2 - y1 + 1,
                         -1, -1, -1, -1])

    def load_results(self, output_dir):
        file_path = osp.join(output_dir, self._seq_name)
        results = {}

        if not os.path.isfile(file_path):
            if os.path.isfile(f'{file_path}.txt'):
                file_path = f'{file_path}.txt'
            else:
                return results

        with open(file_path, "r") as of:
            csv_reader = csv.reader(of, delimiter=',')
            for row in csv_reader:
                frame_id, track_id = int(row[0]) - 1, int(row[1]) - 1

                if not track_id in results:
                    results[track_id] = {}

                x1 = float(row[2]) - 1
                y1 = float(row[3]) - 1
                x2 = float(row[4]) - 1 + x1
                y2 = float(row[5]) - 1 + y1

                results[track_id][frame_id] = [x1, y1, x2, y2]

        return results

