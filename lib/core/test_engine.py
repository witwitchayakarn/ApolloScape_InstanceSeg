# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as patches

import datetime
import logging
import numpy as np
import os
import yaml

import torch
from core.config import cfg
from core.test import im_detect_all
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from modeling import model_builder
import nn as mynn
import utils.env as envu
import utils.net as net_utils
import utils.subprocess as subprocess_utils
import utils.vis as vis_utils
from utils.io import save_object, load_object
from utils.timer import Timer
from tqdm import tqdm
from utilities.eval_car_instances import Detect3DEval

logger = logging.getLogger(__name__)


def get_eval_functions(dataset_name):
    # Determine which parent or child function should handle inference
    if cfg.MODEL.RPN_ONLY:
        raise NotImplementedError
        # child_func = generate_rpn_on_range
        # parent_func = generate_rpn_on_dataset
    elif dataset_name[0] == 'Car3D':
        child_func = test_net_Car3D
        parent_func = test_net_Car3D
    else:
        # Generic case that handles all network types other than RPN-only nets
        # and RetinaNet
        child_func = test_net
        parent_func = test_net_on_dataset

    return parent_func, child_func


def get_inference_dataset(index, is_parent=True):
    assert is_parent or len(cfg.TEST.DATASETS) == 1, \
        'The child inference process can only work on a single dataset'

    dataset_name = cfg.TEST.DATASETS[index]

    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert is_parent or len(cfg.TEST.PROPOSAL_FILES) == 1, \
            'The child inference process can only work on a single proposal file'
        assert len(cfg.TEST.PROPOSAL_FILES) == len(cfg.TEST.DATASETS), \
            'If proposals are used, one proposal file must be specified for ' \
            'each dataset'
        proposal_file = cfg.TEST.PROPOSAL_FILES[index]
    else:
        proposal_file = None

    return dataset_name, proposal_file


def run_inference(
        args, ind_range=None,
        multi_gpu_testing=False, gpu_id=0,
        check_expected_results=False):
    parent_func, child_func = get_eval_functions(cfg.TEST.DATASETS)
    is_parent = ind_range is None

    def result_getter():
        if is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            all_results = {}
            for i in range(len(cfg.TEST.DATASETS)):
                dataset_name, proposal_file = get_inference_dataset(i)
                output_dir = args.output_dir
                results = parent_func(
                    args,
                    dataset_name,
                    proposal_file,
                    output_dir,
                    multi_gpu=multi_gpu_testing,
                )
                all_results.update(results)

            return all_results
        else:
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
            output_dir = args.output_dir
            return child_func(
                args,
                dataset_name,
                proposal_file,
                output_dir,
                ind_range=ind_range,
                gpu_id=gpu_id
            )

    all_results = result_getter()
    if check_expected_results and is_parent:
        task_evaluation.check_expected_results(
            all_results,
            atol=cfg.EXPECTED_RESULTS_ATOL,
            rtol=cfg.EXPECTED_RESULTS_RTOL
        )
        task_evaluation.log_copy_paste_friendly_results(all_results)

    return all_results


def run_inference_wad(args, ind_range=None, multi_gpu_testing=False, gpu_id=0):
    parent_func, child_func = get_eval_functions()
    is_parent = ind_range is None

    def result_getter():
        if is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            all_results = {}
            for i in range(len(cfg.TEST.DATASETS)):
                dataset_name, proposal_file = get_inference_dataset(i)
                output_dir = args.output_dir
                results = parent_func(
                    args,
                    dataset_name,
                    proposal_file,
                    output_dir,
                    multi_gpu=multi_gpu_testing
                )
                all_results.update(results)

            return all_results
        else:
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
            output_dir = args.output_dir
            return child_func(
                args,
                dataset_name,
                proposal_file,
                output_dir,
                ind_range=ind_range,
                gpu_id=gpu_id
            )

    all_results = result_getter()

    #if check_expected_results and is_parent:
    if True:
        task_evaluation.check_expected_results(
            all_results,
            atol=cfg.EXPECTED_RESULTS_ATOL,
            rtol=cfg.EXPECTED_RESULTS_RTOL
        )
        task_evaluation.log_copy_paste_friendly_results(all_results)

    return all_results


def test_net_on_dataset(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        multi_gpu=False,
        gpu_id=0):
    """Run inference on a dataset."""
    dataset = JsonDataset(dataset_name)
    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        num_images = len(dataset.get_roidb(gt=True))
        all_boxes, all_segms, all_keyps = multi_gpu_test_net_on_dataset(
            args, dataset_name, proposal_file, num_images, output_dir
        )
    else:
        all_boxes, all_segms, all_keyps = test_net(args, dataset_name, proposal_file, output_dir, gpu_id=gpu_id)
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))
    results = task_evaluation.evaluate_all(dataset, all_boxes, all_segms, all_keyps, output_dir)
    return results


def multi_gpu_test_net_on_dataset(
        args, dataset_name, proposal_file, num_images, output_dir):
    """Multi-gpu inference on a dataset."""
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, args.test_net_file + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Pass the target dataset and proposal file (if any) via the command line
    opts = ['TEST.DATASETS', '("{}",)'.format(dataset_name)]
    if proposal_file:
        opts += ['TEST.PROPOSAL_FILES', '("{}",)'.format(proposal_file)]

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = subprocess_utils.process_in_parallel(
        'detection', num_images, binary, output_dir,
        args.load_ckpt, opts)

    # Collate the results from each subprocess
    all_boxes = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_segms = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_keyps = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    for det_data in outputs:
        all_boxes_batch = det_data['all_boxes']
        all_segms_batch = det_data['all_segms']
        all_keyps_batch = det_data['all_keyps']
        for cls_idx in range(1, cfg.MODEL.NUM_CLASSES):
            all_boxes[cls_idx] += all_boxes_batch[cls_idx]
            all_segms[cls_idx] += all_segms_batch[cls_idx]
            all_keyps[cls_idx] += all_keyps_batch[cls_idx]
    det_file = os.path.join(output_dir, 'detections.pkl')
    cfg_yaml = yaml.dump(cfg)
    save_object(
        dict(
            all_boxes=all_boxes,
            all_segms=all_segms,
            all_keyps=all_keyps,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))

    return all_boxes, all_segms, all_keyps


def test_net(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        ind_range=None,
        gpu_id=0):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    assert not cfg.MODEL.RPN_ONLY, 'Use rpn_generate to generate proposals from RPN-only models'
    dataset = JsonDataset(dataset_name, args.dataset_dir)
    timers = defaultdict(Timer)
    if ind_range is not None:
        if cfg.TEST.SOFT_NMS.ENABLED:
            det_name = 'detection_range_%s_%s_soft_nms.pkl' % tuple(ind_range)
        else:
            det_name = 'detection_range_(%d_%d)_nms_%.1f.pkl' % (ind_range[0], ind_range[1], cfg.TEST.NMS)
    else:
        det_name = 'detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(dataset, proposal_file, ind_range, args)
    num_images = len(roidb)
    image_ids = []
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)

    for i, entry in enumerate(roidb):
        image_ids.append(entry['image'])
    args.image_ids = image_ids

    # If we have already computed the boxes
    if os.path.exists(det_file):
        obj = load_object(det_file)
        all_boxes, all_segms, all_keyps = obj['all_boxes'], obj['all_segms'], obj['all_keyps']

    else:
        model = initialize_model_from_cfg(args, gpu_id=gpu_id)
        for i, entry in enumerate(roidb):
            if cfg.TEST.PRECOMPUTED_PROPOSALS:
                # The roidb may contain ground-truth rois (for example, if the roidb
                # comes from the training or val split). We only want to evaluate
                # detection on the *non*-ground-truth rois. We select only the rois
                # that have the gt_classes field set to 0, which means there's no
                # ground truth.
                box_proposals = entry['boxes'][entry['gt_classes'] == 0]
                if len(box_proposals) == 0:
                    continue
            else:
                # Faster R-CNN type models generate proposals on-the-fly with an
                # in-network RPN; 1-stage models don't require proposals.
                box_proposals = None

            im = cv2.imread(entry['image'])
            cls_boxes_i, cls_segms_i, cls_keyps_i, car_cls_i, euler_angle_i, trans_pred_i = im_detect_all(model, im, box_proposals, timers, dataset)
            extend_results(i, all_boxes, cls_boxes_i)
            if cls_segms_i is not None:
                extend_results(i, all_segms, cls_segms_i)
            if cls_keyps_i is not None:
                extend_results(i, all_keyps, cls_keyps_i)

            if i % 10 == 0:  # Reduce log file size
                ave_total_time = np.sum([t.average_time for t in timers.values()])
                eta_seconds = ave_total_time * (num_images - i - 1)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                det_time = (
                    timers['im_detect_bbox'].average_time +
                    timers['im_detect_mask'].average_time +
                    timers['im_detect_keypoints'].average_time
                )
                misc_time = (
                    timers['misc_bbox'].average_time +
                    timers['misc_mask'].average_time +
                    timers['misc_keypoints'].average_time
                )
                logger.info(
                    (
                        'im_detect: range [{:d}, {:d}] of {:d}: '
                        '{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
                    ).format(
                        start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                        start_ind + num_images, det_time, misc_time, eta
                    )
                )

            if cfg.VIS:
                im_name = os.path.splitext(os.path.basename(entry['image']))[0]
                vis_utils.vis_one_image_eccv2018_car_3d(
                    im[:, :, ::-1],
                    '{:d}_{:s}'.format(i, im_name),
                    os.path.join(output_dir, 'vis'),
                    boxes=cls_boxes_i,
                    car_cls_prob=car_cls_i,
                    euler_angle=euler_angle_i,
                    trans_pred=trans_pred_i,
                    car_models=dataset.Car3D.car_models,
                    intrinsic=dataset.Car3D.get_intrinsic_mat(),
                    segms=cls_segms_i,
                    keypoints=cls_keyps_i,
                    thresh=0.9,
                    box_alpha=0.8,
                    dataset=dataset.Car3D)
        cfg_yaml = yaml.dump(cfg)
        save_object(
            dict(
                all_boxes=all_boxes,
                all_segms=all_segms,
                all_keyps=all_keyps,
                cfg=cfg_yaml
            ), det_file
        )
        logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))

    results = task_evaluation.evaluate_all(dataset, all_boxes, all_segms, all_keyps, output_dir, args)
    return results


def test_net_Car3D(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        ind_range=None,
        gpu_id=0):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    assert not cfg.MODEL.RPN_ONLY, 'Use rpn_generate to generate proposals from RPN-only models'
    dataset = JsonDataset(dataset_name, args.dataset_dir)
    timers = defaultdict(Timer)

    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(dataset, proposal_file, ind_range, args)
    num_images = len(roidb)
    image_ids = []
    if cfg.MODEL.TRANS_HEAD_ON:
        json_dir = os.path.join(output_dir, 'json_'+args.list_flag+'_trans')
    else:
        json_dir = os.path.join(output_dir, 'json_'+args.list_flag)

    json_dir += '_iou_' + str(args.iou_ignore_threshold)
    if not cfg.TEST.BBOX_AUG.ENABLED:
        json_dir += '_BBOX_AUG_single_scale'
    else:
        json_dir += '_BBOX_AUG_multiple_scale'

    if not cfg.TEST.CAR_CLS_AUG.ENABLED:
        json_dir += '_CAR_CLS_AUG_single_scale'
    else:
        json_dir += '_CAR_CLS_AUG_multiple_scale'

    if cfg.TEST.GEOMETRIC_TRANS:
        json_dir += '_GEOMETRIC_TRANS'

    if cfg.TEST.CAR_CLS_AUG.H_FLIP and cfg.TEST.CAR_CLS_AUG.SCALE_H_FLIP:
        json_dir += '_hflipped'

    roidb = roidb
    for i, entry in enumerate(roidb):
        image_ids.append(entry['image'])
    args.image_ids = image_ids

    all_boxes = [[[] for _ in range(num_images)] for _ in range(cfg.MODEL.NUM_CLASSES)]
    if ind_range is not None:
        if cfg.TEST.SOFT_NMS.ENABLED:
            det_name = 'detection_range_%s_%s_soft_nms' % tuple(ind_range)
        else:
            det_name = 'detection_range_(%d_%d)_nms_%.1f' % (ind_range[0], ind_range[1], cfg.TEST.NMS)
        if cfg.TEST.BBOX_AUG.ENABLED:
            det_name += '_multiple_scale'
        det_name += '.pkl'
    else:
        det_name = 'detections.pkl'
    det_file = os.path.join(output_dir, det_name)

    file_complete_flag = [not os.path.exists(os.path.join(json_dir, entry['image'].split('/')[-1][:-4] + '.json')) for entry in roidb]
    # If we don't have the complete json file, we will load the model and execute the following:
    if np.sum(file_complete_flag) or not os.path.exists(det_file):
        model = initialize_model_from_cfg(args, gpu_id=gpu_id)
        for i in tqdm(range(len(roidb))):
            entry = roidb[i]
            if cfg.TEST.PRECOMPUTED_PROPOSALS:
                # The roidb may contain ground-truth rois (for example, if the roidb
                # comes from the training or val split). We only want to evaluate
                # detection on the *non*-ground-truth rois. We select only the rois
                # that have the gt_classes field set to 0, which means there's no
                # ground truth.
                box_proposals = entry['boxes'][entry['gt_classes'] == 0]
                if len(box_proposals) == 0:
                    continue
            else:
                # Faster R-CNN type models generate proposals on-the-fly with an
                # in-network RPN; 1-stage models don't require proposals.
                box_proposals = None

            im = cv2.imread(entry['image'])
            ignored_mask_img = os.path.join(('/').join(entry['image'].split('/')[:-2]), 'ignore_mask', entry['image'].split('/')[-1])
            ignored_mask = cv2.imread(ignored_mask_img, cv2.IMREAD_GRAYSCALE)
            if ignored_mask is None:
                ignored_mask = np.zeros(im.shape, dtype='uint8')
            ignored_mask_binary = np.zeros(ignored_mask.shape)
            ignored_mask_binary[ignored_mask > 250] = 1
            if cfg.MODEL.NON_LOCAL_TEST and not cfg.TEST.BBOX_AUG.ENABLED:
                cls_boxes_i, cls_segms_i, _, car_cls_i, euler_angle_i, trans_pred_i, f_div_C = im_detect_all(model, im, box_proposals, timers, dataset)
            else:
                cls_boxes_i, cls_segms_i, _, car_cls_i, euler_angle_i, trans_pred_i = im_detect_all(model, im, box_proposals, timers, dataset)
            extend_results(i, all_boxes, cls_boxes_i)

            # We draw the grid overlap with an image here
            if False:
                f_div_C_plot = f_div_C.copy()
                grid_size = 32  # This is the res5 output space
                fig = plt.figure()
                ax1 = fig.add_subplot(1, 2, 1)
                ax2 = fig.add_subplot(1, 2, 2)
                ax1.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                ax1.grid(which='minor')

                # We choose the point here:
                # x, y = int(1757/grid_size), int(1040/grid_size)   # val 164
                x, y = int(1830/grid_size), int(1855/grid_size)

                # draw a patch hre
                rect = patches.Rectangle((x*grid_size-grid_size, y*grid_size-grid_size), grid_size*3, grid_size*3,
                                         linewidth=1, edgecolor='m', facecolor='m')
                ax1.add_patch(rect)
                #att_point_map = f_div_C_plot[106*x+y, :]
                att_point_map = f_div_C_plot[106*y+x, :]
                att_point_map = np.reshape(att_point_map, (85, 106))
                ax2.imshow(att_point_map, cmap='jet')

                # we draw 20 arrows
                for i in range(10):
                    x_max, y_max = np.unravel_index(att_point_map.argmax(), att_point_map.shape)
                    v = att_point_map[x_max, y_max]
                    att_point_map[x_max, y_max] = 0
                    ax1.arrow(x*grid_size, y*grid_size, (y_max-x)*grid_size, (x_max-y)*grid_size,
                              fc="r", ec="r", head_width=(10-i)*grid_size/2, head_length=grid_size)

            if i % 10 == 0:  # Reduce log file size
                ave_total_time = np.sum([t.average_time for t in timers.values()])
                eta_seconds = ave_total_time * (num_images - i - 1)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                det_time = timers['im_detect_bbox'].average_time
                triple_head_time = timers['triple_head'].average_time
                misc_time = (
                    timers['misc_bbox'].average_time +
                    timers['misc_mask'].average_time
                )
                logger.info(
                    (
                        'im_detect: range [{:d}, {:d}] of {:d}: '
                        '{:d}/{:d} det-time: {:.3f}s + triple-head-time: {:.3f}s + misc_time: {:.3f}s (eta: {})'
                    ).format(
                        start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                        start_ind + num_images, det_time, triple_head_time, misc_time, eta
                    )
                )

            im_name = os.path.splitext(os.path.basename(entry['image']))[0]
            vis_utils.write_pose_to_json(
                im_name=im_name,
                output_dir=json_dir,
                boxes=cls_boxes_i,
                car_cls_prob=car_cls_i,
                euler_angle=euler_angle_i,
                trans_pred=trans_pred_i,
                segms=cls_segms_i,
                dataset=dataset.Car3D,
                thresh=cfg.TEST.SCORE_THRESH_FOR_TRUTH_DETECTION,
                ignored_mask_binary=ignored_mask_binary.astype('uint8'),
                iou_ignore_threshold=args.iou_ignore_threshold
            )

            if cfg.VIS:
                vis_utils.vis_one_image_eccv2018_car_3d(
                    im[:, :, ::-1],
                    '{:d}_{:s}'.format(i, im_name),
                    os.path.join(output_dir, 'vis_'+args.list_flag),
                    boxes=cls_boxes_i,
                    car_cls_prob=car_cls_i,
                    euler_angle=euler_angle_i,
                    trans_pred=trans_pred_i,
                    car_models=dataset.Car3D.car_models,
                    intrinsic=dataset.Car3D.get_intrinsic_mat(),
                    segms=cls_segms_i,
                    keypoints=None,
                    thresh=0.9,
                    box_alpha=0.8,
                    dataset=dataset.Car3D)

        save_object(dict(all_boxes=all_boxes), det_file)

    # The following evaluate the detection result from Faster-RCNN Head
    # If we have already computed the boxes
    if os.path.exists(det_file):
        obj = load_object(det_file)
        all_boxes = obj['all_boxes']

    # this is a hack
    if False:

        import glob
        det_files = sorted(glob.glob(args.output_dir+'/detection_range_*.pkl'))
        det_files = [det_files[4], det_files[1], det_files[2], det_files[3]]
        obj = load_object(det_files[0])
        all_boxes = obj['all_boxes']
        for df in det_files:
            obj = load_object(df)
            boxes = obj['all_boxes']
            for i in range(len(boxes)):
                all_boxes[i] = all_boxes[i] + boxes[i]
        save_object(dict(all_boxes=all_boxes), det_file)

    results = task_evaluation.evaluate_boxes(dataset, all_boxes, output_dir, args)

    # The following evaluate the mAP of car poses
    args.test_dir = json_dir
    args.gt_dir = args.dataset_dir + 'car_poses'
    args.res_file = os.path.join(output_dir, 'json_'+args.list_flag+'_res.txt')
    args.simType = None
    det_3d_metric = Detect3DEval(args)
    det_3d_metric.evaluate()
    det_3d_metric.accumulate()
    det_3d_metric.summarize()


def initialize_model_from_cfg(args, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = model_builder.Generalized_RCNN()
    model.eval()

    if args.cuda:
        model.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])

    model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True)

    return model


def get_roidb_and_dataset(dataset, proposal_file, ind_range, args):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert proposal_file, 'No proposal file given'
        roidb = dataset.get_roidb(
            proposal_file=proposal_file,
            proposal_limit=cfg.TEST.PROPOSAL_LIMIT
        )
    else:
        roidb = dataset.get_roidb(gt=True, list_flag=args.list_flag)
        dataset.roidb = roidb

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images


def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes, all_segms, all_keyps


def empty_results_car_3d(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_car_cls = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_pose = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes, all_segms, all_car_cls, all_pose


def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]
