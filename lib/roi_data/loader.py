import math

import numpy as np
import numpy.random as npr
import torch.utils.data as data
import torch.utils.data.sampler as torch_sampler
from pycocotools import mask as COCOmask
from torch._six import int_classes as _int_classes
from torch.utils.data.dataloader import default_collate

import utils.blob as blob_utils
from core.config import cfg
from roi_data.minibatch import get_minibatch

from utils.transform import RandomTransformPixels

class RoiDataLoader(data.Dataset):
    def __init__(self, roidb, num_classes, training=True, valid_keys=[]):
        """
        :param roidb:
        :param num_classes:
        :param training:
        :param roidb_extra_keys: is Wudi's customized keys for input
        """
        self._roidb = roidb
        self._num_classes = num_classes
        self.training = training
        self.DATA_SIZE = len(self._roidb)
        self.valid_keys = valid_keys
        self.transform = RandomTransformPixels()

    def __getitem__(self, index_tuple):
        index, ratio = index_tuple
        single_db = [self._roidb[index]]
        blobs, valid = get_minibatch(single_db, self.transform, self.valid_keys)
        # TODO: Check if minibatch is valid ? If not, abandon it.
        # Need to change _worker_loop in torch.utils.data.dataloader.py.

        # Squeeze batch dim
        for key in blobs:
            if key != 'roidb':
                blobs[key] = blobs[key].squeeze(axis=0)

        if self._roidb[index]['need_crop']:
            self.crop_data(blobs, ratio)
            # Check bounding box
            entry = blobs['roidb'][0]
            boxes = entry['boxes']
            invalid = (boxes[:, 0] == boxes[:, 2]) | (boxes[:, 1] == boxes[:, 3])
            valid_inds = np.nonzero(~ invalid)[0]
            if len(valid_inds) < len(boxes):
                for key in ['boxes', 'gt_classes', 'seg_areas', 'gt_overlaps', 'is_crowd',
                            'box_to_gt_ind_map', 'gt_keypoints']:
                    if key in entry:
                        entry[key] = entry[key][valid_inds]
                entry['segms'] = [entry['segms'][ind] for ind in valid_inds]

        if cfg.TRAIN.RANDOM_CROP > 0:
            if 'segms_origin' not in blobs['roidb'][0].keys():
                blobs['roidb'][0]['segms_origin'] = blobs['roidb'][0]['segms'].copy()

            self.crop_data_train(blobs)
            # Check bounding box, actually, it is not necessary...
            # entry = blobs['roidb'][0]
            # boxes = entry['boxes']
            # invalid = (boxes[:, 0] < 0) | (boxes[:, 2] < 0)
            # valid_inds = np.nonzero(~ invalid)[0]
            # if len(valid_inds) < len(boxes):
            #     for key in ['boxes', 'gt_classes', 'seg_areas', 'gt_overlaps', 'is_crowd']:
            #         if key in entry:
            #             entry[key] = entry[key][valid_inds]
            #
            #     entry['box_to_gt_ind_map'] = np.array(list(range(len(valid_inds)))).astype(int)
            #     entry['segms'] = [entry['segms'][ind] for ind in valid_inds]

        blobs['roidb'] = blob_utils.serialize(blobs['roidb'])  # CHECK: maybe we can serialize in collate_fn
        return blobs

    def crop_data(self, blobs, ratio):
        data_height, data_width = map(int, blobs['im_info'][:2])
        boxes = blobs['roidb'][0]['boxes']
        if ratio < 1:  # width << height, crop height
            size_crop = math.ceil(data_width / ratio)  # size after crop
            min_y = math.floor(np.min(boxes[:, 1]))
            max_y = math.floor(np.max(boxes[:, 3]))
            box_region = max_y - min_y + 1
            if min_y == 0:
                y_s = 0
            else:
                if (box_region - size_crop) < 0:
                    y_s_min = max(max_y - size_crop, 0)
                    y_s_max = min(min_y, data_height - size_crop)
                    y_s = y_s_min if y_s_min == y_s_max else \
                        npr.choice(range(y_s_min, y_s_max + 1))
                else:
                    # CHECK: rethinking the mechnism for the case box_region > size_crop
                    # Now, the crop is biased on the lower part of box_region caused by
                    # // 2 for y_s_add
                    y_s_add = (box_region - size_crop) // 2
                    y_s = min_y if y_s_add == 0 else \
                        npr.choice(range(min_y, min_y + y_s_add + 1))
            # Crop the image
            blobs['data'] = blobs['data'][:, y_s:(y_s + size_crop), :, ]
            # Update im_info
            blobs['im_info'][0] = size_crop
            # Shift and clamp boxes ground truth
            boxes[:, 1] -= y_s
            boxes[:, 3] -= y_s
            np.clip(boxes[:, 1], 0, size_crop - 1, out=boxes[:, 1])
            np.clip(boxes[:, 3], 0, size_crop - 1, out=boxes[:, 3])
            blobs['roidb'][0]['boxes'] = boxes
        else:  # width >> height, crop width
            size_crop = math.ceil(data_height * ratio)
            min_x = math.floor(np.min(boxes[:, 0]))
            max_x = math.floor(np.max(boxes[:, 2]))
            box_region = max_x - min_x + 1
            if min_x == 0:
                x_s = 0
            else:
                if (box_region - size_crop) < 0:
                    x_s_min = max(max_x - size_crop, 0)
                    x_s_max = min(min_x, data_width - size_crop)
                    x_s = x_s_min if x_s_min == x_s_max else npr.choice(range(x_s_min, x_s_max + 1))
                else:
                    x_s_add = (box_region - size_crop) // 2
                    x_s = min_x if x_s_add == 0 else npr.choice(range(min_x, min_x + x_s_add + 1))
            # Crop the image
            blobs['data'] = blobs['data'][:, :, x_s:(x_s + size_crop)]
            # Update im_info
            blobs['im_info'][1] = size_crop
            # Shift and clamp boxes ground truth
            boxes[:, 0] -= x_s
            boxes[:, 2] -= x_s
            np.clip(boxes[:, 0], 0, size_crop - 1, out=boxes[:, 0])
            np.clip(boxes[:, 2], 0, size_crop - 1, out=boxes[:, 2])
            blobs['roidb'][0]['boxes'] = boxes

    def crop_data_train(self, blobs):
        data_height, data_width = map(int, blobs['im_info'][:2])
        boxes = blobs['roidb'][0]['boxes']
        crop_ratio = blobs['im_info'][2]

        min_x = math.floor(np.min(boxes[:, 0]) * crop_ratio)
        max_x = math.floor(np.max(boxes[:, 2]) * crop_ratio)
        min_y = math.floor(np.min(boxes[:, 1]) * crop_ratio)
        max_y = math.floor(np.max(boxes[:, 3]) * crop_ratio)

        x_s_max = max(max_x - cfg.TRAIN.RANDOM_CROP, 0)
        x_s_min = min(min_x, data_width - cfg.TRAIN.RANDOM_CROP)
        x_start = x_s_min if x_s_min >= x_s_max else npr.choice(range(x_s_min, x_s_max + 1))

        y_s_max = max(max_y - cfg.TRAIN.RANDOM_CROP, 0)
        y_s_min = min(min_y, data_height - cfg.TRAIN.RANDOM_CROP)
        y_start = y_s_min if y_s_min >= y_s_max else npr.choice(range(y_s_min, y_s_max + 1))

        # Crop the image
        blobs['data'] = blobs['data'][:, y_start:(y_start + cfg.TRAIN.RANDOM_CROP),
                        x_start:(x_start + cfg.TRAIN.RANDOM_CROP)]
        # Update im_info
        blobs['im_info'][:2] = cfg.TRAIN.RANDOM_CROP, cfg.TRAIN.RANDOM_CROP
        # Shift and clamp boxes ground truth
        boxes[:, 0] -= math.ceil(x_start / crop_ratio)
        boxes[:, 2] -= math.ceil(x_start / crop_ratio)
        boxes[:, 1] -= math.ceil(y_start / crop_ratio)
        boxes[:, 3] -= math.ceil(y_start / crop_ratio)
        blobs['roidb'][0]['boxes'] = boxes

        # Unfortunately, we need to transform the semgs
        h, w = blobs['roidb'][0]['segms_origin'][0]['size']
        h_new, w_new = cfg.TRAIN.RANDOM_CROP / crop_ratio, cfg.TRAIN.RANDOM_CROP / crop_ratio

        semgs_masks = np.zeros(shape=(len(blobs['roidb'][0]['segms_origin']), h, w))
        for i in range(len(blobs['roidb'][0]['segms_origin'])):
            semgs_masks[i, :, :] = COCOmask.decode(blobs['roidb'][0]['segms_origin'][i])

        semgs_masks_new = semgs_masks[:, math.ceil(y_start / crop_ratio): (math.ceil(y_start / crop_ratio + h_new)),
                          math.ceil(x_start / crop_ratio): (math.ceil(x_start / crop_ratio + h_new))]

        blobs['roidb'][0]['segms'] = []
        for i in range(semgs_masks_new.shape[0]):
            mask = np.array(semgs_masks_new[i], order='F', dtype=np.uint8)
            rle = COCOmask.encode(mask)
            blobs['roidb'][0]['segms'].append(rle)

    def __len__(self):
        return self.DATA_SIZE


def cal_minibatch_ratio(ratio_list):
    """Given the ratio_list, we want to make the RATIO same for each minibatch on each GPU.
    Note: this only work for 1) cfg.TRAIN.MAX_SIZE is ignored during `prep_im_for_blob` 
    and 2) cfg.TRAIN.SCALES containing SINGLE scale.
    Since all prepared images will have same min side length of cfg.TRAIN.SCALES[0], we can
     pad and batch images base on that.
    """
    DATA_SIZE = len(ratio_list)
    ratio_list_minibatch = np.empty((DATA_SIZE,))
    num_minibatch = int(np.ceil(DATA_SIZE / cfg.TRAIN.IMS_PER_BATCH))  # Include leftovers
    for i in range(num_minibatch):
        left_idx = i * cfg.TRAIN.IMS_PER_BATCH
        right_idx = min((i + 1) * cfg.TRAIN.IMS_PER_BATCH - 1, DATA_SIZE - 1)

        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1

        ratio_list_minibatch[left_idx:(right_idx + 1)] = target_ratio
    return ratio_list_minibatch


class MinibatchSampler(torch_sampler.Sampler):
    def __init__(self, ratio_list, ratio_index):
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.num_data = len(ratio_list)

        if cfg.TRAIN.ASPECT_GROUPING:
            # Given the ratio_list, we want to make the ratio same
            # for each minibatch on each GPU.
            self.ratio_list_minibatch = cal_minibatch_ratio(ratio_list)

    def __iter__(self):
        if cfg.TRAIN.ASPECT_GROUPING:
            # indices for aspect grouping awared permutation
            n, rem = divmod(self.num_data, cfg.TRAIN.IMS_PER_BATCH)
            round_num_data = n * cfg.TRAIN.IMS_PER_BATCH
            indices = np.arange(round_num_data)
            npr.shuffle(indices.reshape(-1, cfg.TRAIN.IMS_PER_BATCH))  # inplace shuffle
            if rem != 0:
                indices = np.append(indices, np.arange(round_num_data, round_num_data + rem))
            ratio_index = self.ratio_index[indices]
            ratio_list_minibatch = self.ratio_list_minibatch[indices]
        else:
            rand_perm = npr.permutation(self.num_data)
            ratio_list = self.ratio_list[rand_perm]
            ratio_index = self.ratio_index[rand_perm]
            # re-calculate minibatch ratio list
            ratio_list_minibatch = cal_minibatch_ratio(ratio_list)

        return iter(zip(ratio_index.tolist(), ratio_list_minibatch.tolist()))

    def __len__(self):
        return self.num_data


class BatchSampler(torch_sampler.BatchSampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, torch_sampler.Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                        batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)  # Difference: batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def collate_minibatch(list_of_blobs):
    """Stack samples seperately and return a list of minibatches
    A batch contains NUM_GPUS minibatches and image size in different minibatch may be different.
    Hence, we need to stack smaples from each minibatch seperately.
    """
    #list_of_blobs = filter(lambda x: x is not None, list_of_blobs)
    #list_of_blobs = [x for x in list_of_blobs if x is not None]
    Batch = {key: [] for key in list_of_blobs[0]}
    # Because roidb consists of entries of variable length, it can't be batch into a tensor.
    # So we keep roidb in the type of "list of ndarray".
    list_of_roidb = [blobs.pop('roidb') for blobs in list_of_blobs]
    for i in range(0, len(list_of_blobs), cfg.TRAIN.IMS_PER_BATCH):
        mini_list = list_of_blobs[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        # Pad image data
        mini_list = pad_image_data(mini_list)
        minibatch = default_collate(mini_list)
        minibatch['roidb'] = list_of_roidb[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        for key in minibatch:
            Batch[key].append(minibatch[key])

    return Batch


def pad_image_data(list_of_blobs):
    max_shape = blob_utils.get_max_shape([blobs['data'].shape[1:] for blobs in list_of_blobs])
    output_list = []
    for blobs in list_of_blobs:
        data_padded = np.zeros((3, max_shape[0], max_shape[1]), dtype=np.float32)
        _, h, w = blobs['data'].shape
        data_padded[:, :h, :w] = blobs['data']
        blobs['data'] = data_padded
        output_list.append(blobs)
    return output_list
