# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import math
import time
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument('--train', dest='train_file')
parser.add_argument('--test', dest='test_file')
parser.add_argument('--resample_rate', dest='resample_rate')

args = parser.parse_args()

file_dir = os.path.dirname(args.train_file) + '/'
trainfile_name = os.path.basename(args.train_file)
testfile_name = os.path.basename(args.test_file)

print('Algorithm: ST-DBscan')

split = 0.001

def convert_longitude(data, split):
    return int((data - 115.422) / split)

def convert_latitude(data, split):
    return int((data - 39.445) / split)

def harmonic_mean(v1, v2):
    if v1 + v2 == 0:
        return 0
    else:
        return float(2 * v1 * v2) / (v1 + v2)


start_time = time.time()

test_data = [line.rstrip('\n').split(',') for line in open(args.test_file)]
train_data = test_data
print(len(train_data))
for i in range(len(train_data)):
    if len(train_data[i]) == 4:
        train_data[i].append(0)
    train_data[i].append(int(i))
    train_data[i].append(int(0))

result = [777777] * len(train_data)
vp, vr, sp, sr, acc = 0, 0, 0, 0, 0
# Number of labels, travel (1) labels, stay (0) labels
num = 0
t_num = 0
s_num = 0
# Number of travel labels and stay labels (prediction)
t_pred_num = 0
s_pred_num = 0
# True positive, true negative
tp = 0
tn = 0


def distance(lat1, lon1, lat2, lon2):
    """
    compute distance given two points
    """
    COS_LATITUDE = 0.77
    x = abs(lon2 - lon1) * COS_LATITUDE * 111
    y = abs(lat2 - lat1) * 111
    return math.sqrt(x * x + y * y)


class STDBSCAN(object):

    def __init__(self, spatial_threshold=800.0, temporal_threshold=30.0 * 60.0,
                 min_neighbors=3):

        self.spatial_threshold = spatial_threshold
        self.temporal_threshold = temporal_threshold
        self.min_neighbors = min_neighbors

    def _retrieve_neighbors(self, index_center, matrix):
        copy = matrix[index_center][5]
        matrix = np.array(matrix)
        matrix = matrix.astype(np.int64)
        center_point = matrix[index_center, :]
        start = index_center - 1
        end = index_center + 1
        while (start >= 0):
            if (center_point[1] - matrix[start][1] <= self.temporal_threshold):
                start -= 1
            else:
                break
        while (end < len(matrix)):
            if (matrix[end][1] - center_point[1] <= self.temporal_threshold):
                end += 1
            else:
                break
        neigborhood = []
        while start+1 <= end-1:
            if (distance(int(matrix[start+1][2]), int(matrix[start+1][3]), int(center_point[2]),
                         int(center_point[3])) <= self.spatial_threshold):
                neigborhood.append(matrix[start+1][5])
            start += 1
        neigborhood.remove(copy)
        print(neigborhood)
        return neigborhood

    def fit_transform(self, seg):

        cluster_label = 0
        noise = -1
        unmarked = 777777
        stack = []
        init = seg[0][5]
        # for each point in database
        for index in range(len(seg)):
            if result[seg[index][5]] == unmarked:
                neighborhood = self._retrieve_neighbors(index, seg)
                if len(neighborhood) < self.min_neighbors:
                    result[seg[index][5]] = noise
                else:  # found a core point
                    cluster_label += 1
                    # assign a label to core point
                    result[seg[index][5]] = cluster_label
                    # assign core's label to its neighborhood
                    for neig_index in neighborhood:
                        result[neig_index] = cluster_label
                        stack.append(neig_index - init)  # append neighbors to stack

                    # find new neighbors from core point neighborhood
                    while len(stack) > 0:
                        current_point_index = stack.pop()
                        new_neighborhood = \
                            self._retrieve_neighbors(current_point_index,
                                                     seg)
                        # current_point is a new core
                        if len(new_neighborhood) >= self.min_neighbors:
                            for neig_index in new_neighborhood:
                                neig_cluster = result[neig_index]
                                if any([neig_cluster == noise,
                                        neig_cluster == unmarked]):
                                    result[neig_index] = cluster_label
                                    stack.append(neig_index - init)
        print(init)
        return


st_dbscan = STDBSCAN(spatial_threshold=800, temporal_threshold=30 * 60,
                     min_neighbors=3)

segment = [[]]
temp = 0
for i in range(len(train_data)):
    train_data[i][1] = int(train_data[i][1])
    train_data[i][2] = int(convert_latitude(float(train_data[i][2]),split))
    train_data[i][3] = int(convert_longitude(float(train_data[i][3]),split))
    if (i == 0):
        segment[temp].append(train_data[i])
        continue
    if train_data[i][0] != train_data[i - 1][0]:
        temp = temp + 1
        segment.append([])
    segment[temp].append(train_data[i])

for seg in segment:
    st_dbscan.fit_transform(seg)

for i in range(len(result)):
    if (len(test_data[i]) == 4):
        continue

    label = int(test_data[i][4])
    if result[i] == -1:
        predict = 1
    else:
        predict = 0
    num += 1
    if label == 0:
        s_num += 1
    else:
        t_num += 1
    if predict == 0:
        s_pred_num += 1
    else:
        t_pred_num += 1
    if predict == label:
        acc += 1
        if label == 0:
            tn += 1
        if label == 1:
            tp += 1

if t_pred_num == 0:
    vp = 0
else:
    vp = float(tp) / t_pred_num

vr = float(tp) / t_num

if s_pred_num == 0:
    sp = 0
else:
    sp = float(tn) / s_pred_num

sr = float(tn) / s_num
acc = float(acc) / num

v_f1 = harmonic_mean(vp, vr)
s_f1 = harmonic_mean(sp, sr)
acc_f1 = harmonic_mean(v_f1, s_f1)

vn_p = t_pred_num
sn_p = s_pred_num
vn = t_num
sn = s_num

print('resample_rate:%f\nvp:%f\nvr:%f\nsp:%f\nsr:%f\nacc:%f\nv_f1:%f\ns_f1:%f\nacc_f1:%f\nvn_p:%d\nsn_p:%d\nvn:%d\nsn:%d\ntime:%f' % (
    float(args.resample_rate), vp, vr, sp, sr, acc, v_f1, s_f1, acc_f1, vn_p, sn_p, vn, sn, time.time() - start_time))

result_filename = file_dir + "perf_st-dbscan_" + testfile_name.replace('-' + format(float(args.resample_rate), '.1f'),
                                                                    '') + '.csv'

with open(result_filename, 'a+') as ofile:
    ofile.write(','.join(
        [str(args.resample_rate), str(vp), str(vr), str(sp), str(sr), str(acc), str(v_f1), str(s_f1), str(acc_f1),
         str(vn_p), str(sn_p), str(vn), str(sn), str(time.time() - start_time)]) + '\n')
