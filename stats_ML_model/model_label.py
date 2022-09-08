import numpy as np
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

print('Algorithm: label')

start_time = time.time()

train_data = [line.rstrip('\n').split(',') for line in open(args.train_file)]
train_data = [d for d in train_data if len(d) == 6]
test_data = [line.rstrip('\n').split(',') for line in open(args.test_file)]
record_num = len(test_data)
test_data = [d for d in test_data if len(d) == 6]

def harmonic_mean(v1, v2):
  if v1 + v2 == 0:
    return 0
  else:
    return float(2 * v1 * v2)/(v1 + v2)

def convert_to_grid_id(lat_grid, lng_grid):
  return int(lat_grid * 2100) + int(lng_grid)

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


# Train: statistics
d = {}
for i in range(len(train_data)):
  hour = int(train_data[i][1]) % 24
  location = convert_to_grid_id(int(train_data[i][3]), int(train_data[i][4]))
  label = int(train_data[i][5])
  if (hour, location) in d:
    d[hour, location][label] += 1
  else:
    d[hour, location] = [0, 0, 0]
    d[hour, location][label] += 1

for k in d.keys():
  stay = d[k[0], k[1]][0]
  travel = d[k[0], k[1]][1]
  if travel >= stay:
    d[k[0], k[1]][2] = 1
  else:
    d[k[0], k[1]][2] = 0

# Predict and compute TPR, TNR, ACC
for i in range(len(test_data)):
  hour = int(test_data[i][1]) % 24
  location = convert_to_grid_id(int(test_data[i][3]), int(test_data[i][4]))
  label = int(test_data[i][5])
  if (hour, location) not in d:
    # print(hour, location)
    predict = np.random.choice([0, 1])
  else:
    predict = d[hour, location][2]
  
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

if t_pred_num ==0:
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

print('%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%d\n%d\n%d\n%d\n%f' %(float(args.resample_rate), vp, vr, sp, sr, acc, v_f1, s_f1, acc_f1, vn_p, sn_p, vn, sn, time.time() - start_time))

result_filename = file_dir + "perf_label_" + testfile_name.replace('-'+format(float(args.resample_rate), '.1f'), '') + '.csv'

with open(result_filename, 'a') as ofile:
  ofile.write(','.join([str(args.resample_rate), str(vp), str(vr), str(sp), str(sr), str(acc), str(v_f1), str(s_f1), str(acc_f1), str(vn_p), str(sn_p), str(vn), str(sn), str(record_num), str(float(vn)/record_num), str(float(sn)/record_num), str(time.time() - start_time)]) + '\n')

