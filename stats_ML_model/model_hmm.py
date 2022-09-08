import numpy as np
import time
import math
from argparse import ArgumentParser
import os


parser = ArgumentParser()
parser.add_argument('--train', dest='train_file')
parser.add_argument('--test', dest='test_file')
parser.add_argument('--resample_rate', dest='resample_rate')

args = parser.parse_args()

col_num = 6

def harmonic_mean(v1, v2):
  if v1 + v2 == 0:
    return 0
  else:
    return float(2 * v1 * v2)/(v1 + v2)

# Minkowski distance between location id_1 and id_2
def distance(lat1, lon1, lat2, lon2):
    """
    compute distance given two points
    """
    COS_LATITUDE = 0.77
    
    x = abs(lon2 - lon1) * COS_LATITUDE * 111
    y = abs(lat2 - lat1) * 111
    return math.sqrt(x * x + y * y)


class HMM(object):
  ''' Simple Hidden Markov Model implementation.  User provides
    transition, emission and initial probabilities in dictionaries
    mapping 2-character codes onto floating-point probabilities
    for those table entries.  States and emissions are represented
    with single characters.  Emission symbols comes from a finite.  '''

  def __init__(self, A, E, I):
    ''' Initialize the HMM given transition, emission and initial
      probability tables. '''

    self.A = A  # transition probability matrix
    self.E = E  # emission probability matrix
    self.I = I  # initial probabilities

    # Make log-base-2 versions for log-space functions
    self.Alog = np.log2(self.A)
    self.Elog = np.log2(self.E)
    self.Ilog = np.log2(self.I)

    self.Q = range(2)  # latent values set: {0, 1}

  def viterbiL(self, x):
    ''' Given sequence of emissions, return the most probable path
        along with log2 of its probability.  Just like viterbi(...)
        but in log2 domain. '''
    nrow, ncol = len(self.Q), len(x)
    mat = np.zeros(shape=(nrow, ncol), dtype=float)  # prob
    matTb = np.zeros(shape=(nrow, ncol), dtype=int)   # backtrace
    # Fill in first column
    for i in range(0, nrow):
      mat[i, 0] = self.Elog[i, x[0]] + self.Ilog[i]
    # Fill in rest of prob and Tb tables
    for j in range(1, ncol):
      for i in range(0, nrow):
        ep = self.Elog[i, x[j]]
        mx, mxi = mat[0, j - 1] + self.Alog[0, i] + ep, 0
        for i2 in range(1, nrow):
          pr = mat[i2, j - 1] + self.Alog[i2, i] + ep
          if pr > mx:
            mx, mxi = pr, i2
        mat[i, j], matTb[i, j] = mx, mxi
    # Find final state with maximal probability
    omx, omxi = mat[0, ncol - 1], 0
    for i in range(1, nrow):
      if mat[i, ncol - 1] > omx:
        omx, omxi = mat[i, ncol - 1], i
    # Backtrace
    i, p = omxi, [omxi]
    for j in range(ncol - 1, 0, -1):
      i = matTb[i, j]
      p.append(i)
    p = p[::-1]
    return omx, p  # Return probability and path


MAX_D_T = 30
MAX_D_L = 800

# Train: HMM supervised learning
def train(data):
  I = np.zeros(2)  # initial probabilities
  A = np.zeros((2, 2))  # transition probability matrix
  # emission probability matrix
  E = np.zeros((2, (MAX_D_T + 1) * (MAX_D_L + 1)))
  for trip in data:
    for i in range(len(trip)):
      if len(trip[i]) == col_num:
        if i == 0:
          I[int(trip[i][col_num-1])] += 1
        else:
          delta_t = int((int(trip[i][2]) - int(trip[i-1][2]))/10)
          delta_l = int(distance(int(trip[i][3]), int(trip[i][4]), int(trip[i-1][3]), int(trip[i-1][4])))
          if delta_t > MAX_D_T:
            delta_t = MAX_D_T
          if delta_l > MAX_D_L:
            delta_l = MAX_D_L

          # Observation statistics
          E[int(trip[i][col_num-1]), delta_t*(MAX_D_L+1)+delta_l] += 1
          # State transfer statistics
          if len(trip[i-1]) == col_num:
            A[int(trip[i-1][col_num-1]), int(trip[i][col_num-1])] += 1

  # Compute estimated probabilities with laplace smooth
  I = I + 1
  A = A + 1
  E = E + 1

  I = I / I.sum()
  for i in range(A.shape[0]):
    i_sum = A[i, :].sum()
    A[i, :] = A[i, :] / i_sum
  for i in range(E.shape[0]):
    i_sum = E[i, :].sum()
    E[i, :] = E[i, :] / i_sum

  return I, A, E

# Predict: HMM model
def predict(model, data):
  # Number of labels, travel (1) labels, stay (0) labels
  num = 0
  t_num = 0
  s_num = 0
  t_pred_num = 0
  s_pred_num = 0

  tp = 0
  tn = 0
  acc = 0

  for trip in data:
    labels = [-1 for _ in range(len(trip))]
    no_label = True
    for i in range(len(trip)):
      if len(trip[i]) == col_num:
        labels[i] = int(trip[i][col_num-1])
        no_label = False

    # If some trips labeled
    if not no_label:
      # Conctruct test data for hmm model
      obs = [0 for _ in range(len(trip))]  # observations
      for i in range(1, len(trip)):
        delta_t = int(trip[i][2]) - int(trip[i - 1][2])
        # delta_l = dist(int(trip[i][3]), int(trip[i - 1][3]))
        delta_l = int(distance(int(trip[i][3]), int(trip[i][4]), int(trip[i-1][3]), int(trip[i-1][4])))
        
        if delta_t > MAX_D_T:
          delta_t = MAX_D_T
        if delta_l > MAX_D_L:
          delta_l = MAX_D_L
        obs[i] = delta_t * (MAX_D_L + 1) + delta_l
        if obs[i] < 0:
          print(trip)
      omx, p = model.viterbiL(obs)
      #print(omx,p)

      # Counts
      for i in range(len(labels)):
        if labels[i] != -1:
          num += 1
          if labels[i] == 0:
            s_num += 1
          else:
            t_num += 1
          if p[i] == 0:
            s_pred_num += 1
          else:
            t_pred_num += 1
          if p[i] == labels[i]:
            acc += 1
            if p[i] == 0:
              tn += 1
            if p[i] == 1:
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
  
  return vp, vr, sp, sr, acc, t_pred_num, s_pred_num, t_num, s_num


def preprocess(data):
  start = 0
  end = 0
  fmt_data = []
  while end < len(data):
    is_end = False
    end += 1

    if end == len(data):
      is_end = True
    elif data[end][2] == '0':
      if data[end-1][2] == '0':
        if data[end][1] == data[start][1]:
          is_end = False
        else:
          is_end = True
      else:
        is_end = True

    if is_end:
      fmt_data.append(data[start:end])
      start = end

  return fmt_data


if __name__ == '__main__':
  
  file_dir = os.path.dirname(args.train_file) + '/'
  trainfile_name = os.path.basename(args.train_file)
  testfile_name = os.path.basename(args.test_file)
  
  print('Algorithm: hmm')
  
  start_time = time.time()
  train_data = [line.rstrip('\n').split(',') for line in open(args.train_file)]
  train_data = preprocess(train_data)

  I, A, E = train(train_data)
  hmm = HMM(A, E, I)
  
  test_data = [line.rstrip('\n').split(',') for line in open(args.test_file)]
  test_data = preprocess(test_data)
  
  vp, vr, sp, sr, acc, vn_p, sn_p, vn, sn = predict(hmm, test_data)
  
  v_f1 = harmonic_mean(vp, vr)
  s_f1 = harmonic_mean(sp, sr)
  acc_f1 = harmonic_mean(v_f1, s_f1)

  print('%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%d\n%d\n%d\n%d\n%f' %(float(args.resample_rate), vp, vr, sp, sr, acc, v_f1, s_f1, acc_f1, vn_p, sn_p, vn, sn, time.time() - start_time))
  
  result_filename = file_dir + "perf_hmm_" + testfile_name.replace('-'+format(float(args.resample_rate), '.1f'), '')  + '.csv'

  with open(result_filename, 'a') as ofile:
    ofile.write(','.join([str(args.resample_rate), str(vp), str(vr), str(sp), str(sr), str(acc), str(v_f1), str(s_f1), str(acc_f1), str(vn_p), str(sn_p), str(vn), str(sn), str(time.time() - start_time)]) + '\n')

