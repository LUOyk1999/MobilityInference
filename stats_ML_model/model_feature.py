import numpy as np
from sklearn import svm, linear_model, tree
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import time
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument('--train', dest='train_file')
parser.add_argument('--test', dest='test_file')
parser.add_argument('--algo', dest='algo')
parser.add_argument('--resample_rate', dest='resample_rate')

args = parser.parse_args()

def harmonic_mean(v1, v2):
  if v1 + v2 == 0:
    return 0
  else:
    return float(2 * v1 * v2)/(v1 + v2)

def model1_clf(algo='lr'):
    
  file_dir = os.path.dirname(args.train_file) + '/'
  trainfile_name = os.path.basename(args.train_file)
  testfile_name = os.path.basename(args.test_file)
  
  print('Algorithm: %s' % (algo))
  start_time = time.time()
  
  if algo == 'linear_svm':
    clf = svm.LinearSVC()
  if algo == 'svm':
    clf = svm.SVC(cache_size=10000)
  if algo == 'lr':
    clf = linear_model.LogisticRegression()
  if algo == 'tree':
    clf = tree.DecisionTreeClassifier()
  if algo == 'GuassianNB':
    clf = GaussianNB()
  if algo == 'MultinomialNB':
    clf = MultinomialNB()
  
  vp = 0 # travel precision: TP / (TP + FP)
  vr = 0 # travel recall: TP / (TP + FN)
  sp = 0 # stay precision: TN / (TN + FN)
  sr = 0 # stay recall: TN / (TN + FP)
  acc = 0
  
  train_data = np.loadtxt(args.train_file, delimiter=',')
  X_train = train_data[:,:-1]
  y_train = train_data[:,-1]
  clf.fit(X_train, y_train)
  test_data = np.loadtxt(args.test_file, delimiter=',')
  X_test = test_data[:,:-1]
  y_test = test_data[:,-1]
  y_pred = clf.predict(X_test)
  
  if np.sum(y_pred==1) == 0:
    vp = 0
  else:
    vp = float(np.sum((y_pred==y_test)*(y_test==1))) / np.sum(y_pred==1)
  
  vr = float(np.sum((y_pred==y_test)*(y_test==1))) / np.sum(y_test==1)
  
  if np.sum(y_pred==0) == 0:
    sp = 0
  else:
    sp = float(np.sum((y_pred==y_test)*(y_test==0))) / np.sum(y_pred==0)
    
  sr = float(np.sum((y_pred==y_test)*(y_test==0))) / np.sum(y_test==0)
  acc = float(np.sum(y_pred==y_test)) / len(y_test)
  
  v_f1 = harmonic_mean(vp, vr)
  s_f1 = harmonic_mean(sp, sr)
  acc_f1 = harmonic_mean(v_f1, s_f1)
  
  vn_p = np.sum(y_pred==1)
  sn_p = np.sum(y_pred==0)
  vn = np.sum(y_test==1)
  sn = np.sum(y_test==0)

  print('%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%f\n%d\n%d\n%d\n%d\n%f' %(float(args.resample_rate), vp, vr, sp, sr, acc, v_f1, s_f1, acc_f1, vn_p, sn_p, vn, sn, time.time() - start_time))
  
  result_filename = file_dir + "perf_" + args.algo +"_" + testfile_name.replace('-'+format(float(args.resample_rate), '.1f'), '') + '.csv'

  with open(result_filename, 'a') as ofile:
    ofile.write(','.join([str(args.resample_rate), str(vp), str(vr), str(sp), str(sr), str(acc), str(v_f1), str(s_f1), str(acc_f1), str(vn_p), str(sn_p), str(vn), str(sn), str(time.time() - start_time)]) + '\n')

if __name__ == '__main__':
  model1_clf(algo=args.algo)
