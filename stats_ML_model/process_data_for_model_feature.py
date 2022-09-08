import numpy as np
import os
import time
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input', dest='input_file')
parser.add_argument('--size', type=int, dest='window_size', help='3, 5, 7, 9, 11')

args = parser.parse_args()

TIME_WINDOW_SIZE = 30 * 60 / 6

# window_size is an odd positive integer number
def process_data(data, window_size=3):
    
  hf_size = (window_size - 1) // 2
  n = len(data)
  num = 0
  X = np.zeros(shape=(n,window_size*4), dtype=np.int32)
  y = np.zeros(shape=(n,1), dtype=np.int32)

  start_time = time.time()
  
  for current_index in range(n):
    
    if current_index%1000000 == 0:
      print('process %d line in %f'%(current_index, time.time() - start_time))
      
    if len(data[current_index]) < 6:
      continue
  
    user_id = data[current_index][0]
    num += 1

    features = np.array([], dtype=np.int32)
    
    # early return for half_size == 0
    
    if hf_size <= 0:
      
      features = np.append(features, [int(data[current_index][j]) for j in range(1, 5)])
      
      X[num-1] = features
      y[num-1] = int(data[current_index][5])
      
      continue

    
    # find all the records within the time window in the left side
    
    current_timestamp = int(data[current_index][2])
    
    left_boundary_index = current_index - 1
    
    while left_boundary_index >= 0:
        
      if (data[left_boundary_index][0] == user_id) and (current_timestamp - int(data[left_boundary_index][2]) <= TIME_WINDOW_SIZE):
        left_boundary_index = left_boundary_index - 1
      else:
        break
    
    # left window = [left_boundary_index+1, current_index - 1]
    left_window_size = current_index - left_boundary_index - 1
    
    # select all in the left window
    if hf_size >= left_window_size:
      
      # fill with empty features
      for _ in range(hf_size - left_window_size):
        features = np.append(features, [int(-1) for j in range(1, 5)])
    
      # fill with all the features in the window
      for record_index in range(left_window_size):
        features = np.append(features, [int(data[record_index + left_boundary_index + 1][j]) for j in range(1, 5)])
      
    else:
      
      # need to select indices in the left window
      index_selected_array = [False] * left_window_size
      
      # anchors need to fit for the sample
      time_anchor_array = [current_timestamp - (TIME_WINDOW_SIZE/hf_size) * (i+1) for i in range(hf_size)]
      
      # indices sampled
      index_found_array = []
      
      start_fit_index = 0
      
      # find the fit to each anchor
      for anchor_index in range(hf_size):
          
        time_anchor = time_anchor_array[anchor_index]
          
        last_find_index = -1
          
        # find the first index with timestamp <= anchor
        while start_fit_index < left_window_size:
            
          if int(data[current_index - 1 - start_fit_index][2]) <= time_anchor and (not index_selected_array[start_fit_index]):
            break
          else:
            if not index_selected_array[start_fit_index]:
              last_find_index = start_fit_index
              
            start_fit_index = start_fit_index + 1
            
        # use the last not used one  
        if start_fit_index == left_window_size:
              
          index_found = left_window_size - 1
              
          while index_selected_array[index_found]:
            index_found = index_found - 1
            
          index_found_array.append(current_index - 1 - index_found)
          index_selected_array[index_found] = True
            
        else:
          
          # locate last find
          if last_find_index == -1:
              
            last_find_index = start_fit_index - 1
            
            while last_find_index >= 0 and index_selected_array[last_find_index]:
              last_find_index = last_find_index - 1  
              
          # use start_fit_index
          if last_find_index < 0:
            
            index_found_array.append(current_index - 1 - start_fit_index)
            index_selected_array[start_fit_index] = True
          
          # compare start_fit_index and last_find_index
          else:
            
            # use start_fit_index
            if abs(time_anchor - int(data[current_index - 1 - start_fit_index][2])) <= abs(time_anchor - int(data[current_index - 1 - last_find_index][2])):
              index_found_array.append(current_index - 1 - start_fit_index)
              index_selected_array[start_fit_index] = True
            # use last_find_index
            else:
              index_found_array.append(current_index - 1 - last_find_index)
              index_selected_array[last_find_index] = True
          
      index_found_array.sort()
      
      # fill with all the features in the window
      for record_index in index_found_array:
        features = np.append(features, [int(data[record_index][j]) for j in range(1, 5)])


    # fill with the features in the current_index
    features = np.append(features, [int(data[current_index][j]) for j in range(1, 5)])

    
    # find all the records within the time window in the right side

    right_boundary_index = current_index + 1
    
    while right_boundary_index < n:
        
      if (data[right_boundary_index][0] == user_id) and (int(data[right_boundary_index][2]) - current_timestamp <= TIME_WINDOW_SIZE):
        right_boundary_index = right_boundary_index + 1
      else:
        break
    
    # right window = [current_index + 1, right_boundary_index - 1]
    right_window_size = right_boundary_index - 1 - current_index
    
    # select all in the left window
    if hf_size >= right_window_size:
    
      # fill with all the features in the window
      for record_index in range(right_window_size):
        features = np.append(features, [int(data[record_index + current_index + 1][j]) for j in range(1, 5)])
      
      # fill with empty features
      for _ in range(hf_size - right_window_size):
        features = np.append(features, [int(-1) for j in range(1, 5)])
        
    else:
      
      # need to select indices in the right window
      index_selected_array = [False] * right_window_size
      
      # anchors need to fit for the sample
      time_anchor_array = [current_timestamp + (TIME_WINDOW_SIZE/hf_size) * (i+1) for i in range(hf_size)]
      
      # indices sampled
      index_found_array = []
      
      start_fit_index = 0
      
      # find the fit to each anchor
      for anchor_index in range(hf_size):
          
        time_anchor = time_anchor_array[anchor_index]
          
        last_find_index = -1
          
        # find the first index with timestamp <= anchor
        while start_fit_index < right_window_size:
            
          if int(data[current_index + 1 + start_fit_index][2]) >= time_anchor and (not index_selected_array[start_fit_index]):
            break
          else:
            if not index_selected_array[start_fit_index]:
              last_find_index = start_fit_index
              
            start_fit_index = start_fit_index + 1
            
        # use the last not used one  
        if start_fit_index == right_window_size:
              
          index_found = right_window_size - 1
              
          while index_selected_array[index_found]:
            index_found = index_found - 1
            
          index_found_array.append(current_index + 1 + index_found)
          index_selected_array[index_found] = True
            
        else:
          
          # locate last find
          if last_find_index == -1:
              
            last_find_index = start_fit_index - 1
            
            while last_find_index >= 0 and index_selected_array[last_find_index]:
              last_find_index = last_find_index - 1  
              
          # use start_fit_index
          if last_find_index < 0:
            
            index_found_array.append(current_index + 1 + start_fit_index)
            index_selected_array[start_fit_index] = True
          
          # compare start_fit_index and last_find_index
          else:
            
            # use start_fit_index
            if abs(time_anchor - int(data[current_index + 1 + start_fit_index][2])) <= abs(time_anchor - int(data[current_index + 1 + last_find_index][2])):
              index_found_array.append(current_index + 1 + start_fit_index)
              index_selected_array[start_fit_index] = True
            # use last_find_index
            else:
              index_found_array.append(current_index + 1 + last_find_index)
              index_selected_array[last_find_index] = True
          
      index_found_array.sort()
      
      # fill with all the features in the window
      for record_index in index_found_array:
        features = np.append(features, [int(data[record_index][j]) for j in range(1, 5)])
        
    
    X[num-1] = features
    y[num-1] = int(data[current_index][5])

  return X[:num], y[:num]

if __name__ == '__main__':
    
  if os.path.isfile(args.input_file + '_model1_size_' + str(args.window_size)):
    print ('feature file exists...')
  else:
    raw_data = [line.rstrip('\n').split(',') for line in open(args.input_file)]
    X_train, y_train = process_data(raw_data, args.window_size)
    np.savetxt(args.input_file + '_model1_size_' + str(args.window_size),
             np.concatenate((X_train, y_train), axis=1), delimiter=',', fmt='%d')
