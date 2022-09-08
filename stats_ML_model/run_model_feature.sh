echo "run model feature ..."
echo "train data: ${train_file}"
echo "test data: ${test_file}"
for window_size in 9
do
  echo "window_size: ${window_size}"
  echo "preprocess data..."
  python3 process_data_for_model_feature.py --input $1 --size ${window_size}
  python3 process_data_for_model_feature.py --input $2 --size ${window_size}
  echo "classification..."
  for algo in "lr"
  do
    python3 model_feature.py --train "$1_model1_size_${window_size}" --test "$2_model1_size_${window_size}" --algo ${algo} --resample_rate $3
  done
done
