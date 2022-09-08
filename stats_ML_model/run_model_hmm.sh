echo "run model hmm ..."
echo "train data $1"
echo "test data $2"
python3 model_hmm.py --train $1 --test $2 --resample_rate $3
