echo "run model voting ..."
echo "train data $1"
echo "test data $2"
python3 model_voting.py --train $1 --test $2 --resample_rate $3
