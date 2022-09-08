echo "run model smot ..."
echo "train data $1"
echo "test data $2"
python smot.py --train $1 --test $2 --resample_rate $3
