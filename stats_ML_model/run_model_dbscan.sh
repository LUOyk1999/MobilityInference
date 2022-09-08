echo "run model st-dbscan ..."
echo "train data $1"
echo "test data $2"
python3 st-dbscan.py --train $1 --test $2 --resample_rate $3
