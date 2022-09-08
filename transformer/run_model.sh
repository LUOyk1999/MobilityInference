echo "run model transformer ..."
echo "train data $1"
echo "test data $2"
/usr/bin/python3.6 test.py --train $1 --eval $2 --logdir $3

