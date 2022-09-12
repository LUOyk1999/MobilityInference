# Mobility Inference on Long-Tailed Sparse Trajectory

The repository implements the SDS algorithm and Singleton transformer in tensorflow described in the following paper

> *Lei Shi, Yuankai Luo, Shuai Ma, Hanghang Tong, Zhetao Li, Xiatian Zhang, and Zhiguang Shan. 2022. Mobility Inference on Long-Tailed Sparse Trajectory. ACM Trans. Intell. Syst. Technol. Just Accepted (September 2022). https://doi.org/10.1145/3563457*

## SDS Algorithm

![](https://raw.githubusercontent.com/LUOyk1999/images/main/images/image-20220909205642330.png)

## A figure description about Singleton transformer for mobility Inference

![](https://raw.githubusercontent.com/LUOyk1999/images/main/images/image-20220909205836797.png)

## Installation

The dependencies are managed:

```
python=3.6
tensorflow==1.12.0
numpy>=1.15.4
sentencepiece==0.1.8
tqdm>=4.28.1
```

Train the model:

```shell
cd transformer
cd eval
unzip test_data.zip
unzip train_data.zip
cd ..
./run_model.sh "eval/train_data" "eval/test_data" "log"
```

