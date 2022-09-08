import json
import math
import os
import pickle
from bisect import bisect
import argparse
import nni
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils import shuffle
from torch import nn
from tqdm import tqdm, trange

from stlstm import STLSTM

def convert_longitude(data,result):
    if result=='1':
	    return int((data - 115.422) / 0.1)
    if result=='2':
        return int((data - 116.43) / 0.1)
    if result=='3':
        return int((data - 117.300) / 0.1)

def convert_latitude(data,result):
    if result=='1':
	    return int((data - 39.445) / 0.1)
    if result=='2':
        return int((data - 38.34) / 0.1)
    if result=='3':
        return int((data - 38.55) / 0.1)

def get_performance(output, vocab_size, targets, mask):
    logits = np.reshape(output, [-1, vocab_size])
    labels = np.reshape(targets, [-1])
    mask_ = (np.reshape(mask, [-1])).astype(np.float32)
    # mask_ = np.cast(mask_, np.float32)
    # mask_ = mask_.astype(np.float32)
    predictions = (np.argmax(logits, 1)).astype(np.int32)
    stayonly_labels = 2 * labels
    travel_tensor = np.ones_like(np.reshape(mask, [-1]))
    stay_tensor = np.zeros_like(np.reshape(mask, [-1]))

    travel_labels = np.equal(labels, travel_tensor)
    stay_labels = np.equal(labels, stay_tensor)
    correct_prediction = np.equal(predictions, labels)
    correct_negative_prediction = np.equal(predictions, stayonly_labels)

    positive = np.sum(travel_labels.astype(np.float32) * mask_)
    negative = np.sum(stay_labels.astype(np.float32) * mask_)
    accuracy = np.sum(correct_prediction.astype(np.float32) * mask_)
    negative_accuracy = np.sum(correct_negative_prediction.astype(np.float32) * mask_)
    TP = accuracy - negative_accuracy
    TN = negative_accuracy
    FN = positive - TP
    FP = negative - TN
    return TP, TN, FN, FP


def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]

def cal_slot_distance(value, slots):
    """
    Calculate a value's distance with nearest lower bound and higher bound in slots.

    :param value: The value to be calculated.
    :param slots: values of slots, needed to be sorted.
    :return: normalized distance with lower bound and higher bound,
        and index of lower bound and higher bound.
    """
    higher_bound = bisect(slots, value)
    lower_bound = higher_bound - 1
    if higher_bound == len(slots):
        return 1., 0., lower_bound, lower_bound
    else:
        lower_value = slots[lower_bound]
        higher_value = slots[higher_bound]
        total_distance = higher_value - lower_value
        return (value - lower_value) / total_distance, \
               (higher_value - value) / total_distance, \
               lower_bound, higher_bound


def cal_slot_distance_batch(batch_value, slots):
    """
    Proceed `cal_slot_distance` on a batch of data.

    :param batch_value: a batch of value, size (batch_size, step)
    :param slots: values of slots, needed to be sorted.
    :return: batch of distances and indexes. All with shape (batch_size, step).
    """
    # Lower bound distance, higher bound distance, lower bound, higher bound.
    ld, hd, l, h = [], [], [], []
    for batch in batch_value:
        ld_row, hd_row, l_row, h_row = [], [], [], []
        for step in batch:
            ld_one, hd_one, l_one, h_one = cal_slot_distance(step, slots)
            ld_row.append(ld_one)
            hd_row.append(hd_one)
            l_row.append(l_one)
            h_row.append(h_one)
        ld.append(ld_row)
        hd.append(hd_row)
        l.append(l_row)
        h.append(h_row)
    return np.array(ld), np.array(hd), np.array(l), np.array(h)


def construct_slots(min_value, max_value, num_slots, type):
    """
    Construct values of slots given min value and max value.

    :param min_value: minimum value.
    :param max_value: maximum value.
    :param num_slots: number of slots to construct.
    :param type: type of slots to construct, 'linear' or 'exp'.
    :return: values of slots.
    """
    if type == 'exp':
        n = (max_value - min_value) / (math.exp(num_slots - 1) - 1)
        return [n * (math.exp(x) - 1) + min_value for x in range(num_slots)]
    elif type == 'linear':
        n = (max_value - min_value) / (num_slots - 1)
        return [n * x + min_value for x in range(num_slots)]


class Dataset:
    """
    Dataset class for training ST-LSTM classifier.
    """
    def __init__(self, train_data, val_data, history_count,
                 poi_count):
        """
        :param train_data: pandas DataFrame containing the training dataset.
        :param val_data: pandas DataFrame containing the validation dataset.
        :param history_count: length of historical sequence in every training set.
        :param poi_count: total count of POIs.
        """

        self.min_t, self.max_t, self.min_d, self.max_d = 1e8, 0., 1e8, 0.
        self.history_count = history_count
        self.train_pair = self.construct_sequence(train_data)
        self.val_pair = self.construct_sequence(val_data)
        self.train_size = len(self.train_pair)
        self.val_size = len(self.val_pair)

        _, _, _, self.val_label,_ = zip(*self.val_pair)

    def construct_sequence(self, data):
        """
        Construct history sequence and label pairs for training.

        :param data: pandas DataFrame containing the dataset.
        :return: pairs of history sequence and label.
        """
        # Preprocess dataset, calculate time delta and distances
        # between sequential visiting records.
        data_ = pd.DataFrame(data, copy=True)
        data_.index -= 1
        data_.columns = [f'{c}_' for c in data.columns]
        data = pd.concat([data, data_], axis=1).iloc[1:-1]
        data['delta_t'] = (data['time_'] - data['time'])
        data['delta_d'] = (((data['latitude'] - data['latitude_'])/0.001).pow(2) +
                           ((data['longitude'] - data['longitude_'])/0.001).pow(2)).pow(0.5)
        data['user_id_'] = data['user_id_'].astype(int)
        data['user_id'] = data['user_id'].astype(int)
        data = data[data['user_id'] == data['user_id_']]

        # Update the min and max value of time delta and distance.
        self.min_t = min(self.min_t, data['delta_t'].min())
        self.max_t = max(self.max_t, data['delta_t'].max())
        self.min_d = min(self.min_d, data['delta_d'].min())
        self.max_d = max(self.max_d, data['delta_d'].max())

        # Construct history and label pairs.
        pairs = []
        for user_id, group in tqdm(data.groupby('user_id'),
                                   total=data['user_id'].drop_duplicates().shape[0],
                                   desc='Construct sequences'):
            for i in range(group.shape[0]//self.history_count):
                if((i+1)*self.history_count<=group.shape[0]):
                    his_rows = group.iloc[i*self.history_count:(i+1)*self.history_count]
                else:
                    his_rows = group.iloc[i*self.history_count:group.shape[0]]
                history_location = his_rows['poi_id_'].tolist()
                history_t = his_rows['delta_t'].tolist()
                history_d = his_rows['delta_d'].tolist()
                label_location = his_rows['label'].tolist()
                mask = his_rows['mask'].tolist()
                pairs.append((history_location, history_t, history_d, label_location, mask))
        return pairs

    def train_iter(self, batch_size):
        return next_batch(shuffle(self.train_pair), batch_size)

    def val_iter(self, batch_size):
        return next_batch(self.val_pair, batch_size)


class STLSTMClassifier(nn.Module):
    """
    RNN classifier using ST-LSTM as its core.
    """
    def __init__(self, input_size, output_size, hidden_size,
                 temporal_slots, spatial_slots,
                 device, learning_rate):
        """
        :param input_size: The number of expected features in the input vectors.
        :param output_size: The number of classes in the classifier outputs.
        :param hidden_size: The number of features in the hidden state.
        :param temporal_slots: values of temporal slots.
        :param spatial_slots: values of spatial slots.
        :param device: The name of the device used for training.
        :param learning_rate: Learning rate of training.
        """
        super(STLSTMClassifier, self).__init__()
        self.temporal_slots = sorted(temporal_slots)
        self.spatial_slots = sorted(spatial_slots)

        # Initialization of network parameters.
        self.st_lstm = STLSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

        # Embedding matrix for every temporal and spatial slots.
        self.embed_s = nn.Embedding(len(temporal_slots), input_size)
        self.embed = nn.Embedding(400, input_size)
        self.embed.weight.data.normal_(0, 0.1)
        self.embed_s.weight.data.normal_(0, 0.1)
        self.embed_q = nn.Embedding(len(spatial_slots), input_size)
        self.embed_q.weight.data.normal_(0, 0.1)

        # Initialization of network components.
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_func = nn.CrossEntropyLoss()

        self.device = torch.device(device)
        self.to(self.device)

    def place_parameters(self, ld, hd, l, h):
        ld = torch.from_numpy(np.array(ld)).type(torch.FloatTensor).to(self.device)
        hd = torch.from_numpy(np.array(hd)).type(torch.FloatTensor).to(self.device)
        l = torch.from_numpy(np.array(l)).type(torch.LongTensor).to(self.device)
        h = torch.from_numpy(np.array(h)).type(torch.LongTensor).to(self.device)

        return ld, hd, l, h

    def cal_inter(self, ld, hd, l, h, embed):
        """
        Calculate a linear interpolation.

        :param ld: Distances to lower bound, shape (batch_size, step)
        :param hd: Distances to higher bound, shape (batch_size, step)
        :param l: Lower bound indexes, shape (batch_size, step)
        :param h: Higher bound indexes, shape (batch_size, step)
        """
        # Fetch the embed of higher and lower bound.
        # Each result shape (batch_size, step, input_size)
        l_embed = embed(l)
        h_embed = embed(h)
        return torch.stack([hd], -1) * l_embed + torch.stack([ld], -1) * h_embed

    def forward(self, batch_l, batch_t, batch_d):
        """
        Process forward propagation of ST-LSTM classifier.

        :param batch_l: batch of input location sequences,
            size (batch_size, time_step, input_size)
        :param batch_t: batch of temporal interval value, size (batch_size, step)
        :param batch_d: batch of spatial distance value, size (batch_size, step)
        :return: prediction result of this batch, size (batch_size, output_size, step).
        """
        batch_l = torch.from_numpy(np.array(batch_l)).type(torch.LongTensor).to(self.device)
        batch_l = self.embed(batch_l)
        t_ld, t_hd, t_l, t_h = self.place_parameters(*cal_slot_distance_batch(batch_t, self.temporal_slots))
        d_ld, d_hd, d_l, d_h = self.place_parameters(*cal_slot_distance_batch(batch_d, self.spatial_slots))

        batch_s = self.cal_inter(t_ld, t_hd, t_l, t_h, self.embed_s)
        batch_q = self.cal_inter(d_ld, d_hd, d_l, d_h, self.embed_q)

        hidden_out, cell_out = self.st_lstm(batch_l, batch_s, batch_q)
        linear_out = self.linear(hidden_out[:,:,:])
        return linear_out

    def predict(self, batch_l, batch_t, batch_d):
        """
        Predict a batch of data.

        :param batch_l: batch of input location sequences,
            size (batch_size, time_step, input_size)
        :param batch_t: batch of temporal interval value, size (batch_size, step)
        :param batch_d: batch of spatial distance value, size (batch_size, step)
        :return: batch of predicted class indices, size (batch_size).
        """
        output = self.forward(batch_l, batch_t, batch_d)
        output = torch.reshape(output,[-1,2])
        return output.detach().cpu().numpy()


def batch_train(model: STLSTMClassifier, batch_l, batch_t, batch_d, batch_label):
    """
    Train model using one batch of data and return loss value.

    :param model: One instance of STLSTMClassifier.
    :param batch_l: batch of input location sequences,
        size (batch_size, time_step, input_size)
    :param batch_t: batch of temporal interval value, size (batch_size, step)
    :param batch_d: batch of spatial distance value, size (batch_size, step)
    :param batch_label: batch of label, size (batch_size)
    :return: loss value.
    """
    prediction = model(batch_l, batch_t, batch_d)
    prediction = torch.reshape(prediction,[-1,2])
    batch_label = torch.from_numpy(np.array(batch_label)).type(torch.LongTensor).to(model.device)
    batch_label = torch.reshape(batch_label,[-1])
    model.optimizer.zero_grad()
    loss = model.loss_func(prediction, batch_label)
    loss.backward()
    model.optimizer.step()

    return loss.detach().cpu().numpy(),prediction.detach().cpu().numpy()


def predict(model: STLSTMClassifier, batch_l, batch_t, batch_d):
    """
    Predict a batch of data using ST-LSTM classifier.

    :param model: One instance of STLSTMClassifier.
    :param batch_l: batch of input location sequences,
        size (batch_size, time_step, input_size)
    :param batch_t: batch of temporal interval value, size (batch_size, step)
    :param batch_d: batch of spatial distance value, size (batch_size, step)
    :return: batch of predicted class indices, size (batch_size).
    """
    prediction = model(batch_l, batch_t, batch_d)
    prediction = torch.reshape(prediction,[-1,2])
    return prediction.detach().cpu().numpy()



def get_dataset(dataset_t, dataset_v, history_count, test, result):
    """
    Get certain dataset for training.
    Will read from cache file if cache exists, or build from scratch if not.

    :param dataset_name: name of dataset.
    :param history_count: length of historical sequence in training set.
    :param test: Use test set or validation set to build this dataset.
    :param force_build: Ignore the existence of cache file and re-build dataset.
    :return: a instance of Dataset class.
    """
    #iterate the records
    train_content=open(dataset_t, 'r')
    train_records=[line.rstrip('\n') for line in train_content]
    c_uid=-1
    time_list=[]
    user_list=[]
    log_list=[]
    lat_list=[]
    target_list=[]
    mask_list=[]
    poi=[]
    counter = 0
    for record_index in range(len(train_records)-1):
        columns=train_records[record_index].split(",")
        user_list.append(int(columns[0]))
        time_list.append(int(columns[1]))
        lat_list.append(float(columns[2]))
        log_list.append(float(columns[3]))
        poi.append(convert_latitude(float(columns[2]),result)*convert_longitude(float(columns[3]),result))
        if(len(columns) >= 5):
            target_list.append(int(columns[4]))
            mask_list.append(1)
        else:
            target_list.append(0)
            mask_list.append(0)
    
    train_df={"user_id" : user_list,"time" : time_list,"latitude":lat_list,"longitude":log_list,"label":target_list,"poi_id":poi,"mask":mask_list}
    train_df=pd.DataFrame(train_df)
    print("train",len(target_list))

    train_content=open(dataset_v, 'r')
    train_records=[line.rstrip('\n') for line in train_content]
    c_uid=-1
    time_list=[]
    user_list=[]
    log_list=[]
    lat_list=[]
    target_list=[]
    mask_list=[]
    poi=[]
    counter = 0
    for record_index in range(len(train_records)-1):
        columns=train_records[record_index].split(",")
        user_list.append(int(columns[0]))
        time_list.append(int(columns[1]))
        lat_list.append(float(columns[2]))
        log_list.append(float(columns[3]))
        poi.append(convert_latitude(float(columns[2]),result)*convert_longitude(float(columns[3]),result))
        if(len(columns) >= 5):
            target_list.append(int(columns[4]))
            mask_list.append(1)
        else:
            target_list.append(0)
            mask_list.append(0)
    val_df={"user_id" : user_list,"time" : time_list,"latitude":lat_list,"longitude":log_list,"label":target_list,"poi_id":poi,"mask":mask_list}
    val_df=pd.DataFrame(val_df)
    print("valid",len(target_list))

    dataset = Dataset(train_df, val_df, history_count=history_count,poi_count=200*200)

    return dataset

def print_to_file(filename, string_info, mode="a"):
	with open(filename, mode) as f:
		f.write(str(string_info) + "\n")

def train(dataset, display_batch, hidden_size,
          device, training_epochs, batch_size, learning_rate,
          num_temporal_slots, num_spatial_slots,
          temporal_slot_type, spatial_slot_type,dataset_t,dataset_v,result):
    
    temporal_slots = construct_slots(dataset.min_t, dataset.max_t,
                                     num_temporal_slots, temporal_slot_type)
    spatial_slots = construct_slots(dataset.min_d, dataset.max_d,
                                    num_spatial_slots, spatial_slot_type)

    model = STLSTMClassifier(input_size=100, output_size=2,
                             hidden_size=hidden_size, device=device,
                             learning_rate=learning_rate,
                             temporal_slots=temporal_slots, spatial_slots=spatial_slots)

    acc_list = []
    trained_batches = 0
    with trange(training_epochs * math.ceil(dataset.train_size / batch_size), desc='Training') as bar:
        for epoch in range(training_epochs):
            TPs = 0.0
            TNs = 0.0
            FNs = 0.0
            FPs = 0.0
            for train_batch in dataset.train_iter(batch_size):
                batch_l, batch_t, batch_d, batch_label, mask = zip(*train_batch)
                _, _pred = batch_train(model, batch_l, batch_t, batch_d, batch_label)
                TP, TN, FN, FP = get_performance(_pred, 2, batch_label, mask)
                TPs += TP
                TNs += TN
                FNs += FN
                FPs += FP
                VPs = TPs / (TPs + FPs)  # Travel Precision
                VRs = TPs / (TPs + FNs)  # Travel Recall
                SPs = TNs / (TNs + FNs)  # Stay Precision
                SRs = TNs / (TNs + FPs)  # Stay Recall
                TOL = TPs + TNs + FNs + FPs
                ACC = (TPs + TNs) / TOL
                F1_S = 2 * SPs * SRs / (SPs + SRs)
                F1_V = 2 * VPs * VRs / (VPs + VRs)
                F1 = 2 * F1_S * F1_V / (F1_V + F1_S)
                print("train : VP: %.3f VR: %.3f SP: %.3f SR: %.3f ACC: %.3f F1-S: %.3f F1-V: %.3f F1: %.3f" %
                            ( VPs, VRs, SPs, SRs, ACC, F1_S, F1_V, F1))
                trained_batches += 1
                bar.update(1)
            if (epoch+1) % 10 == 0 or epoch == 0:
                bar.set_description('Testing')
                pres = []
                TPs = 0.0
                TNs = 0.0
                FNs = 0.0
                FPs = 0.0
                for test_batch in dataset.val_iter(batch_size):
                    test_l, test_t, test_d, test_label, mask = zip(*test_batch)
                    pre_batch = predict(model, test_l, test_t, test_d)
                    TP, TN, FN, FP = get_performance(pre_batch, 2, test_label, mask)
                    TPs += TP
                    TNs += TN
                    FNs += FN
                    FPs += FP
                    VPs = TPs / (TPs + FPs)  # Travel Precision
                    VRs = TPs / (TPs + FNs)  # Travel Recall
                    SPs = TNs / (TNs + FNs)  # Stay Precision
                    SRs = TNs / (TNs + FPs)  # Stay Recall
                    TOL = TPs + TNs + FNs + FPs
                    ACC = (TPs + TNs) / TOL
                    F1_S = 2 * SPs * SRs / (SPs + SRs)
                    F1_V = 2 * VPs * VRs / (VPs + VRs)
                    F1 = 2 * F1_S * F1_V / (F1_V + F1_S)
                    print("test : VP: %.3f VR: %.3f SP: %.3f SR: %.3f ACC: %.3f F1-S: %.3f F1-V: %.3f F1: %.3f" %
                                ( VPs, VRs, SPs, SRs, ACC, F1_S, F1_V, F1))
                print_to_file("./"+result+"/record.txt", dataset_v+"Epoch: %d test: VP: %.3f VR: %.3f SP: %.3f SR: %.3f ACC: %.3f F1-S: %.3f F1-V: %.3f F1: %.3f " % (
                                epoch, VPs, VRs, SPs, SRs, ACC, F1_S, F1_V, F1))

    return model, np.array(acc_list)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_t', type=str, default='./1/train-10000-30-800-1.0')
    parser.add_argument('--dataset_v', type=str, default='./1/test-10000-30-800-1.0')
    parser.add_argument('--num_spatial_slots', type=int, default=1000)
    parser.add_argument('--num_temporal_slots', type=int, default=100)
    parser.add_argument('--spatial_slot_type', type=str, default='linear')
    parser.add_argument('--temporal_slot_type', type=str, default='linear')
    parser.add_argument('--result', type=str, default='1')
    args = parser.parse_args()
    dataset = get_dataset(dataset_t=args.dataset_t, dataset_v=args.dataset_v, history_count=100, test=True,result=args.result)
    
    _, acc_log = train(dataset=dataset,
                        display_batch=20000, hidden_size=200, device='cuda:0',
                        training_epochs=10, batch_size=20, learning_rate=1e-4,
                        num_temporal_slots=args.num_temporal_slots, num_spatial_slots=args.num_spatial_slots,
                        temporal_slot_type=args.temporal_slot_type, spatial_slot_type=args.spatial_slot_type, dataset_t=args.dataset_t,
                        dataset_v=args.dataset_v,result=args.result)