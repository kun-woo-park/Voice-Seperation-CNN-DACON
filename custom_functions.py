import torch
import time
import pandas as pd
import numpy as np
import torchaudio.transforms as AT
import torch.nn as nn
from glob import glob
from tqdm import tqdm
from scipy.io import wavfile
from torch.utils.data import Dataset


def data_loader(files):
    out = []
    for file in tqdm(files):
        fs, data = wavfile.read(file)
        out.append(data)
    out = np.array(out, dtype=np.float32)
    return out


def Mel_spectroize(train_feature_path, train_label_path):
    x_data = sorted(glob(train_feature_path))
    x_data = data_loader(x_data)
    y_data = pd.read_csv(train_label_path, index_col=0)
    y_data = y_data.values

    mel_spectrogram = nn.Sequential(
        AT.MelSpectrogram(sample_rate=16000,
                          n_fft=512,
                          win_length=400,
                          hop_length=160,
                          n_mels=80),
        AT.AmplitudeToDB()
    )

    mel0 = mel_spectrogram(torch.tensor(x_data[0])).view(1, 1, 80, 101)
    mel1 = mel_spectrogram(torch.tensor(x_data[1])).view(1, 1, 80, 101)
    mel = torch.cat((mel0, mel1), 0)
    for i in range(2, 100000):
        if i % 100 == 0:
            print("Mel spectrogram progress: {}%".format(i/100000*100))
        mel_temp = mel_spectrogram(torch.tensor(x_data[i])).view(1, 1, 80, 101)
        mel = torch.cat((mel, mel_temp), 0)
    return mel, y_data


class CustomDataset(Dataset):
    def __init__(self, x_dat, y_dat):
        x = x_dat
        y = y_dat
        self.len = x.shape[0]
        y = y.astype('float32')
        x = x.astype('float32')
        self.x_data = torch.tensor(x)
        self.y_data = torch.tensor(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class VRModel(torch.nn.Module):
    def __init__(self, batch_size, num_gpus):
        super(VRModel, self).__init__()
        self.batch_size = batch_size
        self.num_gpus = num_gpus

        self.layer_1 = nn.Conv2d(1, 8, (4, 5), 2)
        self.act_1 = nn.ReLU()

        self.layer_2 = nn.Conv2d(8, 16, (4, 5), 2)
        self.act_2 = nn.ReLU()

        self.layer_3 = nn.Conv2d(16, 32, (4, 5))
        self.act_3 = nn.ReLU()

        self.layer_4 = nn.Conv2d(32, 64, (4, 4))
        self.act_4 = nn.ReLU()

        self.layer_5 = nn.Conv2d(64, 64, (3, 4), (2, 3))
        self.act_5 = nn.ReLU()

        self.fc_layer_1 = nn.Linear(25*64, 256)
        self.act_7 = nn.ReLU()

        self.bnm1 = nn.BatchNorm1d(256)

        self.fc_layer_2 = nn.Linear(256, 256)
        self.act_8 = nn.ReLU()

        self.bnm2 = nn.BatchNorm1d(256)

        self.fc_layer_3 = nn.Linear(256, 256)
        self.act_9 = nn.ReLU()

        self.bnm3 = nn.BatchNorm1d(256)

        self.fc_layer_4 = nn.Linear(256, 30)

        self.act_10 = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(self.batch_size//self.num_gpus, 1, 80, 101)
        out = self.layer_1(x)
        out = self.act_1(out)
        for module in list(self.modules())[2:-11]:
            out = module(out)
        out = out.view(self.batch_size//self.num_gpus, -1)
        for module in list(self.modules())[-11:]:
            out = module(out)
        return out


def train_model(model, total_epoch, train_loader, val_loader):
    if torch.cuda.is_available():
        model.cuda()
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters())

    trn_loss_list = []
    val_loss_list = []

    for epoch in range(total_epoch):
        trn_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # grad init
            optimizer.zero_grad()
            # forward propagation
            output = model(inputs)
            # calculate loss
            loss = criterion(output.log(), labels)
            # back propagation
            loss.backward()
            # weight update
            optimizer.step()

            # trn_loss summary
            trn_loss += loss.item()

        with torch.no_grad():
            val_loss = 0.0
            for j, val in enumerate(val_loader):
                val_x, val_label = val
                if torch.cuda.is_available():
                    val_x = val_x.cuda()
                    val_label = val_label.cuda()
                val_output = model(val_x)
                v_loss = criterion(val_output.log(), val_label)
                val_loss += v_loss

        trn_loss_list.append(trn_loss/len(train_loader))
        val_loss_list.append(val_loss/len(val_loader))
        now = time.localtime()
        print("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon,
              now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

        print("epoch: {}/{} | trn loss: {:.4f} | val loss: {:.4f} \n".format(
            epoch+1, total_epoch, trn_loss /
            len(train_loader), val_loss / len(val_loader)
        ))

    torch.save(model, "model_fin.pth")
    print("model saved complete")
