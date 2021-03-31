import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from custom_functions import Mel_spectroize, CustomDataset, VRModel, train_model

if __name__ == "__main__":
    # set number of usable gpus
    num_gpus = 4
    # set batch size
    batch_size = 256
    # set total train epoch
    total_epoch = 100
    # set path of data
    train_feature_path = './train/*.wav'
    train_label_path = 'train_answer.csv'

    # generate mel spectrogram
    mel, y_data = Mel_spectroize(train_feature_path, train_label_path)
    train_data_x, val_data_x, train_data_y, val_data_y = train_test_split(mel.numpy(), y_data,
                                                                            test_size=0.2)
    # load data
    train_dataset = CustomDataset(train_data_x, train_data_y)
    train_loader = DataLoader(dataset=train_dataset, pin_memory=True,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=10, drop_last=True)
    val_dataset = CustomDataset(val_data_x, val_data_y)
    val_loader = DataLoader(dataset=val_dataset, pin_memory=True,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=10, drop_last=True)
    # define model

    model = nn.DataParallel(VRModel(batch_size, num_gpus))
    # train model
    print(len(train_loader))
    print(len(val_loader))

    train_model(model, total_epoch, train_loader, val_loader)


