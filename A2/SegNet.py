import datetime
import json
import math
import os
import pickle
from os.path import exists

import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
from PIL import Image
from matplotlib import pyplot as plt
from torch import optim

from A2.util import performanceImage

"""
    Implementation of SegNet using PyTorch.
    
    Badrinarayanan, Vijay, Alex Kendall, and Roberto Cipolla. 
    "Segnet: A deep convolutional encoder-decoder architecture for image segmentation." 
    IEEE transactions on pattern analysis and machine intelligence 39.12 (2017): 2481-2495.

    Note: It's unfinished right now.
"""


class SegNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Containers for encoder/decoder layers
        self.encoder = []
        self.decoder = []
        self.maxunpools = []

        # First encoding layers
        self.encoder.append(nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        ))

        # Second encoding layers
        self.encoder.append(nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        ))

        # Third encoding layers
        self.encoder.append(nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        ))

        # Fourth encoding layers
        self.encoder.append(nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        ))

        # Fifth encoding layers
        self.encoder.append(nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        ))

        # First decoding layer
        self.maxunpools.append(nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0))
        self.decoder.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        ))

        # Second decoding layer
        self.maxunpools.append(nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0))
        self.decoder.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ))

        # Third decoding layer
        self.maxunpools.append(nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0))
        self.decoder.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ))

        # Fourth decoding layer
        self.maxunpools.append(nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0))
        self.decoder.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ))

        # Fifth decoding layer
        self.maxunpools.append(nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0))
        self.decoder.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=self.out_channels, kernel_size=(3, 3), padding=(1, 1)),
        ))

        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)
        self.maxunpools = nn.ModuleList(self.maxunpools)

    def forward(self, input_img):
        """
        :param input_img: Input image
        """
        dimsList = []
        indicesList = []
        res = input_img
        for i in range(len(self.encoder)):
            dimsList.append(res.size())
            res, indices = self.encoder[i](res)
            indicesList.append(indices)

        assert len(dimsList) == len(self.decoder)

        for i in range(len(self.decoder)):
            idx = len(dimsList) - i - 1
            res = self.maxunpools[i](input=res, indices=indicesList[idx], output_size=dimsList[idx])
            res = self.decoder[i](res)

        return res, nn.Softmax(dim=1)(res)


if __name__ == "__main__":
    params = {
        "epochs": 100,
        "images": {
            "1": {  # Doesn't really matter if they're named in any order or fashion, but they just can't repeat.
                "filePath": "",  # for standard image
                "classFilePath": "",  # for classification image
                "trainingCoords": [[0, 0, 100, 100]],  # [x0, y0, x1, y1]
                "testingCoords": [[0, 0, 100, 100]]  # [x0, y0, x1, y1]
            },
            "2": {
                "filePath": "",  # for standard image
                "classFilePath": "",  # for classification image
                "trainingCoords": [[0, 0, 100, 100]],  # [x0, y0, x1, y1]
                "testingCoords": [[0, 0, 100, 100]]  # [x0, y0, x1, y1]
            }
        }
    }

    try:
        with open('segnetparams.json', 'r') as f:
            params = json.load(f)
    except FileNotFoundError:
        with open('segnetparams.json', 'w') as f:
            json.dump(params, f)
        print("Please fill out the params file as it's formatted")
        exit()

    dataset = {
        "training": [],  # contains tuple (standard_image, class_image)
        "testing": []
    }

    print("Gathering image data.")
    for key in params['images']:
        standard_image: Image.Image = Image.open(params['images'][key]['filePath']).convert(mode="RGB")
        class_image: Image.Image = Image.open(params['images'][key]['classFilePath']).convert(mode="RGB")
        for arr in params['images'][key]['trainingCoords']:
            tmp_std = torch.Tensor(np.asarray(standard_image.crop((x for x in arr))))
            tmp_std = tmp_std.permute(2, 0, 1).unsqueeze(0).to('cuda')

            tmp_class = torch.Tensor(np.asarray(class_image.crop((x for x in arr))))
            tmp_class = torch.where(torch.sum(tmp_class, dim=2) > 0, 1.0, 0.0)
            tmp_class = tmp_class.unsqueeze(0).unsqueeze(0).to('cuda')

            dataset['training'].append((tmp_std, tmp_class))

        for arr in params['images'][key]['testingCoords']:
            tmp_std = torch.Tensor(np.asarray(standard_image.crop((x for x in arr))))
            tmp_std = tmp_std.permute(2, 0, 1).unsqueeze(0).to('cuda')

            tmp_class = torch.Tensor(np.asarray(class_image.crop((x for x in arr))))
            tmp_class = torch.where(torch.sum(tmp_class, dim=2) > 0, 1.0, 0.0)
            tmp_class = tmp_class.unsqueeze(0).unsqueeze(0).to('cuda')

            dataset['testing'].append((tmp_std, tmp_class))

    # Shuffle the training set
    np.random.shuffle(dataset['training'])

    # Create outputs folder for SegNet
    try:
        os.mkdir("sn-outputs")
    except FileExistsError:
        pass

    fileSuffix = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    os.mkdir("./sn-outputs/" + fileSuffix)

    bestLossSuffix = "-BCELogits"

    try:
        with open("./sn-outputs/bestLoss" + bestLossSuffix + ".dat", "rb") as f:
            best_loss = pickle.load(f)
    except FileNotFoundError:
        best_loss = math.inf

    print("Best known testing loss: " + str(best_loss))

    losses = []
    epochs = params['epochs']
    model = SegNet(3, 1).to('cuda')
    #loss_func = nn.MSELoss()
    loss_func = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    print("Starting the training loop.")
    for i in range(epochs):
        trainingLoss = 0
        tmp = []
        for item in dataset['training']:
            optimizer.zero_grad()
            pred, _ = model(item[0])
            loss = loss_func(pred, item[1])
            loss.backward()
            optimizer.step()
            tmp.append(loss.item())
        trainingLoss = np.average(tmp)
        np.random.shuffle(dataset['training'])

        testingLoss = 0
        tmp = []
        for item in dataset['testing']:
            pred, _ = model(item[0])
            loss = loss_func(pred, item[1])
            tmp.append(loss.item())
        testingLoss = np.average(tmp)

        if testingLoss < best_loss:
            best_loss = testingLoss
            torch.save(model.state_dict(), "./sn-outputs/model" + bestLossSuffix + ".pt")
            with open("./sn-outputs/bestLoss" + bestLossSuffix + ".dat", "wb") as f:
                pickle.dump(best_loss, f)

        print('Epoch', i, 'Training loss:', trainingLoss, 'Testing loss:', testingLoss)
        losses.append((trainingLoss, testingLoss))

    for j in range(len(dataset['testing'])):
        model.load_state_dict(torch.load("./sn-outputs/model" + bestLossSuffix + ".pt"))
        model.eval()
        pred = model(dataset['testing'][j][0])[0].squeeze(0).squeeze(0)
        target = dataset['testing'][j][1].squeeze(0).squeeze(0)
        performanceImage(pred, target).save("./sn-outputs/" + str(fileSuffix) + "/performance-" + str(j) + ".png")

        res = torch.where(pred >= 0.5, 1.0, 0.0)
        plt.close('all')
        data = {'y_Actual': target.flatten().tolist(),
                'y_Predicted': res.flatten().tolist()}
        df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        sn.heatmap(confusion_matrix, fmt='d', annot=True)
        plt.savefig("./sn-outputs/" + str(fileSuffix) + "/confusion-" + str(j) + ".png")

    plt.close('all')
    fig, ax = plt.subplots()
    plt.plot([i for i in range(len(losses))], [x[0] for x in losses], label="Training")
    plt.plot([i for i in range(len(losses))], [x[1] for x in losses], label="Testing")
    plt.legend()
    plt.yscale('log')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    plt.savefig('./sn-outputs/' + fileSuffix + '/loss.png')

