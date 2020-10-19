import numpy as np
import os
import struct
import matplotlib.pyplot as plt
from PIL import Image
import math


file_list = [
    "t10k-images.idx3-ubyte",
    "t10k-labels.idx1-ubyte",
    "train-images.idx3-ubyte",
    "train-labels.idx1-ubyte",
]

path = "./MNIST/"
BATCH_SIZE = 6
lr = 0.0001
epochs = 10


class DataLoader():
    def __init__(self, split=0.7):
        self.image_data = self.init_Image(2)
        self.label_data = self.init_Label(3)
        self.train_Image = self.image_data[0:int(split*60000), :]
        self.validation_Image = self.image_data[int(split * 60000): 60000]
        self.train_label = self.image_data[0:int(split*60000), :]
        self.validation_label = self.label_data[int(split * 60000): 60000]

    def init_Image(self, index):
        fp = path + file_list[index]
        File = open(fp, 'rb')
        raw_header = File.read(16)
        image_header_data = struct.unpack(">4I", raw_header)
        #print("image header data : ", image_header_data)
        list_img = []
        for i in range(image_header_data[1]):
            img = File.read(28*28)
            image_data = struct.unpack(">784B", img)
            list_img.append(image_data)
        image = np.asarray(list_img)
        #print("image shape = ", image.shape)
        return image

    def init_Label(self, index):
        fp = path + file_list[index]
        File = open(fp, 'rb')
        raw_header = File.read(8)
        #label_header_data = struct.unpack(">2I", raw_header)
        #print("label_header_data : ", label_header_data)
        lab = File.read(60000)
        label_data = struct.unpack(">60000B", lab)
        label = np.asarray(label_data)
        #print("label shape = ", label.shape)
        self.label_data = label
        return label


class nn_module():
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


def Cross_Entropy(y, y_predict):
    # -(sigma(target * log(predict))) / size
    cross_entropy = -np.sum(y * np.log(y_predict)) / y_predict.shape[0]
    return cross_entropy


def Softmax(y):
  # e^z(index)/sum of(all e^z(index))
    exp = [np.exp(i) for i in y]
    sum_exp = sum(exp)
    softmax = [j / sum_exp for j in exp]
    return softmax


class ReLU():
    def __init__(self):
        pass

    def forward(self, input):
        return np.maximum(input, 0)

    def backward(self, input):
        pass


class Dense():
    def __init__(self, input, output, lr=0.1):
        self.input = input
        self.output = output
        self.lr = lr
        self.weight = np.random.normal(loc=0.0,
                                       scale=np.sqrt(
                                           2/(self.input+self.output)),
                                       size=(self.input, self.output))
        self.biases = np.zeros(output)

    def forward(self):
        return np.dots()
        pass

    def backward(self):
        pass


class Model():
    def __init__(self, input_shape):
        model = []
        model.append(Dense(input_shape, 10))
        model.append(ReLU())
        #model.append(Dense(200, 10))
        # model.append(ReLU())
        self.model = model

    def forward(self, intput):
        activations = []
        x = input
        for layer in self.model:
            activations.append(layer.forward(x))
            x = activations[-1]
        return activations


def main():
    dataLoader = DataLoader()
    print("train_Image shape : ", dataLoader.train_Image.shape)
    print("validation_Image shape : ", dataLoader.validation_Image.shape)
    print("train_label shape : ", dataLoader.train_label.shape)
    print("validation_label shape : ", dataLoader.validation_label.shape)

    train_Image = dataLoader.train_Image
    validation_Image = dataLoader.validation_Image
    train_label = dataLoader.train_label
    validation_label = dataLoader.validation_label

    print("training start \n")

    for i in range(epochs):
        print("epoch", i+1, "/", epochs)
        for i in range(train_Image.shape[0]):
            x = train_Image[i]
            y = train_label[i]
            model = Model(x.shape[1])
            activatons = model.forward(x)
            layer_inputs = [x] + activations
            logits = activations[-1]
            loss = Cross_Entropy(logits, y)


if __name__ == '__main__':
    main()
