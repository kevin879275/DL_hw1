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
        image_data = self.init_Image(2)
        label_data = self.init_Label(3)
        self.train_Image = image_data[0:int(split*60000), :]
        self.validation_Image = image_data[int(split * 60000): 60000]
        self.train_label = label_data[0:int(split*60000)]
        self.validation_label = label_data[int(split * 60000): 60000]

        self.train_Image = self.train_Image.astype(float) / 255
        self.validation_Image = self.validation_Image.astype(float) / 255

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

    def get_minibatch(self, batch_size=BATCH_SIZE, shuffle=False):
        if shuffle == True:
            pass

        for start in range(0, len(self.train_Image), batch_size):
            batch_image = self.train_Image[start:start+batch_size, ]
            batch_label = self.train_label[start:start+batch_size, ]
            yield batch_image, batch_label


class nn_module():
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass


def Cross_Entropy(y, y_predict):
    # -(sigma(target * log(predict))) / size
    cross_entropy = [y_predict[i][y[i]] for i in range(len(y_predict))]
    cross_entropy = np.asarray(cross_entropy)
    cross_entropy = -np.log(cross_entropy)
    return cross_entropy


def Softmax(logits):
    exps = [np.exp(i) for i in logits]
    sum_of_exps = sum(exps)
    softmax = [j / sum_of_exps for j in exps]
    return softmax


class ReLU():
    def __init__(self):
        pass

    def forward(self, input):
        return np.maximum(input, 0)

    def backward(self, input, grad_output):
        return grad_output*input if input > 0 else 0


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

    def forward(self, input):
        return np.dot(input, self.weight) + self.biases

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

    def forward(self, input, y):
        activations = []
        x = input
        for layer in self.model:
            activations.append(layer.forward(x))
            x = activations[-1]
        activations = [input] + activations
        return activations

    def backward(self, loss_grad, layer_inputs):
        for layer_index in range(len(self.model))[::-1]:
            loss_grad = self.model[layer_index].backward(
                layer_inputs[layer_index], loss_grad)

        return np.mean(loss_grad)


def main():
    dataLoader = DataLoader()
    print("train_Image shape : ", dataLoader.train_Image.shape)
    print("validation_Image shape : ", dataLoader.validation_Image.shape)
    print("train_label shape : ", dataLoader.train_label.shape)
    print("validation_label shape : ", dataLoader.validation_label.shape)

    # train_Image = dataLoader.train_Image
    # validation_Image = dataLoader.validation_Image
    # train_label = dataLoader.train_label
    # validation_label = dataLoader.validation_label
    print("training start \n")
    model = Model(dataLoader.train_Image.shape[1])
    for i in range(epochs):
        print("epoch", i+1, "/", epochs)
        for train_image_batch, train_label_batch in dataLoader.get_minibatch():
            activations = model.forward(train_image_batch, train_label_batch)
            logits = activations[-1]
            softmax = Softmax(logits)
            loss = Cross_Entropy(train_label_batch, softmax)
            loss_grad = 0

            total_loss = model.backward(loss_grad, activations)


if __name__ == '__main__':
    main()
