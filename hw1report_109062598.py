import numpy as np
import os
import struct
import matplotlib.pyplot as plt
from PIL import Image
import math


file_list = [
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
]

path = "./MNIST/"
BATCH_SIZE = 64
lr = 0.01
epochs = 10


class DataLoader():
    def __init__(self, split=0.7):
        image_data = self.init_Image(2)
        label_data = self.init_Label(3)
        self.test_image_data = self.init_Image(0)
        self.test_label_data = self.init_Label(1, False)
        self.train_Image = image_data[0:int(split*60000), :]
        self.validation_Image = image_data[int(split * 60000): 60000]
        self.train_label = label_data[0:int(split*60000)]
        self.validation_label = label_data[int(split * 60000): 60000]

        self.train_Image = self.train_Image.astype(float) / 255
        self.validation_Image = self.validation_Image.astype(float) / 255
        self.test_image_data = self.test_image_data.astype(float) / 255

    def init_Image(self, index):
        fp = path + file_list[index]
        File = open(fp, 'rb')
        raw_header = File.read(16)
        image_header_data = struct.unpack(">4I", raw_header)
        # print("image header data : ", image_header_data)
        list_img = []
        for i in range(image_header_data[1]):
            img = File.read(28*28)
            image_data = struct.unpack(">784B", img)
            list_img.append(image_data)
        image = np.asarray(list_img)
        # print("image shape = ", image.shape)
        return image

    def init_Label(self, index, is_Train=True):
        fp = path + file_list[index]
        File = open(fp, 'rb')
        raw_header = File.read(8)
        # label_header_data = struct.unpack(">2I", raw_header)
        # print("label_header_data : ", label_header_data)
        lab = File.read(60000)
        if(is_Train):
            label_data = struct.unpack(">60000B", lab)
        else:
            label_data = struct.unpack(">10000B", lab)
        label = np.asarray(label_data)
        # print("label shape = ", label.shape)
        self.label_data = label
        return label

    def get_minibatch(self, batch_size=BATCH_SIZE, shuffle=True):
        if shuffle:
            indices = np.random.permutation(self.train_Image.shape[0])
        for start in range(0, len(self.train_Image), batch_size):
            if shuffle:
                excerpt = indices[start:start + batch_size]
            else:
                excerpt = slice(start, start + batch_size)
            yield self.train_Image[excerpt], self.train_label[excerpt]


def Cross_Entropy(y, y_predict):
    # -(sigma(target * log(predict))) / size
    # y_predict.shape = (batch_size , num_of_classes)
    # y.shape = (batch_size, )
    reference = np.zeros_like(y_predict)
    reference[np.arange(y_predict.shape[0]), y] = 1
    mul = np.multiply(reference, np.log(y_predict))
    Sum = np.sum(mul)
    loss = - (1 / BATCH_SIZE) * Sum
    return loss


def Grad_Cross_Entropy(y, y_predict):
    reference = np.zeros_like(y_predict)
    reference[np.arange(y.shape[0]), y] = 1
    softmax = np.exp(y_predict)/np.sum(np.exp(y_predict),
                                       axis=-1, keepdims=True)
    return (-reference + softmax) / y_predict.shape[0]
    # return (-softmax) / y_predict.shape[0]


def Softmax(logits):
    exps = np.exp(logits)
    sum_of_exps = np.sum(exps, axis=1)
    softmax = [exps[i] / sum_of_exps[i] for i in range(sum_of_exps.shape[0])]
    return np.asarray(softmax)


class ReLU():
    def __init__(self):
        pass

    def forward(self, input):
        return np.maximum(input, 0)

    def backward(self, input, grad_output):
        relu_grad = input > 0
        return grad_output*relu_grad


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

    def backward(self, input, grad_input):
        grad_output = np.dot(grad_input, (self.weight).T)
        # (d z / d w) * ( d C / d z) = ( d C / d w)
        grad_weights = np.dot(input.T, grad_input)
        grad_biases = grad_input.mean(axis=0)
        self.weight = self.weight - self.lr * grad_weights
        self.biases = self.biases - self.lr * grad_biases
        return grad_output


class Model():
    def __init__(self, input_shape):
        model = []
        model.append(Dense(input_shape, 100))
        model.append(ReLU())
        model.append(Dense(100, 200))
        model.append(ReLU())
        model.append(Dense(200, 10))
        self.model = model

    def forward(self, input):
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


def predict_(network, X, Y, is_Test=True):
    # Compute network predictions. Returning indices of largest Logit probability
    logits = network.forward(X)[-1]
    if(is_Test == True):
        return np.mean(logits.argmax(axis=-1) == Y)
    else:
        softmax = Softmax(logits)
        loss = Cross_Entropy(Y, softmax)
        return np.mean(logits.argmax(axis=-1) == Y), loss


def main():
    dataLoader = DataLoader()
    print("train_Image shape : ", dataLoader.train_Image.shape)
    print("validation_Image shape : ", dataLoader.validation_Image.shape)
    print("train_label shape : ", dataLoader.train_label.shape)
    print("validation_label shape : ", dataLoader.validation_label.shape)

    print("training start \n")
    model = Model(dataLoader.train_Image.shape[1])
    train_loss = []
    val_loss_list = []
    train_log = []
    val_log = []
    print(model.model)
    for i in range(epochs):
        print("epoch", i+1, "/", epochs)
        batch = 0
        loss = 0
        for train_image_batch, train_label_batch in dataLoader.get_minibatch():
            batch = batch + 1
            activations = model.forward(train_image_batch)
            logits = activations[-1]
            softmax = Softmax(logits)
            loss += Cross_Entropy(train_label_batch, softmax)
            loss_grad = Grad_Cross_Entropy(train_label_batch, logits)
            loss_grad = model.backward(loss_grad, activations)
        train_loss.append(loss/batch)
        print("Train loss:", train_loss[-1])
        train_log.append(
            predict_(model, dataLoader.train_Image, dataLoader.train_label))
        val_pre, val_loss = predict_(
            model, dataLoader.validation_Image, dataLoader.validation_label, False)
        val_log.append(val_pre)
        val_loss_list.append(val_loss)
        print("Val loss:", val_loss)
        print("Train accuracy:", train_log[-1])
        print("Val accuracy:", val_log[-1])

    print("----------------Test start ------------")
    test_log = []
    test_log.append(
        predict_(model, dataLoader.test_image_data, dataLoader.test_label_data))
    # plt.figure()
    list_epochs = [i + 1 for i in range(epochs)]
    plt.subplot(211)
    plt.plot(list_epochs, train_loss)
    plt.xlabel("epoch")
    plt.ylabel("Train_loss")
    plt.subplot(212)
    plt.plot(list_epochs, val_loss_list)
    plt.xlabel("epoch")
    plt.ylabel("Val_loss")
    plt.show()
    print("Test accuracy:", test_log[-1])


if __name__ == '__main__':
    main()
