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
BATCH_SIZE = 32
lr = 0.1
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
    cross_entropy = [y_predict[i][y[i]] for i in range(len(y_predict))]
    cross_entropy = np.asarray(cross_entropy)
    x_cross_entropy = -np.log(cross_entropy)
    return x_cross_entropy


def Grad_Cross_Entropy(y, y_predict):
    reference = np.zeros_like(y_predict)
    reference[np.arange(y.shape[0]), y] = 1
    softmax = np.exp(y_predict)/np.sum(np.exp(y_predict),
                                       axis=-1, keepdims=True)
    # return (-reference + softmax) / y_predict.shape[0]
    return (-softmax) / y_predict.shape[0]


def Softmax(logits):
    exps = [np.exp(i) for i in logits]
    sum_of_exps = sum(exps)
    softmax = [j / sum_of_exps for j in exps]
    return softmax


def softmax_crossentropy_with_logits(logits, reference_answers):
    # Compute crossentropy from logits[batch,n_classes] and ids of correct answers
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]

    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))

    return xentropy


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


def predict(train_label, predict_label):
    correct = 0
    for i in range(len(predict_label)):
        if(train_label[i] == np.argmax(predict_label[i])):
            correct = correct + 1
    return correct


def predict_(network, X):
    # Compute network predictions. Returning indices of largest Logit probability
    logits = network.forward(X)[-1]
    return logits.argmax(axis=-1)


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
    train_loss = []
    train_log = []
    val_log = []
    for i in range(epochs):
        print("epoch", i+1, "/", epochs)
        batch = 0
        loss = 0
        for train_image_batch, train_label_batch in dataLoader.get_minibatch():
            batch = batch + 1
            activations = model.forward(train_image_batch)
            logits = activations[-1]
            softmax = Softmax(logits)
            loss += sum(Cross_Entropy(train_label_batch, softmax)) / BATCH_SIZE
            # loss += sum(softmax_crossentropy_with_logits(logits,
            #                                              train_label_batch)) / BATCH_SIZE
            loss_grad = Grad_Cross_Entropy(train_label_batch, logits)
            grad_loss = model.backward(loss_grad, activations)
            # acc += predict(train_label_batch, softmax)
            # if(batch % 40 == 0):
            #     print("Batch : ", batch * BATCH_SIZE, "/",
            #           dataLoader.train_Image.shape[0], "  : loss = ", loss / batch, " : acc = ", (acc * 100) / (batch * BATCH_SIZE), " % ")
        train_log.append(loss/batch)
        print("Train loss:", train_log[-1])
        train_log.append(
            np.mean(predict_(model, dataLoader.train_Image) == dataLoader.train_label))
        val_log.append(
            np.mean(predict_(model, dataLoader.validation_Image) == dataLoader.validation_label))
        print("Train accuracy:", train_log[-1])
        print("Val accuracy:", val_log[-1])

        print("----------------Test start ------------")
    test_log = []
    test_log.append(
        np.mean(predict_(model, dataLoader.test_image_data) == dataLoader.test_label_data))

    print("Test accuracy:", test_log[-1])


if __name__ == '__main__':
    main()
