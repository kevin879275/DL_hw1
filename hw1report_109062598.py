import numpy as np
import os
import struct
import matplotlib.pyplot as plt
from PIL import Image
import math
import gzip
from sklearn import manifold

# file list
file_list = [
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
]

# Global Variable
path = "./MNIST/"
BATCH_SIZE = 64
lr = 0.01
epochs = 20

# DataLoader


class DataLoader():
    def __init__(self, split=0.7):
        '''
        Step 1 : Loading data Value
        Step 2 : split trainSet and ValidationSet with 7 : 3
        Step 3 : lets value range to [0,1]
        '''
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
        '''
        fp = File path
        >4I = unpack with 4*4 bytes (big Endian)
        Read image Data into Image List, image_header_data[1] = total image
        image Size = 28*28 pixel
        >784B = unpack with 784*1 bytes (big Endian)
        return data.np.array
        '''
        fp = path + file_list[index]
        with gzip.open(fp, 'rb') as File:
            raw_header = File.read(16)
            image_header_data = struct.unpack(">4I", raw_header)
            # print("image header data : ", image_header_data)
            list_img = []
            for i in range(image_header_data[1]):
                img = File.read(28*28)
                image_data = struct.unpack(">784B", img)
                list_img.append(image_data)
        image = np.asarray(list_img)
        return image

    def init_Label(self, index, is_Train=True):
        '''
        read 60000 label data to np.array
        >60000B = read 60000 bytes use Big Endian
        '''
        fp = path + file_list[index]
        with gzip.open(fp, 'rb') as File:
            raw_header = File.read(8)
            lab = File.read(60000)
            if(is_Train):
                label_data = struct.unpack(">60000B", lab)
            else:
                label_data = struct.unpack(">10000B", lab)
        label = np.asarray(label_data)
        self.label_data = label
        return label

    def get_minibatch(self, batch_size=BATCH_SIZE):
        '''
        for 0 to len(data), step = BATCH_SIZE
        '''
        for start in range(0, len(self.train_Image), batch_size):
            yield self.train_Image[start:start + batch_size, :], self.train_label[start:start + batch_size]

# Cross Entropy loss function


def Cross_Entropy(y, y_predict):
    """
    -(sigma(target * log(predict))) / size
    y_predict.shape = (batch_size , num_of_classes)
    y.shape = (batch_size, )
    output : loss value
    """
    reference = np.zeros_like(y_predict)
    reference[np.arange(y_predict.shape[0]), y] = 1
    mul = np.multiply(reference, np.log(y_predict))
    Sum = np.sum(mul)
    loss = - (1 / y_predict.shape[0]) * Sum
    return loss


def Grad_Cross_Entropy(y, y_predict):
    '''
    gradient of softmax + crossentropy = -t + y
    t = predict label
    y = predict softmax value
    '''
    reference = np.zeros_like(y_predict)
    reference[np.arange(y.shape[0]), y] = 1
    softmax = np.exp(y_predict)/np.sum(np.exp(y_predict),
                                       axis=-1, keepdims=True)
    return (-reference + softmax) / y_predict.shape[0]

# Softmax Layer


def Softmax(logits):
    """
    logits.shape(batch_size, num_of_classes)
    softmax = e^(aj) / e^(a sum) for j in num_of_classes
    output : softmax value (shape : (batch_size, num_of_classes))
    """
    exps = np.exp(logits)
    sum_of_exps = np.sum(exps, axis=1)
    softmax = [exps[i] / sum_of_exps[i] for i in range(sum_of_exps.shape[0])]
    return np.asarray(softmax)

# ReLU layer


class ReLU():
    def __init__(self):
        pass

    def forward(self, input):
        '''
        relu forward
        if input > 0  return input
        else return 0
        '''
        return np.maximum(input, 0)

    def backward(self, input, grad_output):
        '''
        if(input > 0) grad_input = 1
        else grad_input = 0
        return f'(input) = grad_input * grad_output
        '''
        relu_grad = input > 0
        return grad_output*relu_grad

# Dense layer


class Dense():
    def __init__(self, input, output, lr=0.1):
        '''
        init weight by normal distribution
        init bias by zero
        '''
        self.input = input
        self.output = output
        self.lr = lr
        self.weight = np.random.normal(loc=0.0,
                                       scale=np.sqrt(
                                           2/(self.input+self.output)),
                                       size=(self.input, self.output))
        self.biases = np.zeros(output)

    def forward(self, input):
        '''
        f(input) = W(input) + b
        '''
        return np.dot(input, self.weight) + self.biases

    def backward(self, input, grad_input):
        '''
        a -> (w) -> activation -> z
        input = forward propagation layer input
        grad_input = gradient value ( d C / d z) C = loss function, z = last output (use chain rule)
        grad_output = ( d C / d z) * ( d z / d a) = weight(a) * (d C / d z) , a = forward layer input before forward input z layer
        (d C / d w) = ( d z / d w ) * ( d C / d z) 
        (d C / d b) = (d z / d b) * (d C / d z), (d z / d b) = 1
        w' = w - lr * (d C / d w)
        b' = b - lr * (d C / d b)
        '''
        grad_output = np.dot(grad_input, (self.weight).T)
        # (d z / d w) * ( d C / d z) = ( d C / d w)
        grad_weights = np.dot(input.T, grad_input)
        grad_biases = grad_input.mean(axis=0)
        self.weight = self.weight - self.lr * grad_weights
        self.biases = self.biases - self.lr * grad_biases
        return grad_output

# define Model


class Model():
    def __init__(self, input_shape):
        '''
        Init model architecture
        '''
        model = []
        model.append(Dense(input_shape, 100))
        model.append(ReLU())
        model.append(Dense(100, 200))
        model.append(ReLU())
        model.append(Dense(200, 10))
        self.model = model

    # define forward propagation
    def forward(self, input):
        '''
        input = Image data, shape = (Batch_size, 784) 
        activations = all layer output
        '''
        activations = []
        x = input
        for layer in self.model:
            activations.append(layer.forward(x))
            x = activations[-1]
        activations = [input] + activations
        return activations

    # define backward propagation
    def backward(self, loss_grad, layer_inputs):
        '''
        loss_grad = gradient parameter of each layer
        layer_inputs = forward outputs in each layer 
        '''
        for layer_index in range(len(self.model))[::-1]:
            loss_grad = self.model[layer_index].backward(
                layer_inputs[layer_index], loss_grad)


def predict_(network, X, Y, is_Test=True):
    '''
    network = our model
    X = our inputs (Image)
    Y = our outputs (Label)
    is_Test : 1 = Test Case, 0 = Validation Case
    '''
    logits = network.forward(X)[-1]
    # if test only return acc
    if(is_Test == True):
        return np.mean(logits.argmax(axis=-1) == Y)
    else:  # validation return acc, loss
        softmax = Softmax(logits)
        loss = Cross_Entropy(Y, softmax)
        return np.mean(logits.argmax(axis=-1) == Y), loss, logits


def main():
    '''
    init dataloader
    '''
    dataLoader = DataLoader()
    print("train_Image shape : ", dataLoader.train_Image.shape)
    print("validation_Image shape : ", dataLoader.validation_Image.shape)
    print("train_label shape : ", dataLoader.train_label.shape)
    print("validation_label shape : ", dataLoader.validation_label.shape)

    print("training start \n")
    model = Model(dataLoader.train_Image.shape[1])
    train_loss = []
    val_loss_list = []
    train_acc = []
    val_acc = []
    print(model.model)
    '''
    logits = output value before softmax
    loss = loss value with softmax value with cross_entropy function
    loss_grad = gradient by cross_entropy and softmax
    batch = run per BATCH_SIZE 
    predict validation acc and loss
    '''
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
            model.backward(loss_grad, activations)
        train_loss.append(loss/batch)
        print("Train loss:", train_loss[-1])
        train_acc.append(
            predict_(model, dataLoader.train_Image, dataLoader.train_label))
        val_pre, val_loss, val_logits = predict_(
            model, dataLoader.validation_Image, dataLoader.validation_label, False)
        val_acc.append(val_pre)
        val_loss_list.append(val_loss)
        print("Val loss:", val_loss)
        print("Train accuracy:", train_acc[-1]*100, "%")
        print("Val accuracy:", val_acc[-1]*100, "%")
        # every 5 epoch run tsne
        if((i + 1) % 5 == 0):
            tsne = manifold.TSNE(n_components=2, init='random',
                                 random_state=5, verbose=1).fit_transform(val_logits)
            x_min, x_max = tsne.min(0), tsne.max(0)

            X_norm = (tsne - x_min) / (x_max - x_min)  # normalization
            plt.figure()
            # draw picture
            for i in range(X_norm.shape[0]):
                plt.text(X_norm[i, 0], X_norm[i, 1], str(dataLoader.validation_label[i]), color=plt.cm.Set3(dataLoader.validation_label[i]),
                         fontdict={'weight': 'bold', 'size': 9})

            plt.xticks([])
            plt.yticks([])
            plt.show()

    print("----------------Test start ------------")
    test_log = []
    test_log.append(
        predict_(model, dataLoader.test_image_data, dataLoader.test_label_data))
    '''
    draw picture with loss value
    '''
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
    print("Test accuracy:", test_log[-1]*100, "%")

    '''
    TSNE Dimensionality reduction
    '''
    logits = model.forward(dataLoader.test_image_data)[-1]
    # Dimensionality reduction 784->2
    tsne = manifold.TSNE(n_components=2, init='random',
                         random_state=5, verbose=1).fit_transform(logits)
    x_min, x_max = tsne.min(0), tsne.max(0)

    X_norm = (tsne - x_min) / (x_max - x_min)  # normalization
    plt.figure()
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(dataLoader.test_label_data[i]), color=plt.cm.Set3(dataLoader.test_label_data[i]),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    plt.show()


# main function
if __name__ == '__main__':
    main()
