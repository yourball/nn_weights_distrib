import numpy as np
from keras.layers import (
    Conv2D,
    Dense,
    BatchNormalization,
    MaxPooling2D,
    Activation,
    Dropout,
    Flatten,
)
from keras.models import Sequential
from keras import regularizers
from utils import build_model
import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam, SGD
import os
from six.moves import cPickle as pickle
from keras.callbacks import Callback


def save_dict(di_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)


def prepare_data():
    # Prepare Cifar data
    num_classes = 10
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")
    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, x_test, y_train, y_test


class WeightsSaver(Callback):
    def __init__(self, dir_name, batch_granularity):
        self.batch_granularity = batch_granularity
        self.dir_name = dir_name
        self.batch = 0
        self.epoch = 0

    def on_batch_end(self, batch, logs={}):
        save_after_batch = (self.batch % self.batch_granularity == 0)
        if save_after_batch:
            name = os.path.join(self.dir_name, f'weights_batch={self.batch}.h5')
            self.model.save_weights(name)
        self.batch += 1

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1


class LossHistory(Callback):
    def __init__(self, dir_name):
        self.dir_name = dir_name

    def on_train_begin(self, logs={}):
        self.history = {'loss': [],
                        'val_loss': [],
                        'accuracy': [],
                        'val_accuracy': [],
                        }

    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['accuracy'].append(logs.get('accuracy'))
        self.history['val_accuracy'].append(logs.get('val_accuracy'))
        save_dict(self.history, os.path.join(dir_name, "history.h5"))


def train_model(
    model, dir_name, x_train, y_train, x_test, y_test, epochs, batch_size, seed
):
    full_hist = []
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[WeightsSaver(dir_name, 1), LossHistory(dir_name)]
    )
    save_dict(history.history, os.path.join(dir_name, "epoch_history.h5"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Configuration for training", add_help=True
    )

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--opt_name", type=str)
    parser.add_argument("--initializer", type=str)
    parser.add_argument("--augmentation", type=str)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--padding", type=str)

    args = parser.parse_args()
    lr = args.lr
    batch_size = args.batch_size
    opt_name = args.opt_name
    initializer = args.initializer
    padding = args.padding
    augmentation = args.augmentation == "True"
    seed = args.seed
    l2 = args.l2

    epochs = 5
    print('! Number of epochs', epochs)
    x_train, x_test, y_train, y_test = prepare_data()
    model = build_model(initializer=initializer, padding=padding, l2_penalty=l2, seed=1)
    if opt_name == "SGD":
        opt = SGD(lr, momentum=0.9)
    elif opt_name == "Adam":
        opt = Adam(lr)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    dir_name = f"/content/drive/My Drive/Colab Notebooks/data/{initializer}_{padding}_lr_{lr}_opt_{opt_name}_l2_{l2}_batch_{batch_size}_seed_{seed}"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"Created a new directory {dir_name}")
    train_model(
        model,
        dir_name,
        x_train,
        y_train,
        x_test,
        y_test,
        epochs,
        batch_size,
        seed,
    )
