import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.datasets as dsets
from torch.utils.data import DataLoader, sampler
from torchsummary import summary

from six.moves import cPickle as pickle
import os

################################################################################
# Barebones PyTorch\
################################################################################
# Before we start, we define the flatten function for your convenience.

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("using device:", device)


def flatten(x, start_dim=1, end_dim=-1):
    return x.flatten(start_dim=start_dim, end_dim=end_dim)


# Define function to evaluate accuracy
def check_accuracy(loader, model, num_batches=50):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode

    with torch.no_grad():
        for n, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=torch.float)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            if n > num_batches:
                break
        acc = float(num_correct) / num_samples
    return acc


def save_dict(di_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)


################################################################################
# Part III. PyTorch Module API
################################################################################


class ConvNet(nn.Module):
    def __init__(
        self,
        in_channel,
        channel_1,
        channel_2,
        fc_size,
        initializer="xavier_normal",
        kern_size=3,
        padding=1,
        maxpool_padding=1,
        num_classes=10,
    ):
        super().__init__()
        ############################################################################
        # TODO: Set up the layers you need for a three-layer ConvNet with the
        # architecture defined below. You should initialize the weight  of the
        # model using Kaiming normal initialization, and zero out the bias vectors.
        #
        # The network architecture should be the same as in Part II:
        #   1. Convolutional layer with channel_1 5x5 filters with zero-padding of 2
        #   2. ReLU
        #   3. Convolutional layer with channel_2 3x3 filters with zero-padding of 1
        #   4. ReLU
        #   5. Fully-connected layer to num_classes classes
        #
        # We assume that the size of the input of this network is `H = W = 32`, and
        # there is no pooing; this information is required when computing the number
        # of input channels in the last fully-connected layer.
        #
        # HINT: nn.Conv2d, nn.init.kaiming_normal_, nn.init.zeros_
        ############################################################################
        # Replace "pass" statement with your code

        # First block (3*Conv+MaxPool)
        self.cn1 = nn.Conv2d(in_channel, channel_1, kern_size, padding=padding)
        H = int((32 + 2 * padding - (kern_size - 1) - 1) / 1 + 1)
        W = int((32 + 2 * padding - (kern_size - 1) - 1) / 1 + 1)

        self.cn2 = nn.Conv2d(channel_1, channel_1, kern_size, padding=padding)
        H = int((H + 2 * padding - (kern_size - 1) - 1) / 1 + 1)
        W = int((W + 2 * padding - (kern_size - 1) - 1) / 1 + 1)

        self.cn3 = nn.Conv2d(channel_1, channel_1, kern_size, padding=padding)
        H = int((H + 2 * padding - (kern_size - 1) - 1) / 1 + 1)
        W = int((W + 2 * padding - (kern_size - 1) - 1) / 1 + 1)

        self.mpool1 = nn.MaxPool2d(kern_size, padding=maxpool_padding)
        H = 11
        W = 11

        # Second block (3*Conv+MaxPool)
        self.cn4 = nn.Conv2d(channel_1, channel_2, kern_size, padding=padding)
        H = int((H + 2 * padding - (kern_size - 1) - 1) / 1 + 1)
        W = int((W + 2 * padding - (kern_size - 1) - 1) / 1 + 1)

        self.cn5 = nn.Conv2d(channel_2, channel_2, kern_size, padding=padding)
        H = int((H + 2 * padding - (kern_size - 1) - 1) / 1 + 1)
        W = int((W + 2 * padding - (kern_size - 1) - 1) / 1 + 1)

        self.cn6 = nn.Conv2d(channel_2, channel_2, kern_size, padding=padding)
        H = int((H + 2 * padding - (kern_size - 1) - 1) / 1 + 1)
        W = int((W + 2 * padding - (kern_size - 1) - 1) / 1 + 1)

        self.mpool2 = nn.MaxPool2d(kern_size, padding=maxpool_padding)
        H = 4
        W = 4

        self.fc1 = nn.Linear(channel_2 * H * W, fc_size)
        self.fc2 = nn.Linear(fc_size, fc_size)
        self.fc3 = nn.Linear(fc_size, num_classes)

        # Initializers for weights and biases

        weight_initializer = self.get_initializer_by_name(initializer)
        weight_initializer(self.cn1.weight)
        weight_initializer(self.cn2.weight)
        weight_initializer(self.cn3.weight)
        weight_initializer(self.cn4.weight)
        weight_initializer(self.cn5.weight)
        weight_initializer(self.cn6.weight)

        weight_initializer(self.fc1.weight)
        weight_initializer(self.fc2.weight)
        weight_initializer(self.fc3.weight)

        nn.init.zeros_(self.cn1.bias)
        nn.init.zeros_(self.cn2.bias)
        nn.init.zeros_(self.cn3.bias)
        nn.init.zeros_(self.cn4.bias)
        nn.init.zeros_(self.cn5.bias)
        nn.init.zeros_(self.cn6.bias)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def get_initializer_by_name(self, init_name):
        dict = {
            "xavier_normal": nn.init.xavier_normal_,
            "xavier_uniform": nn.init.xavier_uniform_,
            "normal": nn.init.normal_,
            "uniform": nn.init.uniform_,
        }
        return dict[init_name]

    def forward(self, x):
        scores = None
        ############################################################################
        # TODO: Implement the forward function for a 3-layer ConvNet. you
        # should use the layers you defined in __init__ and specify the
        # connectivity of those layers in forward()
        # Hint: flatten (implemented at the start of part II)
        ############################################################################

        x = F.relu(self.cn1(x))
        x = F.relu(self.cn2(x))
        x = F.relu(self.cn3(x))
        x = self.mpool1(x)
        x = F.relu(self.cn4(x))
        x = F.relu(self.cn5(x))
        x = F.relu(self.cn6(x))
        x = self.mpool2(x)
        # print("MaxPool2 layer")
        # print(x.size())
        x = flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        scores = self.fc3(x)
        ############################################################################
        #                            END OF YOUR CODE
        ############################################################################
        return scores


def train_model(
    model,
    optimizer,
    save_dir,
    epochs=1,
    batch_size=64,
    schedule=[],
    verbose=True,
    print_every=10,
):
    """
  Train a model on CIFAR-10 using the PyTorch Module API.

  Inputs:
  - model: A PyTorch Module giving the model to train.
  - optimizer: An Optimizer object we will use to train the model
  - epochs: (Optional) A Python integer giving the number of epochs to train for

  Returns: Nothing, but prints model accuracies during training.
  """
    # First get the data
    loader_train, loader_val, loader_test = load_CIFAR(
        path="./cifar10_data/", batch_size=batch_size
    )
    print(f"Total number of train batches {len(loader_train)}")
    # Create directory to save model weights
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created a new directory {save_dir}")

    num_iters = epochs * len(loader_train)

    acc_train_history = torch.zeros(num_iters, dtype=torch.float)
    acc_val_history = torch.zeros(num_iters, dtype=torch.float)
    loss_history = torch.zeros(num_iters, dtype=torch.float)
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=torch.float)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            tt = t + e * len(loader_train)

            # Saving important metrics (accuracy, loss) on each batch
            acc_train_history[tt] = check_accuracy(loader_train, model)
            acc_val_history[tt] = check_accuracy(loader_val, model)
            loss_history[tt] = loss.item()
            acc_train = acc_train_history.cpu().detach().numpy()
            acc_val = acc_val_history.cpu().detach().numpy()
            loss_train = loss_history.cpu().detach().numpy()

            print(
                "Step: {:}, Train accuracy {:2.2f}, Val accuracy {:2.2f}, Loss {:2.2f}".format(
                    tt, 100 * acc_train[tt], 100 * acc_val[tt], loss_train[tt]
                )
            )
            save_dict(
                {"acc_train": acc_train, "acc_val": acc_val, "loss_train": loss_train},
                os.path.join(save_dir, "history.npy"),
            )

            if e == 0 or (tt % 10 == 0):
                path = os.path.join(save_dir, f"torch_step_{tt}.pth")
                torch.save(model.state_dict(), path)
                print('Saved model in', path)


def load_CIFAR(path, batch_size):
    NUM_TRAIN = 49000
    # The torchvision.transforms package provides tools for preprocessing data
    # and for performing data augmentation; here we set up a transform to
    # preprocess the data by subtracting the mean RGB value and dividing by the
    # standard deviation of each RGB value; we've hardcoded the mean and std.
    transform = T.Compose(
        [T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    # We set up a Dataset object for each split (train / val / test); Datasets load
    # training examples one at a time, so we wrap each Dataset in a DataLoader which
    # iterates through the Dataset and forms minibatches. We divide the CIFAR-10
    # training set into train and val sets by passing a Sampler object to the
    # DataLoader telling how it should sample from the underlying Dataset.
    cifar10_train = dsets.CIFAR10(path, train=True, download=True, transform=transform)
    loader_train = DataLoader(
        cifar10_train,
        batch_size=batch_size,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)),
    )

    cifar10_val = dsets.CIFAR10(path, train=True, download=True, transform=transform)
    loader_val = DataLoader(
        cifar10_val,
        batch_size=batch_size,
        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)),
    )

    cifar10_test = dsets.CIFAR10(path, train=False, download=True, transform=transform)
    loader_test = DataLoader(cifar10_test, batch_size=batch_size)
    return loader_train, loader_val, loader_test


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Configuration for training", add_help=True
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument(
        "--initializer",
        type=str,
        default="normal",
        choices=["normal", "xavier_normal", "xavier_uniform", "uniform"],
    )
    parser.add_argument("--weight_decay", type=float, default=0)

    args = parser.parse_args()
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    optimizer = args.optimizer
    initializer = args.initializer
    weight_decay = args.weight_decay

    print(
        f"Creating model: initializer {initializer}, optimizer {optimizer}, lr={lr}, weight decay {weight_decay}"
    )
    print(f"Train for {epochs} epochs, batch_size {batch_size}")
    torch.manual_seed(1)

    model = ConvNet(
        in_channel=3, channel_1=128, channel_2=128, fc_size=512, initializer=initializer
    )
    model.to(device)
    if optimizer == "SGD":
        opt = optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    elif optimizer == "Adam":
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    summary(model, (3, 32, 32))
    # Finally training the model on CIFAR10
    data_dir = "/content/drive/My Drive/Colab Notebooks/weights_data"
    save_dir = os.path.join(
        data_dir,
        f"{initializer}_opt_{optimizer}_lr_{lr}_wdecay_{weight_decay}_batch_{batch_size}",
    )
    train_model(model, opt, save_dir, epochs=epochs, batch_size=batch_size)
