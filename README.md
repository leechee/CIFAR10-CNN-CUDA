# CIFAR10-CNN-CUDA
![[cifar10]](assets/cifar10.png)

The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.

## Objective

In this repository, I coded a convolutional neural network with three conv layers, three pooling layers, and two fc layers (multilayer perceptron). There is also normalization and a dropout in each layer. The activation function is ReLU and CUDA is implemented. I ran the model on an NVIDIA Geforce RTX 3050.

## Getting Started
### Python Environment
Download and install Python 3.8 or higher from the [official Python website](https://www.python.org/downloads/)

Optional, but I would recommend creating a venv. For Windows installation:
```
py -m venv .venv
.venv\Scripts\activate
```
For Unix/macOS:
```
python3 -m venv .venv
source .venv/bin/activate
```

Now install the necessary AI stack in the venv terminal. These libraries will aid with computational coding, data visualization, accuracy reports, preprocessing, etc. I used pip for this project.
```
pip install numPy
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install seaborn
```

For PyTorch:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

You will also need to install the torchvision MNIST dataset, which will be prompted in the terminal when called upon.

For CUDA: I downloaded the CUDA toolkit version 12.5 from the NVIDIA website [here](https://developer.nvidia.com/cuda-downloads). I used the network Windows 11 installation.

### Data Input
To input data from the CIFAR10 data set, use the Torchvision library. Below is the code that transforms and splits the data into two sets of loaders. The training set and testing set.

```
# import CIFAR10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465),  (0.2023, 0.1994, 0.2010))]) # normalized with MNIST mean and std

train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
```

### Results
![[loss]](assets/loss.png)

This graph demonstrates the training loss with respect to the epochs. 
I ran the model with 10 epochs. The training and testing accuracy are both 79.9%.

![[matrix]](assets/matrix.png)

This is the confusion matrix for the model's results. Here we can visualize the 79.9% testing accuracy.
