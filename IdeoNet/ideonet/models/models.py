
import os
from tqdm import tqdm

import torch

from ideonet.network import SNNNetwork
from ideonet.network.topology import Conv2d
from ideonet.network.neuron import FedeNeuron, Input
from ideonet.learning import FedeSTDP

class Network(SNNNetwork):
    def __init__(self):
        super(Network, self).__init__()

        # Input
        self.input = Input(
            (batch_size, 2, 34, 34), *i_dynamics, update_type="exponential"
        )

        # Layer 1
        self.conv1 = Conv2d(2, 4, 5, (34, 34), *c_dynamics, padding=1, stride=1)
        self.neuron1 = FedeNeuron((batch_size, 4, 32, 32), *n_dynamics)
        self.add_layer("conv1", self.conv1, self.neuron1)

        # Layer 2
        self.conv2 = Conv2d(4, 8, 5, (32, 32), *c_dynamics, padding=1, stride=2)
        self.neuron2 = FedeNeuron((batch_size, 8, 15, 15), *n_dynamics)
        self.add_layer("conv2", self.conv2, self.neuron2)

        # Layer out
        self.conv3 = Conv2d(8, 1, 3, (15, 15), *c_dynamics)
        self.neuron3 = FedeNeuron((batch_size, 1, 13, 13), *n_dynamics)
        self.add_layer("conv3", self.conv3, self.neuron3)

    def forward(self, input):
        x, t = self.input(input)

        # Layer 1
        x, t = self.conv1(x, t)
        x, t = self.neuron1(x, t)

        # Layer 2
        # x = self.pool2(x)
        x, t = self.conv2(x, t)
        x, t = self.neuron2(x, t)

        # Layer out
        x, t = self.conv3(x, t)
        x, t = self.neuron3(x, t)

        return x, t