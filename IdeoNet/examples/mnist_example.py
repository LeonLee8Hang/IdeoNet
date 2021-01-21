import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from ideonet.network import SNNNetwork
from ideonet.network.topology import Conv2d
from ideonet.network.neuron import FedeNeuron, Input
from ideonet.learning import FedeSTDP
from ideonet.datasets import nmnist_train_test
from ideonet.models import Network


#########################################################
# Dataset
#########################################################
root = "nmnist"
if os.path.isdir(root):
    train_dataset, test_dataset = nmnist_train_test(root)
else:
    raise NotADirectoryError(
        "Make sure to download the N-MNIST dataset from https://www.garrickorchard.com/datasets/n-mnist and put it in the 'nmnist' folder."
    )

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
)


#########################################################
# Training
#########################################################
net = Network()
if torch.cuda.is_available():
    device = torch.device("cuda")
    net = net.to(torch.float16).cuda()
else:
    device = torch.device("cpu")
learning_rule = FedeSTDP(net.layer_state_dict(), lr, w_init, a)

output = []
for batch in tqdm(train_dataloader):
    input = batch[0]
    for idx in range(input.shape[-1]):
        x = input[:, :, :, :, idx].to(device)
        out, _ = net(x)
        output.append(out)

        learning_rule.step()

    net.reset_state()

    break

print(torch.stack(output, dim=-1).shape)
