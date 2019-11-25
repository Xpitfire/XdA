import torch
import utils
import global_params
from networks.activations import XdA
logger = global_params.args.logger


class Net(torch.nn.Module):

    def __init__(self, inputsize, taskcla):
        super(Net, self).__init__()

        ncha, size, _ = inputsize
        self.taskcla = taskcla

        self.conv1 = torch.nn.Conv2d(ncha, 64, kernel_size=size // 8)
        s = utils.compute_conv_output_size(size, size // 8)
        s = s // 2
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=size // 10)
        s = utils.compute_conv_output_size(s, size // 10)
        s = s // 2
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=2)
        s = utils.compute_conv_output_size(s, 2)
        s = s // 2
        self.maxpool = torch.nn.MaxPool2d(2)

        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(256 * s * s, 2048)
        self.fc2 = torch.nn.Linear(2048, 2048)
        self.last = torch.nn.ModuleList()
        for t, n in self.taskcla:
            self.last.append(torch.nn.Linear(2048, n))

        tasks = len(self.taskcla)
        self.c1_xda = XdA([tasks, 64, 29, 29])
        self.c2_xda = XdA([tasks, 128, 12, 12])
        self.c3_xda = XdA([tasks, 256, 5, 5])
        self.fc1_xda = XdA([tasks, 2048])
        self.fc2_xda = XdA([tasks, 2048])

        return

    def forward(self, x, t):
        h = self.c1_xda(self.conv1(x), t)
        logger.log_hist("c1_phi", self.c1_xda.phi[t].data.cpu().numpy())
        logger.log_hist("c1_mu", self.c1_xda.mu[t].data.cpu().numpy())
        h = self.drop1(h)
        h = self.maxpool(h)

        h = self.c2_xda(self.conv2(h), t)
        logger.log_hist("c2_phi", self.c2_xda.phi[t].data.cpu().numpy())
        logger.log_hist("c2_mu", self.c2_xda.mu[t].data.cpu().numpy())
        h = self.drop1(h)
        h = self.maxpool(h)

        h = self.c3_xda(self.conv3(h), t)
        logger.log_hist("c3_phi", self.c3_xda.phi[t].data.cpu().numpy())
        logger.log_hist("c3_mu", self.c3_xda.mu[t].data.cpu().numpy())
        h = self.drop2(h)
        h = self.maxpool(h)

        h = h.view(x.size(0), -1)
        h = self.fc1_xda(self.fc1(h), t)
        logger.log_hist("fc1_phi", self.fc1_xda.phi[t].data.cpu().numpy())
        logger.log_hist("fc1_mu", self.fc1_xda.mu[t].data.cpu().numpy())
        h = self.drop2(h)
        h = self.fc2_xda(self.fc2(h), t)
        logger.log_hist("fc2_phi", self.fc2_xda.phi[t].data.cpu().numpy())
        logger.log_hist("fc2_mu", self.fc2_xda.mu[t].data.cpu().numpy())
        h = self.drop2(h)

        y = []
        for t, i in self.taskcla:
            y.append(self.last[t](h))

        return y
