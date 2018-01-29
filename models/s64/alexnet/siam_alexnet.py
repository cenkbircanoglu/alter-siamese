import torch

from models.s64.alexnet.alexnet import AlexNet


class SiamAlexNet(AlexNet):
    def forward(self, (input1, input2)):
        x = torch.cat([input1, input2], dim=0)
        output = self.forward_once(x)
        output = output.view(-1, 2, int(output.size()[1] / 2))
        output1, output2 = output[:, 0, :], output[:, 1, :]
        return output1, output2
