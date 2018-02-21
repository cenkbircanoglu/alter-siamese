import torch

from models.s299.inception_resnet_v2.net import InceptionResnetV2


class SiamInceptionResnetV2Net(InceptionResnetV2):
    def forward(self, (input1, input2)):
        x = torch.cat([input1, input2], dim=1)
        output = self.forward_once(x)
        output = output.view(-1, 2, int(output.size()[1] / 2))
        output1, output2 = output[:, 0, :], output[:, 1, :]
        return (output1, output2)


def get_network():
    return SiamInceptionResnetV2Net
