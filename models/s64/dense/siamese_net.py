from models.s64.dense.net import DenseNet


class SiameseDenseNet(DenseNet):
    def forward(self, (input1, input2)):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return (output1, output2)


def get_network():
    return SiameseDenseNet
