from models.alexnet.alexnet import AlexNet


class SiamALexNet(AlexNet):
    def __init__(self, input_shape=(3, 64, 64), embedding_size=10, **kwargs):
        super(SiamALexNet, self).__init__(input_shape=input_shape, embedding_size=embedding_size, **kwargs)

    def forward_once(self, x):
        out = super(SiamALexNet, self).forward(x)
        out = out.view(-1, 2, int(out.size()[1] / 2))
        return out[:, 0, :], out[:, 1, :]

    def forward(self, x):
        output1, output2 = self.forward_once(x)
        return output1, output2
