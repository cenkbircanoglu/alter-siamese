from models.alexnet.alexnet import AlexNet


class SiameseALexNet(AlexNet):
    def __init__(self, input_shape=(3, 64, 64), embedding_size=10, **kwargs):
        super(SiameseALexNet, self).__init__(input_shape=input_shape, embedding_size=embedding_size, **kwargs)

    def forward_once(self, x):
        return super(SiameseALexNet, self).forward(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
