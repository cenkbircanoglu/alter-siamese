from models.s224.inception_resnet_v2.net import InceptionResnetV2


class SiameseInceptionResnetV2Net(InceptionResnetV2):
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return (output1, output2)


def get_network():
    return SiameseInceptionResnetV2Net
