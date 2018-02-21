from models.s299.inception_resnet_v2.net import InceptionResnetV2


class TripletInceptionResnetV2Net(InceptionResnetV2):
    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return (output1, output2, output3)


def get_network():
    return TripletInceptionResnetV2Net
