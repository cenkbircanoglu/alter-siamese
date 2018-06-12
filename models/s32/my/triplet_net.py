from models.s32.my.net import MyNet


class TripletMyNet(MyNet):

    def forward(self, input1, input2, input3):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)
        return (output1, output2, output3)


def get_network():
    return TripletMyNet
