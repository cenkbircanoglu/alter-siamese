import torch
from graphviz import Digraph
from torch.autograd import Variable

from models.s32.vgg.net import Net
from models.s32.vgg.siam_net import SiamNet
from models.s32.vgg.siamese_net import SiameseNet


def make_dot(var):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, Variable):
                value = '(' + (', ').join(['%d' % v for v in var.size()]) + ')'
                dot.node(str(id(var)), str(value), fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'previous_functions'):
                for u in var.previous_functions:
                    dot.edge(str(id(u[0])), str(id(var)))
                    add_nodes(u[0])

    add_nodes(var.grad_fn)
    return dot


inputs = torch.randn(1, 3, 32, 32)
net = SiameseNet()
y = net((Variable(inputs),Variable(inputs)))
print (net)
g = make_dot(y[0])
g.view()

