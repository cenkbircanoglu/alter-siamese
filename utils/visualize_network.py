import torch
from graphviz import Digraph
from torch.autograd import Variable

from network import MySiameseNetwork, MyLeNet, LeNet


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


inputs = torch.randn(1, 2, 100, 100)
net = MySiameseNetwork(1)
y = net(Variable(inputs))
print (net)
g = make_dot(y[0])
g.view()

inputs = torch.randn(1, 3, 32, 32)
net = LeNet(3)
y = net(Variable(inputs), Variable(inputs))
print (net)
g = make_dot(y[0])
g.view()

inputs = torch.randn(1, 6, 32, 32)
net = MyLeNet(3)
y = net(Variable(inputs))
print (net)
g = make_dot(y[0])
g.view()
