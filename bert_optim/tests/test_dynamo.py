import torch
import torchdynamo
from topt.py_topt import to_topt

INP = 64
OUTP = 64

class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        inP = INP
        outP = OUTP
        self.op = torch.nn.Linear(inP, outP)

    def forward(self, x):
        y = self.op(x)
        return y

example_inputs = torch.ones(1, INP)*0.1
nn_module = model()

with torchdynamo.optimize(to_topt):
    yy = nn_module(example_inputs)
