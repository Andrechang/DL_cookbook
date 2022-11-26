import torch
import torch.utils
import topt
import numpy as np

import torch.fx as fx
from torch.fx import Node
from torch.fx.passes.split_module import split_module
from functools import partial
from typing import List


class Parts:

    def __init__(self):
        self.partition_counter = 0
        self.prev_type = ''

    def tpartition(self, node: Node):
        tag = 'topt' if 'topt' in node.name else 'cpu'
        if tag != self.prev_type:
            self.partition_counter += 1
        self.prev_type = tag
        partition = tag + '_' + str(self.partition_counter)
        return partition


def t_run(tensor_in, mod=None):
    out = mod(tensor_in)
    return out


def to_topt(fx_trace: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    for nd in fx_trace.graph.nodes:
        if nd.op == 'call_module':
            mod = fx_trace.get_submodule(nd.target)
            if (isinstance(mod, torch.nn.ReLU) or
                isinstance(mod, torch.nn.Linear)
                ):
                nd.name += "_topt"

    mp = Parts()
    module_with_submodules = split_module(fx_trace, fx_trace, mp.tpartition)

    exec_graph = []  # list of compiled functions to run
    xx = example_inputs[0]
    for n in module_with_submodules.graph.nodes:
        if n.op == 'call_module':
            a = module_with_submodules.get_submodule(n.target)
            if 'topt' in n.name:
                ts_trace = torch.jit.trace(a, xx)
                ts_trace = torch.jit.freeze(ts_trace.eval())
                topt._c.topt_compile(ts_trace.graph, [xx])
                fun = partial(t_run, mod=a)
                exec_graph.append(fun)
            else:  # fallback to pytorch run
                exec_graph.append(a)
            xx = a(xx)

    def exec_topt(*args):
        outs = None
        for fun in exec_graph:
            if outs is None:
                outs = fun(args[-1])
            else:
                outs = fun(outs)
        return [outs]

    return exec_topt  # return a python callable
