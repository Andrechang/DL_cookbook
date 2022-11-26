# Adding Graph Optim to Bert in pytorch

This folder contains example implementation of some optimizations using torchscript for [Bert](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L873) 

## Install

Install pytorch tested for v1.12.0

## Build

Run will install topt library:
```
python3 setup.py install
```


## Run test

Run BertModel with topt 

```
cd tests
python3 test.py
```

To test topt on BertSelfAttention layer only

```
cd tests
python3 test.py --testL
```

To generate trace profile Bert ops 

```
cd tests
python3 test.py --genTrace
```
genTrace will generate trace using pytorch profiler that can be visualized in chrome://tracing/

## References 


[PyTorch JIT compiler tutorial](https://jott.live/markdown/Writing%20a%20Toy%20Backend%20Compiler%20for%20PyTorch).

[torch tvm example](https://github.com/pytorch/tvm)

[transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py)

[torchdynamo](https://github.com/pytorch/torchdynamo)
