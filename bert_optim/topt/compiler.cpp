
#include "compiler.h"
#include <stack>
#include <numeric>
#include <c10/util/C++17.h>
#include <ATen/ATen.h>

using namespace torch::jit;

bool tdebug = false;

c10::List<int64_t> get_const_intlist(Value *input)
{
    Node *nn = input->node();
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isIntList());
    return ivalue->toIntList();
}
int64_t get_const_int(Value *input)
{
    Node *nn = input->node();
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isInt());
    return ivalue->toInt();
}
double get_const_double(Value *input)
{
    Node *nn = input->node();
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isDouble());
    return ivalue->toDouble();
}
bool get_const_bool(Value *input)
{
    Node *nn = input->node();
    assert(nn->kind() == prim::Constant);
    assert(nn->outputs().size() == 1);
    auto ivalue = toIValue(nn->outputs()[0]);
    assert(ivalue.has_value());
    assert(ivalue->isBool());
    return ivalue->toBool();
}
at::Tensor get_tensor(IValue *input)
{
    assert(input->isTensor());
    return input->toTensor();
}
c10::List<at::Tensor> get_listtensor(IValue *input)
{
    assert(input->isTensorList());
    return input->toTensorList();
}

static void print_node(Node *node)
{
    std::cout << "Running this " << node->kind().toDisplayString() << " inputs: " << node->inputs().size() << std::endl;
    std::cout << "input:";
    for (size_t ii = 0; ii < node->inputs().size(); ii++)
    {
        std::cout << node->inputs()[ii]->unique() << ", ";
    }
    std::cout << std::endl;
    std::cout << "output: ";
    for (size_t ii = 0; ii < node->outputs().size(); ii++)
    {
        std::cout << node->outputs()[ii]->unique() << ", ";
    }
    std::cout << std::endl;
}

void optCompiler::run(torch::jit::Stack *stack)
{
    // Get the number of expected inputs to the graph we are compiling
    const at::ArrayRef<Value *> &graph_inputs = subgraph_->inputs();
    const auto num_inputs = graph_inputs.size();
    // Pop these inputs from the stack.
    at::ArrayRef<IValue> inputs = last(stack, num_inputs);
    // map from IValue in stack to node input Value in subgraph_->inputs. IValue has data, Value is just pointer
    std::unordered_map<Value *, IValue> value_to_ivalue;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        auto value_input = subgraph_->inputs()[i];
        value_to_ivalue[value_input] = inputs[i];
    }
    Compiled_info *cinfo;
    CompleteArgumentSpec spec{false, ArrayRef<IValue>(inputs)};
    if (tdebug)
    {
        for (auto node : subgraph_->nodes())
            print_node(node);
    }
    at::Tensor in_data;
    // compile if we haven't compiled for the shape/device of these inputs before
    if (cache_.find(spec) == cache_.end())
    {
        Node *in_node = NULL; // get nodes that have input
        // Iterating over graph nodes is guaranteed to be topologically sorted
        cinfo = new Compiled_info();
        int out_dim = 1;
        int w_size = 1;
        for (auto node : subgraph_->nodes())
        {
            // node kind specifies what operation it is. All pytorch kinds are auto generated in: pytorch/torch/csrc/jit/generated/
            if (canHandle(node) && node->kind() != prim::Constant && node->kind() != prim::ListConstruct)
            {

                if (!in_node) // get first node of graph to get input shape
                    in_node = node;
                if (node->kind() == aten::matmul)
                {
                    assert(value_to_ivalue.find(node->input(1)) != value_to_ivalue.end());
                    cinfo->weight = get_tensor(&value_to_ivalue[node->input(1)]);
                    out_dim = cinfo->weight.sizes()[cinfo->weight.dim() - 1];
                    for (auto d : cinfo->weight.sizes())
                        w_size *= d;
                    cinfo->in_node = node;
                    if (tdebug)
                        std::cout << "Weights:" << cinfo->weight.sizes() << std::endl;
                }
                else if (node->kind() == aten::div)
                {
                    float div_const = get_const_double(node->input(1));
                    assert(node->inputs()[0]->unique() == cinfo->in_node->outputs()[0]->unique());
                    if (tdebug)
                        std::cout << "div_const: " << div_const << std::endl;
                    float *weight = (float *)cinfo->weight.data_ptr();
                    for (int x = 0; x < w_size; x++)
                        weight[x] /= div_const;
                }
            }
        }
        assert(in_node);
        assert(value_to_ivalue.find(in_node->input(0)) != value_to_ivalue.end());
        in_data = get_tensor(&value_to_ivalue[in_node->input(0)]);
        std::vector<int64_t> osizes;
        cinfo->out_size = 1;
        for (auto dim : in_data.sizes())
        {
            cinfo->out_size *= (int64_t)dim;
            osizes.push_back((int64_t)dim);
        }
        osizes.back() = out_dim;
        if (tdebug)
            std::cout << "Inputs:" << in_data.sizes() << std::endl;
        cinfo->out_shape = osizes;
        cinfo->in_node = in_node;
        cache_[spec] = cinfo;
    }
    else
    {
        cinfo = cache_[spec];
    }
    assert(cinfo);
    //---------------------------------------------------------------------------------------------
    // get input tensor from Stack using node->input Value to select the correct IValue
    assert(value_to_ivalue.find(cinfo->in_node->input(0)) != value_to_ivalue.end());
    in_data = get_tensor(&value_to_ivalue[cinfo->in_node->input(0)]);
    auto out_tensor = at::matmul(in_data, cinfo->weight);

    drop(stack, num_inputs); // remove input IValues from Stack
    for (auto &output : subgraph_->outputs())
    {
        // add output tensors to Stack for following subgraphs or nodes in the model
        auto var = torch::autograd::make_variable(out_tensor);
        stack->push_back(IValue(var));
    }
}