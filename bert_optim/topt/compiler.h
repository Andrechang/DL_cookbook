

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <ATen/ATen.h>
#include "fusion_pass.h"

using namespace torch::jit;

class Compiled_info
{
public:
    uint64_t out_size;              // total output size
    std::vector<int64_t> out_shape; // output shape
    at::Tensor weight;
    Node *in_node;
    Compiled_info(){};
    ~Compiled_info(){};
};

class optCompiler
{
public:
    int g_size; // graph size
    optCompiler(const torch::jit::Node *node)
        : subgraph_(node->g(torch::jit::attr::Subgraph))
    {
        g_size = 0;
        for (auto node : subgraph_->nodes())
            g_size++;
    };
    ~optCompiler()
    {
        for (auto it : cache_)
            delete it.second;
    };
    void run(torch::jit::Stack *stack);
    // subgraph is the custom function that the fusion_pass created "opt::CompilationGroup"
    // implementation of the custom subgraph
    // Stack is the tensor inputs of the subgraph. The output is added to the Stack

private:
    std::shared_ptr<torch::jit::Graph> subgraph_;                                 // subgraph with node to be run
    std::unordered_map<torch::jit::CompleteArgumentSpec, Compiled_info *> cache_; // cache what has been compiled
};