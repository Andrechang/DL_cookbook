
#include <pybind11/pybind11.h>
#include <torch/script.h>
#include <torch/extension.h>
#include <ATen/WrapDimUtils.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/graph_fuser.h>
#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/fuse_relu.h>
#include <torch/csrc/jit/passes/batch_mm.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/argument_spec.h>
#include <ATen/ATen.h>
#include "fusion_pass.h"
#include "fusion_adddiv.h"
#include "compiler.h"
#include "topt_compiler.h"

namespace py = pybind11;
using namespace torch;
using namespace torch::jit;

static bool opt_enabled = false;

void register_opt()
{
    //  Register a pass to convert the IR into one with our operator
    //  it labels ops with new symbol in graph to be executed using the new symbol
    torch::jit::RegisterPass pass([](std::shared_ptr<Graph> &g)
                                  {
            if (opt_enabled) {
                FuseLinear(g);//input to linear tensor must be dim=1
                BatchMM(g);
                FuseAddRelu(g);
                FuseAddDiv(g);
                FuseSupportedOps(g);
            } });

    // We are only dealing with pure operations (no aliasing or in place mutation), so our subgraph will always be pure
    // Register a custom compiler/implementation for our subgraph
    torch::jit::RegisterOperators op({torch::jit::Operator(
        getoptSymbol(),
        [](const torch::jit::Node *node) -> torch::jit::Operation
        {
                    auto compiler = std::make_shared<optCompiler>(node);
                    return [compiler](Stack* stack) {compiler->run(stack);return 0;}; },
        AliasAnalysisKind::PURE_FUNCTION)});
}

// creates python module torchopt that will register our custom pass and operation
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    register_opt();
    m.def("enable",[](){opt_enabled = true;});
    m.def("disable", [](){ opt_enabled = false; });
    m.def("topt_compile", &topt_compile, "topt_compile");
}