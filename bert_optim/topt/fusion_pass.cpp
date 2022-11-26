// based on: https://github.com/pytorch/tvm
#include "fusion_pass.h"

#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/common_subexpression_elimination.h>
#include <torch/csrc/jit/passes/dead_code_elimination.h>

using namespace torch::jit;

value_list sortReverseTopological(ArrayRef<Value *> inputs, Block *block)
{
    value_list result;
    for (auto i : inputs)
    {
        if (i->node()->owningBlock() == block)
        {
            result.push_back(i);
        }
    }
    // Sort in reverse topological order
    std::sort(result.begin(), result.end(), [&](Value *a, Value *b)
              { return a->node()->isAfter(b->node()); });
    return result;
}

// aten:: operations that is supported and can be combined together into opt::CompilationGroup
bool canHandle(const torch::jit::Node *node)
{
    switch (node->kind())
    {
    case aten::matmul:
    case aten::div:
        return true;
    default:
        return false;
    }
    return false;
}

bool canHandle(Block *block)
{
    for (Node *node : block->nodes())
    {
        if (!canHandle(node))
        {
            return false;
        }
    }
    return true;
}

#define CHECKEQ(cond)        \
    if (!(cond))             \
    {                        \
        return c10::nullopt; \
    }

c10::optional<Node *> tryMerge(
    Node *consumer,
    Node *producer,
    AliasDb &aliasDb)
{
    // std::cout << "Trying producer " << producer->kind().toQualString() << " and consumer " << consumer->kind().toQualString() << ":\n" << std::endl;

    // Symbolic checks
    CHECKEQ(canHandle(producer));
    CHECKEQ((canHandle(consumer) || consumer->kind() == getoptSymbol()));

    // Alias checks
    // Requirement:
    // - moveAfterTopologicallyValid(consumer, producer)
    // - One of:
    //   1) Both are in-place ops
    //   2) Consumer is in-place, producer !hasInputWriters
    //   3) Producer is in-place, consumer !hasOutputWriters
    CHECKEQ(aliasDb.moveAfterTopologicallyValid(consumer, producer));

    // 1)
    if (!(aliasDb.isMutable(consumer) && aliasDb.isMutable(producer)))
    {
        // 2)
        if (aliasDb.isMutable(consumer))
        {
            CHECKEQ(!aliasDb.hasInputWriters(producer));
            // 3)
        }
        else if (aliasDb.isMutable(producer))
        {
            CHECKEQ(!aliasDb.hasOutputWriters(consumer));
        }
    }
    if (!consumer->hasAttribute(attr::Subgraph) &&
        consumer->kind() != getoptSymbol())
    {
        consumer = SubgraphUtils::createSingletonSubgraph(consumer, getoptSymbol());
    }
    if (producer->kind() == prim::Constant)
    {
        auto &subgraph = consumer->g(attr::Subgraph);
        Node *in_const = subgraph->createClone(producer, [](Value *) -> Value *
                                               { throw std::runtime_error("unexpected input"); });
        subgraph->insertNode(in_const);
    }
    else
    {
        SubgraphUtils::mergeNodeIntoSubgraph(producer, consumer);
    }

    return consumer;
}

std::pair<graph_node_list::iterator, bool> scanNode(
    Node *consumer,
    AliasDb &aliasDb,
    Block *block)
{
    auto inputs = sortReverseTopological(consumer->inputs(), block);
    for (auto input : inputs)
    {
        if (auto group = tryMerge(consumer, input->node(), aliasDb))
        {
            return {group.value()->reverseIterator(), true};
        }
    }
    return {++consumer->reverseIterator(), false};
}

// Register a pass that will coalesce operators we can handle  into a single operator containing a subgraph.
void FuseSupportedOps(std::shared_ptr<Graph> graph)
{
    AliasDb aliasDb(graph);
    auto block = graph->block();

    bool any_changed{true};
    while (any_changed)
    {
        any_changed = false;
        for (auto it = block->nodes().rbegin(); it != block->nodes().rend();)
        {
            bool changed;
            std::tie(it, changed) = scanNode(*it, aliasDb, block);
            any_changed |= changed;
        }
    }
    EliminateCommonSubexpression(graph);
    EliminateDeadCode(graph);
}

// node kind is called a Symbol. Our subgraph is called "opt::CompilationGroup"
const torch::jit::Symbol &getoptSymbol()
{
    static torch::jit::Symbol sym =
        torch::jit::Symbol::fromQualString("opt::CompilationGroup");
    return sym;
}