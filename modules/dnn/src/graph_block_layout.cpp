// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include "net_impl.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

using PLayer = Ptr<Layer>;
using PGraph = Ptr<Graph>;

/* Inserts layout conversion operations (if needed) into the model graph and subgraphs.

 Some of the operations
 (let's call them 'operations of category B' or B-operations, 'B' stands for 'Block'),
 most notably Convolution (including depthwise convolution), ConvTranspose, MaxPool and AveragePool,
 can be computed more efficiently if data is represented in so-called block layout,
 i.e. when 4D tensor NxCxHxW is represented in memory as 5D tensor NxC1xHxWxC0,
 where all C channels of the original tensor are split into C1 groups of C0 channels each.
 For each spatial location (y=y0, x=x0) each group of C0 channels is stored sequentially.
 C1 is thus computed as following: C1 = (C + C0-1)/C0, where division is performed with truncation.

 Some other operations (let's call them A-operations, 'A' stands for 'Any'),
 most notably unary element-wise operations or special cases of binary
 element-wise operations can be efficiently computed in any layout, including block layout.

 Finally, all other operations (denoted as C-operations, 'C' for 'Casual' or 'Channels')
 do not support block layout at all. The inputs should come in the original model format
 (e.g. NCHW in the case of Onnx).

 We want to transform the graph so that:
 1. B-operations always take the inputs in block layout. If not, the inputs must be converted to
    block layout prior to B-operation. Note that only the first input should be converted
    in the case of Convolution, convolution weights (if constant) are pre-processed separately.
 2. C-operations always take the inputs in non-block layout, e.g. NCHW. If some of the inputs are
    stored in block layout, they must be converted from block layout prior to C-operation.
 3. the number of layout transformation operations is minimal,
    i.e. we don't do transformations unless it's necessary.

 Note that this graph transformation is applied after fusion, since inside a fused operation
 (e.g. 'Convolution + Batch Norm + Activation + Adding Skip connection') we don't need to
 transform layout. That is, we have to deal with less operations at this stage.
*/

struct BlockLayoutTransformer
{
    BlockLayoutTransformer(Net::Impl* netimpl_) : netimpl(netimpl_) {}

    Net::Impl* netimpl;
    vector<DataLayout> layouts; // layouts for each argument
    vector<Arg> blockCache; // if an Arg needs to be converted to block layout and
                            // if it's used by several operations,
                            // then we reuse once transformed arg,
                            // don't transform it several times
    vector<Arg> nonblockCache; // another cache of non-block args
    DataLayout defaultLayout;
    int trlayoutIdx;

    std::pair<Arg, PLayer> getProperArg(const Arg& arg, bool block, int defaultC0)
    {
        if (arg.empty() || (layouts[arg.idx] == DATA_LAYOUT_BLOCK) == block)
            return {arg, PLayer()};
        std::vector<Arg> *cache, *altCache;
        if (block) {
            cache = &blockCache;
            altCache = &nonblockCache;
        } else {
            cache = &nonblockCache;
            altCache = &blockCache;
        }
        Arg cached = cache->at(arg.idx);
        if (!cached.empty())
            return {cached, PLayer()};
        size_t nargs = netimpl->args.size();
        CV_Assert(layouts.size() == nargs &&
                  cache->size() == nargs &&
                  altCache->size() == nargs);

        const ArgData& adata = netimpl->args.at(arg.idx);
        const char* suffix = block ? "block" : "nonblock";

        std::string newname = format("%s.%s", adata.name.c_str(), suffix);
        int idx = 1;
        for ( ; netimpl->haveArg(newname); idx++) {
            newname = format("%s.%s%d", adata.name.c_str(), suffix, idx);
        }

        cached = netimpl->newArg(newname, DNN_ARG_TEMP);
        CV_Assert(size_t(cached.idx) == nargs);

        cache->at(arg.idx) = cached;
        cache->push_back(cached);
        altCache->push_back(arg);

        LayerParams params;
        params.name = format("trlayout.%d", trlayoutIdx++);
        params.type = "TransformLayout";
        params.set("layout", int(block ? DATA_LAYOUT_BLOCK : defaultLayout));
        params.set("C0", int(defaultC0));
        PLayer trlayer = TransformLayoutLayer::create(params);

        trlayer->netimpl = netimpl;
        trlayer->inputs = {arg};
        trlayer->outputs = {cached};
        layouts.push_back(block ? DATA_LAYOUT_BLOCK : defaultLayout);

        return {cached, trlayer};
    }

    void transformGraph(PGraph& g)
    {
        const vector<PLayer>& currProg = g->prog();
        int defaultC0 = netimpl->defaultC0;
        vector<PLayer> newProg;
        std::vector<Arg> newInputs;
        std::vector<DataLayout> inputLayoutsOrig, inputLayoutsNew, outputLayouts;
        size_t nchanges = 0;

        for (const PLayer& layer: currProg) {
            const vector<Arg>& inputs = layer->inputs;
            const vector<Arg>& outputs = layer->outputs;
            size_t ninputs = inputs.size(), noutputs = outputs.size();
            std::string op_name = layer->type;
            std::string name = layer->name;
            vector<PGraph>* subgraphs = layer->subgraphs();
            std::cout << "name: " << name << ", op_name: " << op_name << ", inp0 layout: " << layoutToString(layouts[inputs[0].idx]) << "\n";

            if (subgraphs) {
                for (PGraph& subgraph: *subgraphs) {
                    transformGraph(subgraph);
                    nchanges++;
                }
            }

            inputLayoutsOrig.clear();
            for (size_t i = 0; i < ninputs; i++) {
                inputLayoutsOrig.push_back(layouts.at(inputs[i].idx));
            }

            layer->getLayouts(inputLayoutsOrig, inputLayoutsNew, int(noutputs), outputLayouts);
            CV_Assert(inputLayoutsNew.size() == ninputs);
            CV_Assert(outputLayouts.size() == noutputs);

            newInputs.clear();
            bool changedInputs = false;
            for (size_t i = 0; i < ninputs; i++) {
                Arg inp = inputs[i];
                DataLayout prevLayout = inputLayoutsOrig[i];
                DataLayout newLayout = inputLayoutsNew[i];
                if (!inp.empty()) {
                    if (prevLayout == DATA_LAYOUT_BLOCK) {
                        blockCache.at(inp.idx) = inp;
                    } else {
                        nonblockCache.at(inp.idx) = inp;
                    }
                }
                if (newLayout != prevLayout &&
                    (newLayout == DATA_LAYOUT_BLOCK ||
                     prevLayout == DATA_LAYOUT_BLOCK)) {
                    auto p = getProperArg(inp, newLayout == DATA_LAYOUT_BLOCK, defaultC0);
                    newInputs.push_back(p.first);
                    if (p.second) {
                        newProg.push_back(p.second);
                    }
                    changedInputs = true;
                    nchanges++;
                } else {
                    newInputs.push_back(inp);
                }
            }

            for (size_t i = 0; i < noutputs; i++) {
                layouts.at(outputs[i].idx) = outputLayouts[i];
            }

            if (changedInputs) {
                layer->inputs = newInputs;
            }
            newProg.push_back(layer);
        }

        if (nchanges > 0) {
            g->setProg(newProg);
        }
    }

    void transform()
    {
        size_t nargs = netimpl->args.size();
        defaultLayout = netimpl->originalLayout;
        trlayoutIdx = 0;

        layouts.assign(nargs, DATA_LAYOUT_UNKNOWN);
        blockCache.assign(nargs, Arg());
        nonblockCache.assign(nargs, Arg());

        transformGraph(netimpl->mainGraph);
    }
};

void Net::Impl::useBlockLayout()
{
    BlockLayoutTransformer use_block_layout(this);
    use_block_layout.transform();
}

CV__DNN_INLINE_NS_END
}}
