// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "../engine/engine.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using std::vector;
using std::string;

/* Assigns buffers for all intermediate tensors of the graph/model

The algorithm is quite simple, but there are some nuances in the attempt to re-use memory more efficiently:

All layer arguments in graph and sub-graphs are classified into 4 categories:
a) inputs, b) outputs, c) constants and d) temporary values/tensors.

Except for the temporary values ("d" category), each other argument gets
its own dedicated storage, which makes things more clear and predictable.
So, this algorithm assigns buffers only for the temporary values.

During the inference process, each temporary value is computed
by one of the layers and then used by zero or more subsequent layers (only as input).
An example of a model where some tensors are used more than once is Resnet.
After a tensor is used for the last time and
won't be used in any subsequent layer, the memory buffer for that tensor could be re-used for
other arguments. We want to assign each temporary tensor to some temporary buffer,
and it's typically N:1 mapping.

We do it using 2-stage algorithm:

1. First, we calculate, how many times each argument is used and store the counters into 'usecounts'.
2. Second, we scan the layers in topologically sorted order
    2.0. Sanity check: We check that each input argument of the operation is either input or constant,
         or it's a temporary tensor with the buffer assigned to it.
         If not, then the layers are not sorted in a topological order.
    2.1. For in-place reshape operations, such as squeeze/unsqueeze/flatten etc.
         or for unary element-wise operations,
         we check whether the input is a temporary value and is not used in any subsequent operations.
         If these checks all pass, we assign output argument to the same buffer as input. Note that
         we don't try to reuse inputs of binary/ternary etc. operation because of the broadcasting.
         We need to do symbolic shape inference to proof that the output is of the same shape as one of the inputs.
    2.2. Otherwise, for each output argument of operation, which is not a network output argument.
         we assign the most recently-used free buffer (i.e. the top buffer in the stack of free buffers).
         If there is no free buffers, i.e. the stack is empty, we create a new buffer, and use it.
    2.3. For each input we decrement the corresponding element of 'usecounts'. If the counter reaches 0 and the input
         is not aliased with one of the outputs (see 2.1),
         we push the corresponding buffer index into the stack of free buffers.
    2.4. In the case of in-place operations and sometimes when using subgraphs (e.g. in If, Loop operations) we may
         re-use the same buffer for several arguments
         (which can be ouputs for some operations and inputs for some subsequent operations).
         In order to handle it all properly, during the buffer assignment algorithm we maintain use counter for each
         buffer, which should not be confused with use counters for arguments. A pool of free buffers contains zero or
         more "spare" buffers with 0 use counts. A buffer in use has the corresponding usage count > 0.
         When some argument is not needed anymore, and if it's not a constant, it decrements the usage counter of the buffer
         where it resides. When the counter reaches zero, we return the buffer into the pool of free buffers and then
         we can reuse the same buffer for another argument (or probably different shape and/or type, see below).
         In principle, we could 'protect' some buffers from the premature release and re-use by incrementing the use counts
         of the respective arguments that reside in those buffers, but that would make the bookkeeping much more complex.

Please, note that when we reuse buffers, we don't check any types, shape or a total size of the buffer needed.
We reallocate each buffer at runtime to fit each single argument that it's used for. For example, let's say the buffer #3
is used for arguments #5 (10x10x10 FP32), #10 (6x6x32 FP32) and #14 (300x1 UINT64). Then during the the first run of
the inference the buffer #3 will be reallocated from 0 bytes to 1000*4 bytes to fit arg #10,
then from 4000 to 6*6*32*4=4608 bytes to fit arg #10 and then it will fit arg #14 without reallocations.
During the second run of inference with the same resolution input the buffer will not be reallocated.

The reallocation is done using Buffer.fit() function.
*/

struct BufferAllocator
{
    BufferAllocator(Net2::Impl* netimpl_) : netimpl(netimpl_) {}

    Net2::Impl* netimpl;
    vector<int> usecounts;
    vector<int> freebufs;
    vector<int> buf_usecounts;
    vector<int> bufidxs;
    int nbufs = 0;

    /*
        Here are 3 workhorse methods that abstract the use and bookkeeping of buffers:
        1. getFreeBuffer() takes the first spare buffer from the pool of free buffers. Since
           we don't necessarily know the shape/type of tensor type at this stage, this is quite
           reasonable behaviour - we cannot do anything more complex that that. On the positive side,
           since the pool of free buffers operates like a stack, the first free buffer is the most
           recently released buffer, so we improve cache locality using this pattern.
           When we don't have spare buffers in the pool, we "virtually" create a new buffer
           (by incrementing the number of buffers used) and return it.

           For the retrieved buffer we set its use count to 1.
        2. releaseBuffer(bufidx) decrements the buffer use count and returns it to the pool
           of free buffers as long as the use counter reaches 0.
        3. shareBuffer(from_arg, to_arg) takes two argument indices.
           It makes argument 'to_arg' use the same buffer as 'from_arg'.
           Use counter for the assigned to 'to_arg' buffer (if any) is decremented.
           Use counter for the 'from_arg' buffer is incremented, correpondingly.
    */

    int getFreeBuffer()
    {
        if (freebufs.empty()) {
            freebufs.push_back(nbufs);
            buf_usecounts.push_back(0);
            //printf("added buf %d\n", nbufs);
            nbufs++;
        }
        int outidx = freebufs.back();
        freebufs.pop_back();
        buf_usecounts[outidx] = 1;
        return outidx;
    }

    void releaseBuffer(int bufidx)
    {
        if (bufidx >= 0) {
            CV_Assert(buf_usecounts[bufidx] > 0);
            if (--buf_usecounts[bufidx] == 0)
                freebufs.push_back(bufidx);
        }
    }

    void shareBuffer(int fromArg, int toArg)
    {
        CV_Assert(!netimpl->isConst(fromArg) && !netimpl->isConst(toArg));
        int fromBuf = bufidxs[fromArg], toBuf = bufidxs[toArg];
        CV_Assert(fromBuf >= 0);
        bufidxs[toArg] = fromBuf;
        buf_usecounts[fromBuf]++;
        if (toBuf >= 0)
            releaseBuffer(toBuf);
    }

    void assign()
    {
        netimpl->useCounts(usecounts);
        size_t nargs = usecounts.size();
        bufidxs.assign(nargs, -1);
        nbufs = 0;
        assign(netimpl->graph);
        netimpl->bufidxs = bufidxs;
        netimpl->buffers.resize(nbufs);
        for (int i = 0; i < nbufs; i++)
            netimpl->buffers[i] = Buffer();
    }

    void assign(const Graph& graph)
    {
        for (const auto& node: graph.prog) {
            Layer* layer;
            bool inplace = false;
            int reuseIdx = -1;
            int i, ninps, nouts;

            if (node.op.empty()) continue;

            layer = node.op.get();
            ninps = (int)node.inputs.size();
            nouts = (int)node.outputs.size();

            /*
                Determine if we can possibly re-use some of the input buffers for the output as well,
                in other words, whether we can run the operation in-place.
                Not only it saves memory, but it can also:
                   1. improve L2/L3 cache re-use
                   2. effectively convert some copy/re-shape operations
                      (Identity, Flatten, Reshape, Squeeze, Unsqueeze)
                      into Nop (no-operation).
            */
            if (dynamic_cast<BatchNormLayer*>(layer) != 0 ||
                dynamic_cast<BlankLayer*>(layer) != 0 ||
                dynamic_cast<FlattenLayer*>(layer) != 0 ||
                dynamic_cast<ActivationLayer*>(layer) != 0 ||
                dynamic_cast<ReshapeLayer*>(layer) != 0 /*||
                dynamic_cast<SqueezeLayer*>(layer) != 0 ||
                dynamic_cast<UnsqueezeLayer*>(layer) != 0*/) {
                CV_Assert(ninps >= 1);
                int inp0 = node.inputs[0];
                inplace = netimpl->kind(inp0) == DNN_ARG_TEMP && usecounts[inp0] == 1;
                reuseIdx = inp0;
            } /*else if (dynamic_cast<ScatterLayer*>(layer) != 0) {
                CV_Assert(ninps == 3);
                int inp0 = node.inputs[0];
                inplace_op = netimpl->kind(inp0) == DNN_ARG_TEMP &&
                             usecounts[inp0] == 1 &&
                             inp0 != net.inputs[1] &&
                             inp0 != net.inputs[2];
                reuse_idx = inp0;
            } else if (dynamic_cast<Convolution*>(layer) != 0) {
                Convolution* conv = dynamic_cast<Convolution*>(layer);
                if (ninps >= 4 && conv->passby >= 0) {
                    int passby = conv->passby;
                    inplace_op = netimpl->kind(passby) == DNN_ARG_TEMP &&
                                 usecounts[passby] == 1 &&
                    reuse_idx = passby;
                }
            }*/

            /*
               Unless the operation is in-place, assign buffers for each output.
               We do it before we recursively process subgraphs inside If/Loop/Scan.
               this way we avoid any possible influence of buffer allocation inside a subgraph
               to the parent graphs.
            */
            //if (layer->type == "Softmax")
            //    putchar('.');
            if (nouts > 0 && netimpl->kind(node.outputs[0]) == DNN_ARG_TEMP) {
                int out0 = node.outputs[0];
                if (inplace && bufidxs.at(out0) < 0)
                    shareBuffer(reuseIdx, out0);
                else
                    for (auto argidx: node.outputs) {
                        if (bufidxs.at(argidx) < 0)
                            bufidxs.at(argidx) = getFreeBuffer();
                    }
            }

            if (layer->type == "If") {
                /*
                    Pre-allocate buffers for the output nodes of then- and else- branches.
                    We try to alias them with the corresponding t_out[i] elements, so
                    that we save one copy operation.
                    [TODO]
                    It's not the most optimal buffer allocation.
                    In the ideal case, e.g. when both then- and else- branches
                    are just sequences of element-wise operations that can be executed in-place,
                    we could simply use a single buffer for both then- and else- branches.
                    Here we will use separate buffers, but let's assume we could
                    optimize out such trivial branches at the graph fusion level
                    (especially when we have JIT).
                */
                CV_Assert(node.subgraphs.size() == 2);

                const Graph& thenBranch = *node.subgraphs[0].get();
                const Graph& elseBranch = *node.subgraphs[1].get();
                const vector<int>& thenOutargs = thenBranch.outputs;
                const vector<int>& elseOutargs = elseBranch.outputs;
                CV_Assert(thenOutargs.size() == nouts && elseOutargs.size() == nouts);
                for (i = 0; i < nouts; i++) {
                    int outarg = node.outputs[i];
                    int thenOutarg = thenOutargs[i];
                    int elseOutarg = elseOutargs[i];

                    if (!netimpl->isConst(thenOutarg) && usecounts[thenOutarg] == 1)
                        shareBuffer(outarg, thenOutarg);
                    if (!netimpl->isConst(elseOutarg) && usecounts[elseOutarg] == 1)
                        shareBuffer(outarg, elseOutarg);
                }

                assign(thenBranch);
                assign(elseBranch);

                for (i = 0; i < nouts; i++) {
                    int thenOutarg = thenOutargs[i];
                    int elseOutarg = elseOutargs[i];
                    releaseBuffer(bufidxs[thenOutarg]);
                    releaseBuffer(bufidxs[elseOutarg]);
                }
            } else if (layer->type == "Loop") {
                /*
                 In the case of loop we try to alias t_v_in[i] and t_v_out[i] so that
                 we eliminate some copy operations after each loop iteration.
                 */
                LoopLayer* loop = dynamic_cast<LoopLayer*>(layer);
                CV_Assert(ninps >= 2);
                CV_Assert(loop != 0 && node.subgraphs.size() == 1);
                Graph& body = *node.subgraphs[0].get();
                int trip_count = node.inputs[0];
                int body_ninps = (int)body.inputs.size();
                int body_nouts = (int)body.outputs.size();
                int n_state_vars = ninps - 2;
                int n_accums = body_nouts - n_state_vars - 1;
                CV_Assert(body_ninps == ninps);
                CV_Assert(body_nouts == nouts+1);
                CV_Assert(n_state_vars >= 0 && n_accums >= 0);
                int inp0 = body.inputs[0];
                if (inp0 > 0 && usecounts[inp0] > 0) {
                    CV_Assert(!netimpl->isConst(inp0));
                    if (!netimpl->isConst(trip_count))
                        shareBuffer(trip_count, body.inputs[0]);
                    else
                        bufidxs.at(body.inputs[0]) = getFreeBuffer();
                }

                for (i = -1; i < n_state_vars; i++) {
                    int inparg = body.inputs[i+2];
                    int outarg = body.outputs[i+1];
                    int v_inp = node.inputs[i+2];
                    int v_out = i >= 0 ? node.outputs[i] : 0;
                    if (inparg > 0 && usecounts[inparg] > 0) {
                        CV_Assert(!netimpl->isConst(inparg));
                        if (!netimpl->isConst(v_inp))
                            shareBuffer(v_inp, inparg);
                        else
                            bufidxs[inparg] = getFreeBuffer();
                    }
                    if (!netimpl->isConst(v_out)) {
                        if (!netimpl->isConst(outarg) && usecounts[outarg] == 1)
                            shareBuffer(v_out, outarg);
                    }
                }

                assign(body);
                for (auto body_out: body.outputs)
                    releaseBuffer(bufidxs.at(body_out));
            }

            for (auto outarg: node.outputs) {
                if (usecounts[outarg] == 0)
                    releaseBuffer(bufidxs.at(outarg));
            }
            // let's release inputs in the reverse order to keep the buffer allocation consistent across the network
            for (i = 0; i < ninps; i++) {
                int inparg = node.inputs[ninps-i-1];
                if (bufidxs[inparg] >= 0) {
                    if (--usecounts.at(inparg) == 0)
                        releaseBuffer(bufidxs.at(inparg));
                }
            }
        }
    }
};

void Net2::Impl::assignBuffers()
{
    BufferAllocator buf_allocator(this);
    buf_allocator.assign();
}

CV__DNN_INLINE_NS_END
}}
