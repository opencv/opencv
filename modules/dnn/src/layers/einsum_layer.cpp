// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <inttypes.h>
#include <opencv2/dnn/shape_utils.hpp>
#include "../precomp.hpp"
#include "../ie_ngraph.hpp"
#include "layers_common.hpp"
#include "cpu_kernels/fast_gemm.hpp"

namespace cv
{
namespace dnn
{

static bool IsTransposeReshapeForEinsum(const std::vector<size_t>& perm,
                                        const MatShape& input_dims,
                                        MatShape& new_shape) {
    // As long as the dims with values > 1 stay in the same order, it's a reshape.
    // Example: Shape=(1,1,1024,4096) -> perm=(2,0,3,1).
    size_t last_permuted_axis = 0;
    for (size_t i = 0; i < perm.size(); ++i) {
        if (input_dims[perm[i]] == 1)
            continue;
        if (perm[i] < last_permuted_axis)
            return false;
        last_permuted_axis = perm[i];
    }
    new_shape.assign(input_dims.begin(), input_dims.end());
    for (size_t i = 0; i < perm.size(); ++i) {
        new_shape[i] = input_dims[perm[i]];
    }
    return true;
}


static Mat Transpose(
    const Mat& input,
    const MatShape& input_shape_override,
    const std::vector<size_t> permutation)
{

    int input_rank = input_shape_override.size();
    CV_Assert(input_rank == permutation.size());

    bool reshape = input.dims != input_rank;

    Mat input_reshaped;
    if(reshape){
        input_reshaped = input.reshape(1, input_shape_override.size(), input_shape_override.data());
    }

    MatShape outputDims;
    outputDims.reserve(input_rank);
    for (const auto& dim : permutation)
        outputDims.emplace_back(input_shape_override[dim]);

    Mat output;
    MatShape order(permutation.begin(), permutation.end());

    std::vector<int> order_(order.begin(), order.end());
    cv::transposeND((reshape ? input_reshaped : input), order_, output);
    return output;
}


bool IsTransposeRequired(size_t input_rank, const std::vector<size_t>& permutation) {

    // No transpose required for scalars
    if (input_rank == 0 || permutation.size() == 0){
        return false;
    }
    CV_Assert(input_rank == permutation.size());

    // Weeds out cases where permutation is something like [0, 1, 2] for a 3D input and so on
    bool transpose_required = false;
    for (size_t i = 0; i < input_rank; ++i) {
        if (permutation[i] != i) {
            transpose_required = true;
            break;
        }
    }

  return transpose_required;
}


bool IsTransposeRequiredForDiagonal(int dim1, int dim2, int rank) {
    // If the input is 2D, we don't need a transpose
    if (rank == 2)
        return false;

    // If the two dims are the innermost dims, no transpose is required
    if ((dim1 == rank - 1 && dim2 == rank - 2) ||
        (dim1 == rank - 2 && dim2 == rank - 1))
        return false;

    // Transpose is required
    return true;
}

template <typename T>
Mat DiagonalDataAssignment(Mat input) {

    int rank = input.dims;
    CV_Assert(rank >= 2);
    CV_Assert(input.size[rank - 1] == input.size[rank - 2]);
    MatShape original_dims = shape(input);

    if (rank > 3){
        //reshape to 3D mat
        int collapsed_size = 1;
        for (int i = 0; i < rank - 2; ++i) {
            collapsed_size *= input.size[i];
        }
        std::vector<int> reshaped_dims = {collapsed_size, input.size[rank - 2], input.size[rank - 1]};
        input = input.reshape(1, reshaped_dims);
    }

    // Compute total number of higher-dimensional slices
    int total_slices = input.size[0];

    original_dims[rank - 1] = 1;  // Set the last dimension to 1, as we have extracted the diagonal
    Mat output = Mat(original_dims, input.type());

    int inner_stride = input.size[input.dims - 1];
    auto inputPtr = input.ptr<T>();
    auto outputPtr = output.ptr<T>();
    for (int slice = 0; slice < total_slices; ++slice) {
        for (int j = 0; j < inner_stride; ++j) {
            // Direct memory access using raw pointers
            outputPtr[slice * inner_stride + j] = inputPtr[slice * inner_stride * inner_stride + j * inner_stride + j];
        }
    }
    return output;
}

/* Extract the diagonal elements from the last two dimensions of the tensor.
For instance, given an input_shape of [1, 2, 3, 3]:

The flexibility in this implementation allows one to choose which of the two
last dimensions retains its value, determined by the `preserve_innermost_dim_val` parameter.

When preserve_innermost_dim_val == true:
    The resulting shape is [1, 2, 1, 3], indicating the diagonal has 3 elements,
    and it keeps the dimension value of the innermost dimension.

When preserve_innermost_dim_val == false:
    The resulting shape is [1, 2, 3, 1], indicating the diagonal also has 3 elements,
    but it retains the dimension value of the penultimate dimension. */
Mat DiagonalInnermostDims(const Mat& input, bool preserve_innermost_dim_val) {
    const MatShape input_dims = shape(input);
    int rank = input_dims.size();

    // This is an internal method and we already have finished all validations in the calling method.
    // We proceed without duplicating all validations again here.

    // We have a minimalistic check here to make sure the innermost dims have the same dim value
    // as the calling method may have done a transpose before calling this method
    CV_CheckEQ(input.size[rank - 1], input.size[rank - 2],
        "innermost dims should have the same dim value to parse the diagonal elements");

    MatShape output_dims = input_dims;  // Copy the original dims
    if (preserve_innermost_dim_val) {
        output_dims[rank - 2] = 1;
    } else {
        output_dims[rank - 1] = 1;
    }

    // TODO: hande different types
    Mat output = DiagonalDataAssignment<float>(input);

    if (output_dims != shape(output)){
        CV_Error(Error::StsError, "Output shape does not match with calculated shape");
    }
    return output;
}

Mat Diagonal(const Mat& input, int dim1, int dim2)
{
    const MatShape input_dims = shape(input);
    int rank = input_dims.size();

    if (!(rank >= 2 && dim1 != dim2 && input_dims[dim1] == input_dims[dim2])){
        std::string input_dims_str = std::accumulate(std::next(input_dims.begin()), input_dims.end(), std::to_string(input_dims[0]),
                                                    [](const std::string& a, int b) {
                                                        return a + ' ' + std::to_string(b);
                                                    });
        CV_Error(Error::StsError, cv::format("Cannot parse the diagonal elements along dims %d and %d for input shape %s",dim1, dim2, input_dims_str.c_str()));
    }

    int first_dim = std::min(dim1, dim2);
    int second_dim = std::max(dim1, dim2);

    Mat output;
    bool preserve_innermost_dim_val = false;

    bool is_transpose_required = IsTransposeRequiredForDiagonal(dim1, dim2, rank);
    if (is_transpose_required)
    {
        std::vector<size_t> permutation(rank, 0);
        int first_dim_axis = -1;  // This is the axis eventually occupied by the first_dim

        // If one of the diagonal dimensions is one of the 2 innermost dims, then leave it as such
        // so as to avoid transpose overhead
        if (first_dim == rank - 2) {  // If rank - 2 is occupied by first_dim, keep it there
            permutation[rank - 2] = first_dim;
            first_dim_axis = rank - 2;
        } else {
            if (second_dim != rank - 2) {  // If rank - 2 is not occupied by second_dim, then put first_dim there
                permutation[rank - 2] = first_dim;
                first_dim_axis = rank - 2;
            } else {  // If rank - 2 is occupied by second_dim, then put first_dim in rank - 1
                permutation[rank - 1] = first_dim;
                first_dim_axis = rank - 1;
                preserve_innermost_dim_val = true;  // We always want to preserve the dim value of the first_dim
            }
        }

        // Put the second_dim in the dim not occupied by the first_dim
        if (first_dim_axis != rank - 1) {
            permutation[rank - 1] = second_dim;
        } else {
            permutation[rank - 2] = second_dim;
        }

        size_t iter = 0;
        for (int i = 0; i < rank; ++i) {
            if (i != first_dim && i != second_dim) {
                permutation[iter++] = i;
            }
        }

        // Permutate the input so that the dims from which we need the diagonal forms the innermost dims
        Mat transposed = Transpose(input, input_dims, permutation);

        // Parse the diagonal from the innermost dims
        output = DiagonalInnermostDims(transposed, preserve_innermost_dim_val);

        // Swap back the dimensions to the original axes ordering using a "reverse permutation"
        // Find the "reverse" permutation
        iter = 0;
        std::vector<size_t> reverse_permutation(rank, 0);
        for (const auto& perm : permutation) {
            reverse_permutation[perm] = iter++;
        }

        // Permutate using the reverse permutation to get back the original axes ordering
        // (Pass in CPU Transpose function here as this Diagonal method will only be used for CPU based diagonal parsing)
        output = Transpose(output, shape(output), reverse_permutation);
    } else {
        // No transposing required
        output = DiagonalInnermostDims(input, preserve_innermost_dim_val);
    }

    // Make copy of the output dims
    MatShape output_dims = shape(output);

    // Unsqueeze the reduced dim
    auto iter = output_dims.begin() + second_dim;
    output_dims.erase(iter);
    output = output.reshape(1, output_dims);
    return output;
}

/**
 * Returns the index associated with the input character.
 * - Returns a value between 0 and 25 for inputs in the range 'a' to 'z'.
 * - Returns a value between 26 and 51 for inputs in the range 'A' to 'Z'.
 * - Returns -1 for invalid input that is not in the range 'a' to 'z' or 'A' to 'Z' (the caller should handle the returned result accordingly).
 */
int letterToIndex(const char ch) {
    if (ch >= 'a' && ch <= 'z') {
        return static_cast<int>(ch) - 'a';
    }

    if (ch >= 'A' && ch <= 'Z') {
        return static_cast<int>('z') + static_cast<int>(ch) - 'A';
    }
    // invalid character - return error value
    return -1;
}

// Implementation of the Einsum layer is heavily influenced by Onnxruntime at the time of writing.
// Main logic is borrowed from onnxrutime:
// https://github.com/microsoft/onnxruntime/blob/eaea34f8e29df9fb21fab675a3a895084407f306/onnxruntime/core/providers/cpu/math/einsum_utils/einsum_compute_preprocessor.cc#L8
class LayerEinsumImpl CV_FINAL : public EinsumLayer
{
private:
    Ptr<ReduceLayer> reduce;
public:
    // Number of inputs and outputs of the layer
    int numInputs;

    // inputShapes;
    std::vector<MatShape> einsumInpShapes;

    // Preprocessed inputs
    std::vector<Mat> preProcessedInputs;

    // This is container for preporcessed inputs
    std::vector<MatShape> homogenizedInputDims;

    // Collect outpus dimentions
    MatShape einsumOutDims; // vector to store output dimentions

    // These hold equation subring, left hand side and right it of
    String lhs_eq, rhs_eq, equation;

    // Holds token from left hand side of the equation
    std::vector<String> lhs_eq_tokens;

    // Idicates if equation substring is defined in explit way such as "ij, jk->ik"
    // as opposed to "ij->"
    bool explicitEquation = false;

    // Stores the subscript indices for each input in the equation
    std::vector<std::vector<int>> inputSubscriptIndices;

    // Keeps track of the input index of the last input that had the subscript label
    // If the value is `-1`, it means the subscript label was never encountered or it appears in the output
    std::vector<int> subscriptIndicesToLastInput;

    // Holds the dimension value of the index corresponding to the subscript label
    // `-1` indicates that the corresponding label was not encountered at all
    std::vector<int> subscriptIndicesToDimValue;

    // Index corresponding to each output dim corresponding to each subscript index
    // A value of -1 means the corresponding subscript index is not found in the output
    std::vector<int> subscriptIndicesToOutputIndices;

    // Hold max number of alphabetic numbers
    static const size_t numOfLetters = 52;

    // Stores the count corresponding to each letter encountered
    // A value of `0` indicates that the corresponding letter hasn't been seen at all
    std::array<int, numOfLetters> letter2count;

    // Hold the assigned index corresponding to the letter seen
    // `-1` means the corresponding letter wasn't seen at all
    std::array<int, numOfLetters> letter2index;

    // Represents the count of unique subscript labels (subscript indices)
    // Example 1: For the equation 'ij, jk -> ik', num_subscript_indices_ = 3 (i, j, k)
    // Example 2: For the equation '...ij', 'jk' -> '...ik',
    // num_subscript_indices_ = 3 (i, j, k) + number of dimensions specified by an ellipsis (across all inputs)
    int numLetterIndices = 0;

    // The number of dimensions that are encompassed by an "ellipsis" - "...".
    size_t numOfEllipsisDims = 0;

    // Backend for fastgemm
    FastGemmOpt opt;

    mutable bool outputShapeComputed;
    mutable MatShape cachedOutputShape;

    void parseEquation(String equation);
    void processEquation(const std::vector<MatShape>& inputs);
    void processBroadcastedDims();
    void validateOutputSubscript();
    void calculateOutputShape();
    void preProcessInputs(InputArrayOfArrays& inputs);
    Mat reduceSum(Mat& src, MatShape& reduceAxis);
    Mat FinalizeOutput(const Mat& candidateOuput, const MatShape& ordered_subscript_indices_in_candidate);
    Mat pairwiseOperandProcess(
        const Mat& left,
        const MatShape& leftShapeOverride,
        const Mat& right,
        const MatShape& rightShapeOverride,
        const MatShape& reduceDims,
        bool isFinalPair
    );
    Mat batchwiseMatMul(
        const Mat& input1,
        const MatShape& input1ShapeOverride,
        const Mat& input2,
        const MatShape& input2ShapeOverride
    );

    void computeOutputShape(const std::vector<MatShape>& inputs) const {
        if (!outputShapeComputed) {
            // Copy of the existing computation logic
            const_cast<LayerEinsumImpl*>(this)->processEquation(inputs);
            const_cast<LayerEinsumImpl*>(this)->processBroadcastedDims();
            const_cast<LayerEinsumImpl*>(this)->validateOutputSubscript();
            const_cast<LayerEinsumImpl*>(this)->calculateOutputShape();

            cachedOutputShape = einsumOutDims;
            outputShapeComputed = true;
        }
    }

    // constructor
    LayerEinsumImpl(const LayerParams& params)
        : outputShapeComputed(false)
    {
        setParamsFrom(params);
        equation = params.get<String>("equation");
        opt.init();

        // We allocate space for 10 values as a precaution,
        // assuming that we won't encounter any input with a rank greater than 10.
        // In such cases, the value of num_subscript_indices_ would be greater than 10.
        subscriptIndicesToLastInput.reserve(10);
        subscriptIndicesToDimValue.reserve(10);

        // fill in vectors to avoid getting random numbers
        letter2count.fill(0);
        letter2index.fill(-1);

        // parser equation and extract tokens from the equation
        // save token to lhs_eq_tokens variable
        parseEquation(equation); // TODO: return lhs_eq_tokens
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
    }

    // getMeoryShapes
    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_UNUSED(requiredOutputs);
        CV_UNUSED(internals);

        // check if input einsumInputShapes is empty
        if (einsumInpShapes.empty()) {
            outputShapeComputed = false;
        } else {
            // check weather shapes in inputs are compatible with shapes in einsumInpShapes
            for (int i = 0; i < inputs.size(); i++) {
                if (inputs[i] != einsumInpShapes[i]) {
                    outputShapeComputed = false;
                    break;
                }
            }
        }

        computeOutputShape(inputs);

        outputs.clear();
        outputs.emplace_back(cachedOutputShape);
        return true;
    } // getMemoryShape

    // forward
    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16F)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        // homogenize inputs
        preProcessInputs(inputs_arr);

        std::vector<cv::Mat> rawInputs, outputs;
        inputs_arr.getMatVector(rawInputs);
        outputs_arr.getMatVector(outputs);
        Mat result;

        // Pre-process the first input so as to reduce any dims that only it has
        {
            MatShape reducedDims;
            MatShape preservedDims;
            MatShape preservedShape;

            reducedDims.reserve(numLetterIndices);    // num_subscript_labels is the upper bound. No harm in over-reserving.
            preservedDims.reserve(numLetterIndices);  // num_subscript_labels is the upper bound. No harm in over-reserving.

            for (size_t i = 0; i < numLetterIndices; ++i) {
                if (subscriptIndicesToLastInput[i] == 0) {
                    reducedDims.push_back(i);
                } else {
                    preservedDims.push_back(i);
                }
            }

            // Reduce the dims that are last seen in the first input alone
            if (reducedDims.size() != 0)
            {
                result = reduceSum((!preProcessedInputs[0].empty() ? preProcessedInputs[0] : rawInputs[0]), reducedDims);
            } else {
                // Check if there is a pre-processed version of this input
                // If so assign it to result
                if (!preProcessedInputs.empty() && !preProcessedInputs[0].empty())
                {
                    result = preProcessedInputs[0];
                }
            }

            // Finalize the output at this stage if num_inputs == 1
            if (numInputs == 1) {
                // Finalize the output by applying any transpose required to get
                // it to the required output ordering and move it to the op's output
                result = FinalizeOutput(!result.empty() ? result : rawInputs[0], preservedDims);
            }
        }


        // Process the operands in a pair-wise fashion
        {
            bool isFinalPair = false;
            // Keep processing each input pair-wise
            for (int input = 1; input < numInputs; ++input) {
                MatShape reducedDims;
                reducedDims.reserve(numLetterIndices);  // num_subscript_labels is the upper bound. No harm in over-reserving by a small margin.
                for (int dim = 0; dim < numLetterIndices; ++dim)
                {
                    if (subscriptIndicesToLastInput[dim] == input)
                    {
                        // This is the last input we are seeing this dimension (and it doesn't occur in the output), so reduce along the dimension
                        reducedDims.push_back(dim);
                    }
                }

                if (input == numInputs - 1)
                    isFinalPair = true;

                // create temporary variable
                MatShape tmpResult;
                for (int i = 0; i < result.size.dims(); i++)
                    tmpResult.emplace_back(result.size[i]);


                // Use either the preprocessed inputs (if it is available) or the corresponding raw inputs
                result = pairwiseOperandProcess(!result.empty() ? result : rawInputs[0],
                                                !result.empty() ? tmpResult : homogenizedInputDims[0],
                                                (!preProcessedInputs.empty() && !preProcessedInputs[input].empty()) ? preProcessedInputs[input] : rawInputs[input],
                                                homogenizedInputDims[input],
                                                reducedDims,
                                                isFinalPair);
            }
        }

        // check of product of output dimentions and computed output dimentions match
        size_t reqProd = std::accumulate(einsumOutDims.begin(), einsumOutDims.end(), 1, std::multiplies<int>());
        MatShape realOutputDims = shape(result);
        size_t realProd = std::accumulate(realOutputDims.begin(), realOutputDims.end(), 1, std::multiplies<int>());

        CV_CheckEQ(reqProd, realProd, "Real output can not be shaped in to required output");

        // reduce dimentions
        result = result.reshape(1, einsumOutDims.size(), einsumOutDims.data());
        result.copyTo(outputs[0]);
    } // forward

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >&,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE {
        ov::OutputVector inputs(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i) {
            inputs[i] = nodes[i].dynamicCast<InfEngineNgraphNode>()->node;
        }
        auto einsum = std::make_shared<ov::op::v7::Einsum>(inputs, equation);
        return new InfEngineNgraphNode(einsum);
    }
#endif // HAVE_DNN_NGRAPH

}; // EinsumClass

Mat LayerEinsumImpl::reduceSum(Mat& src, MatShape& reduceAxis)
{
    // initialize ReduceLayer
    LayerParams lp;
    lp.set("reduce", "SUM");
    int num_axes = reduceAxis.size();
    lp.set("axes", DictValue::arrayInt(&reduceAxis[0] , num_axes));
    reduce = ReduceLayer::create(lp);

    // Compute output shapes
    std::vector<MatShape> inputShapes{shape(src)};
    std::vector<MatShape> outputShapes, internalShapes;
    reduce->getMemoryShapes(inputShapes, 1, outputShapes, internalShapes);

    Mat output(outputShapes[0], CV_32F);

    std::vector<Mat> inputs;
    std::vector<Mat> outputs;
    std::vector<Mat> internals;
    inputs.emplace_back(src);
    outputs.emplace_back(output);

    reduce->forward(inputs, outputs, internals);
    return outputs[0];
}

void LayerEinsumImpl::preProcessInputs(InputArrayOfArrays& inputs_arr)
{
    std::vector<cv::Mat> inputs;
    inputs_arr.getMatVector(inputs);

    preProcessedInputs.reserve(inputs.size());
    homogenizedInputDims.reserve(inputs.size());

    int inputIter = 0;
    for(const Mat& input : inputs)
    {
        Mat preprocessed;

        // variable to hold processed version of the original input
        MatShape input_dims = shape(input);
        if (input_dims.empty()){
            homogenizedInputDims.emplace_back(MatShape(numLetterIndices, 1));
            ++inputIter;
            continue;
        }

        const auto& currSubscriptIndices = inputSubscriptIndices[inputIter];

        // There should be subscript index (subscript label) for each dim of the input
        CV_CheckEQ(input_dims.size(), currSubscriptIndices.size(),
            "Rank of the input must match number of subscript labels corresponding to the input");

        std::vector<int> subscriptIndicesToInputIndex(numLetterIndices, -1);
        // this will hold input dims after reordering so that all inputs have
        // same axes order
        MatShape homogenizedInputDims_(numLetterIndices, 1);

        int dimIndexInIreprocessedInput = 0;
        int dimIndexInOriginalInput = 0;

        for (const auto& subscriptIndex : currSubscriptIndices)
        {
            if(subscriptIndicesToInputIndex[subscriptIndex] == -1){
                subscriptIndicesToInputIndex[subscriptIndex] = dimIndexInIreprocessedInput++;
                homogenizedInputDims_[subscriptIndex] = input_dims[dimIndexInOriginalInput];
            } else {
                // Call diagonal
                preprocessed = Diagonal(
                    !preprocessed.empty() ? preprocessed : inputs[inputIter],
                    subscriptIndicesToInputIndex[subscriptIndex],
                    dimIndexInIreprocessedInput);
            }
            ++dimIndexInOriginalInput;
        }

        std::vector<size_t> permutation;
        for(auto& d : subscriptIndicesToInputIndex)
        {
            if (d != -1)
                permutation.emplace_back(d);
        }

        if (IsTransposeRequired(
            !preprocessed.empty() ? preprocessed.size.dims() : inputs[inputIter].size.dims(),
            permutation))
        {
            // call transpose
            preprocessed = Transpose(
                !preprocessed.empty() ? preprocessed : inputs[inputIter],
                !preprocessed.empty() ? shape(preprocessed) : shape(inputs[inputIter]),
                permutation);
        }

        if (!preprocessed.empty())
        {
            preprocessed = preprocessed.reshape(1, homogenizedInputDims_.size(), homogenizedInputDims_.data());
        }

        preProcessedInputs.emplace_back(preprocessed);
        homogenizedInputDims.emplace_back(homogenizedInputDims_);
        ++inputIter;
    }
}

void LayerEinsumImpl::parseEquation(String equation)
{
    // remove white spaces in the copy
    equation.erase(std::remove_if(equation.begin(), equation.end(), ::isspace), equation.end());

    // check if '->' - the output subscript label is present in the equation;
    std::size_t arrow_idx = equation.find("->");
    if (arrow_idx != std::string::npos)
    {
        // split left and righ hand sides of the equation
        lhs_eq = equation.substr(0, arrow_idx);
        rhs_eq = equation.substr(arrow_idx + 2);
        explicitEquation = true;
    } else {
        lhs_eq = equation;
    }

    // split lhs_eq by ',' - comma and put all created token - splits
    // into lhs_eq_tokens vector
    // the implementation does not ignore empty tokens and trailing comma
    size_t start = 0;
    while(start < lhs_eq.size())
    {
        size_t comma = lhs_eq.find(',', start);
        if (comma != std::string::npos)
        {
            std::string token = lhs_eq.substr(start, comma-start);
            lhs_eq_tokens.push_back(token);
            start = comma+1;
        }
        else
        {
            std::string token = lhs_eq.substr(start);
            lhs_eq_tokens.push_back(token);
            start = lhs_eq.size()+1;
        }
    }

    // trailing comma without token
    if (lhs_eq[lhs_eq.size()-1] == ',')
        lhs_eq_tokens.push_back(std::string());

}


void LayerEinsumImpl::calculateOutputShape()
{
    // Traverse through each of the subscript labels within the output subscript.
    bool middleOfEllipsis = false;
    int ellipsisCharCount = 0;

    subscriptIndicesToOutputIndices.resize(numLetterIndices, -1);

    std::array<int, numOfLetters> outputLetterToCount;
    outputLetterToCount.fill(0);

    int outputDimCounter = 0;
    for (auto letter : rhs_eq)
    {
        if(letter == '.')
        {
            middleOfEllipsis = true;
            // Make sure there aren't more than 3 '.'s in the current subscript
            if (++ellipsisCharCount > 3) {
                CV_Error(Error::StsError, "Found a '.' not part of an ellipsis in the output subscript provided");
            }

            if (ellipsisCharCount == 3) {  // Ellipsis is complete. Process it.
                middleOfEllipsis = false;
                for (size_t i = 0; i < numOfEllipsisDims; ++i) {
                    einsumOutDims.emplace_back(subscriptIndicesToDimValue[i]);
                    // The ellipsis is seen in the output and hence the corresponding dims are to not be reduced
                    subscriptIndicesToLastInput[i] = -1;
                    subscriptIndicesToOutputIndices[i] = outputDimCounter++;
                }
            }
        } else {
            CV_CheckEQ(middleOfEllipsis, false,
                "Encountered '.' character that is not part of output subscript");

            auto letterIndex = letterToIndex(letter);

            CV_CheckNE(letterIndex, -1,
                "The only permissible subscript labels are lowercase letters (a-z) and uppercase letters (A-Z).");
            CV_CheckEQ(outputLetterToCount[letterIndex], 0,
                "Output subscript constains repeated letters");

            ++outputLetterToCount[letterIndex];
            auto mappedIndex = letter2index[letterIndex];

            CV_CheckNE(mappedIndex, -1,
                "Output subscript has letters that were not encountered in the inputs");

            // Push output dimention
            // Einsum layer only has one output vector
            einsumOutDims.emplace_back(subscriptIndicesToDimValue[mappedIndex]);

            // Reset the last input index for this subscript label
            // given that it is seen in the output and hence can't be reduced
            subscriptIndicesToLastInput[mappedIndex] = -1;
            subscriptIndicesToOutputIndices[mappedIndex] = outputDimCounter++;
        }
    }
    if (rhs_eq.empty()) {
        einsumOutDims = MatShape(0, 0); // handle scalar output case
    }
}

void LayerEinsumImpl::validateOutputSubscript()
{
    // The explicit form requires no operation, as the output
    // would have already been parsed during the input parsing process.
    if(explicitEquation)
    {
        // Ensure that the provided explicit equation includes an ellipsis if the input contains ellipses.
        if(numOfEllipsisDims > 0)
        {
            if(rhs_eq.find("...") == std::string::npos)
            {
                CV_Error(Error::StsError,
                "Provided output subscript does not include ellipsis while Inputs subscrits constain ellipsis");
            }
        }
    }
}

void LayerEinsumImpl::processBroadcastedDims()
{
    // Only compute this function if ellipsis "..." was found in the equation
    if (numOfEllipsisDims > 0)
    {
        // extend the number of subscript labels to include each ellipsis dim as
        // theoretically each ellipsis dim does correspond to a "virtual" subscript label
        numLetterIndices += numOfEllipsisDims;

        // We are going to assign the broadcasted dims outermost subscript indices (i.e.) 0 -> numOfEllipsisDims - 1
        // as most likely bradcasted dims will be batch dimensions (i.e.) outermost dimensions and hence we don't have to pay
        // transposing while "homogenizing" the input

        // Hence offset all subscript indices by numOfEllipsisDims
        for (size_t i = 0; i < numOfLetters; ++i){
            if (letter2count[i] != -1){
                letter2index[i] += numOfEllipsisDims;
            }
        }

        std::vector<int> tempIndex2LastInput(numLetterIndices, -1);
        for (int i = 0; i < subscriptIndicesToLastInput.size(); ++i){
            tempIndex2LastInput[i + numOfEllipsisDims] = subscriptIndicesToLastInput[i];
        }
        subscriptIndicesToLastInput = std::move(tempIndex2LastInput);

        std::vector<int> tempIndexToDimValue(numLetterIndices, -1);
        for (int i = 0; i < subscriptIndicesToDimValue.size(); ++i){
            tempIndexToDimValue[i + numOfEllipsisDims] = subscriptIndicesToDimValue[i];
        }
        subscriptIndicesToDimValue = std::move(tempIndexToDimValue);

        for (size_t i = 0; i < inputSubscriptIndices.size(); ++i)
        {
            auto& currentInputDimIndicesToSubscriptIndices = inputSubscriptIndices[i];
            std::vector<int> tempCurrentInputDimIndicesToSubscriptIndices;
            tempCurrentInputDimIndicesToSubscriptIndices.reserve(currentInputDimIndicesToSubscriptIndices.size());

            // make sure it is correct
            const auto& dims = einsumInpShapes[i];
            auto rank = dims.size();

            size_t dimIter = 0;
            size_t numBroadcastedIndices = 0;
            while (dimIter < currentInputDimIndicesToSubscriptIndices.size())
            {
                auto value = currentInputDimIndicesToSubscriptIndices[dimIter];
                if (value == numOfLetters)
                {  // This is a broadcasted dim
                    // Shouldn't hit this error - just a sanity check
                    CV_Assert(numBroadcastedIndices < numOfEllipsisDims);
                    tempCurrentInputDimIndicesToSubscriptIndices.push_back(static_cast<int>(numBroadcastedIndices));
                    subscriptIndicesToLastInput[numBroadcastedIndices] = i;

                    // This is the first time we are seeing this broadcasted dim
                    if (subscriptIndicesToDimValue[numBroadcastedIndices] == -1)
                    {
                        subscriptIndicesToDimValue[numBroadcastedIndices] = dims[dimIter];
                    } else {  // We have seen this broadcasted dim before
                        // Check if the previous value is equal to the current value
                        if (subscriptIndicesToDimValue[numBroadcastedIndices] != dims[dimIter])
                        {
                            // If they are not equal, one of them needs to be 1
                            if (subscriptIndicesToDimValue[numBroadcastedIndices] == 1)
                            {
                                subscriptIndicesToDimValue[numBroadcastedIndices] = dims[dimIter];
                            } else {
                                CV_CheckEQ(dims[dimIter], 1, "The broadcasted dimensions of the inputs are incompatible");
                            }
                        }
                    }
                    ++numBroadcastedIndices;
                } else {  // This is a regular dim - offset it by number of broadcasted dims
                    tempCurrentInputDimIndicesToSubscriptIndices.push_back(value + static_cast<int>(numOfEllipsisDims));
                }
                ++dimIter;
            }
            // Shouldn't hit this error - just a sanity check
            CV_Assert(dimIter == rank);
            currentInputDimIndicesToSubscriptIndices = std::move(tempCurrentInputDimIndicesToSubscriptIndices);
        }
    }
}



void LayerEinsumImpl::processEquation(const std::vector<MatShape>& inputs)
{

    // fill in the einsumInpShapes
    for (const auto& input : inputs) {
        einsumInpShapes.emplace_back(input);
    }


    numInputs = inputs.size();
    inputSubscriptIndices.reserve(numInputs);
    // Check if number of tokens in equal to number of inputs.
    // For install "ij, jk -> ik" needs to have 2 inputs tensors
    int num_input_tensors = inputs.size();
    if (lhs_eq_tokens.empty() || (lhs_eq == ",") ) {
        inputSubscriptIndices.resize(numInputs);
        return;
    }
    // if we have only one token and two inputs lets skip the check
    if (lhs_eq_tokens.size() > 1)
        CV_CheckEQ(static_cast<int>(lhs_eq_tokens.size()), num_input_tensors,
            "Number of input tensors does not match the number of subscripts in the input equation");

    int inputIdx = 0;
    for (const auto& token : lhs_eq_tokens)
    {
        const MatShape shape = inputs[inputIdx];
        size_t rank = shape.size();
        size_t dim_count = 0;

        std::vector<int> currTokenIndices;
        currTokenIndices.reserve(rank);

        // Variable to deal with "ellipsis" - '...' in the input
        bool middleOfellipsis = false;
        int ellipsisCharCount = 0;
        for (auto letter : token)
        {
            if (letter == '.')
            {
                middleOfellipsis = true;

                // there should not be more than 3 '.'s in the current subscript
                if (++ellipsisCharCount > 3)
                {
                    CV_Error(Error::StsError, cv::format("Found a '.' not part of an ellipsis in input: %d", inputIdx));
                }

                // We have seen all 3 '.'s. We can safely process the ellipsis now.
                if (ellipsisCharCount == 3)
                {
                    middleOfellipsis = false;

                    // Example for the following line of code
                    // Subscript "...ij" for an input of rank 6
                    // numOfEllipsisDims = 6 - 5 + 3 = 4
                    int currentNumOfEllipsisDims = static_cast<int>(rank) - token.length() + 3;
                    CV_CheckGE(currentNumOfEllipsisDims, 0,
                        "Einsum subscripts string contains too many subscript labels when compared to the rank of the input");

                    // Theoretically, currentNumOfEllipsisDims could be 0
                    // Example: For an input of rank 2 paired with a subscript "...ij"
                    if (currentNumOfEllipsisDims != 0)
                    {
                        // We have seen a ellipsis before - make sure ranks align as per the ONNX spec -
                        // "Ellipsis must indicate a fixed number of dimensions."
                        if (numOfEllipsisDims != 0){
                            CV_CheckEQ(numOfEllipsisDims, static_cast<size_t>(currentNumOfEllipsisDims),
                                "Ellipsis must indicate a fixed number of dimensions across all inputs");
                        } else {
                            numOfEllipsisDims = static_cast<size_t>(currentNumOfEllipsisDims);
                        }

                        // We reserve 'numOfLetters' for broadcasted dims as we only allow 'a' - 'z'
                        // and 'A' - 'Z' (0 - 51) for non-broadcasted dims.
                        // We will assign appropriate indices (based on number of dimensions the ellipsis corresponds to)
                        // during broadcasting related post-processing.
                        for (size_t i = 0; i < numOfEllipsisDims; ++i){
                            currTokenIndices.push_back(numOfLetters);
                        }

                        // Offset 'dim_count' by number of dimensions the ellipsis corresponds to
                        dim_count += numOfEllipsisDims;
                    }
                }
            } else {
                if (middleOfellipsis){
                    CV_Error(Error::StsAssert,
                    cv::format(
                        "Encountered '.' character that is not part of an ellipsis in the input: [%d]",
                        inputIdx));
                }

                int letterIdx = letterToIndex(letter);
                CV_CheckNE(letterIdx, -1,
                    "The only permissible subscript labels are lowercase letters (a-z) and uppercase letters (A-Z).");

                int dimValue = shape[dim_count];

                // The subscript label was not found in the global subscript label array
                // Therefore, it is added to both the local and global subscript arrays
                if(letter2count[letterIdx] == 0){
                    letter2index[letterIdx] = numLetterIndices++;
                    subscriptIndicesToDimValue.push_back(dimValue);
                    subscriptIndicesToLastInput.push_back(inputIdx);

                } else {
                    // This letter has been seen in at least one other operand's subscript
                    // It must be equal unless one of them is a 1 (Numpy allows this)
                    auto mappedIndx = letter2index[letterIdx];
                    subscriptIndicesToLastInput[mappedIndx] = inputIdx;

                    if (subscriptIndicesToDimValue[mappedIndx] != dimValue) {
                        if (dimValue != 1) {
                            CV_Error(Error::StsError, cv::format("Einsum operands can not be broadcasted."
                                                                "Check input shapes/equation passed."
                                                                "Input shape of operand [%d]", inputIdx) +
                                                    cv::format(" is incompatible in the dimention [%zu].", static_cast<size_t>(dim_count)));
                        }
                    }
                }
                ++letter2count[letterIdx];
                currTokenIndices.push_back(letter2index[letterIdx]);

                CV_CheckLE(++dim_count, rank,
                    "The Einsum subscripts string has an excessive number of subscript labels compared to the rank of the input.");
            }
        }

        // When no broadcasting is requested, the number of subscript labels (dim_counter) should match the input's rank.
        CV_Assert(!(numOfEllipsisDims == 0 && dim_count != rank)
            && "The Einsum subscripts string does not contain required amount of subscript labels and no ellipsis is provided in the input.");

        inputSubscriptIndices.emplace_back(std::move(currTokenIndices));
        ++inputIdx;
    }
}

Mat LayerEinsumImpl::FinalizeOutput(
    const Mat& candidateOutput,
    const MatShape& ordered_subscript_indices_in_candidate)
{
    const std::vector<int>& subscript_indices_to_output_indices = subscriptIndicesToOutputIndices;
    const auto output_dims = einsumOutDims;

    const auto output_rank = output_dims.size();

    // MatShape output_shape = output_dims;
    // CV_CheckEQ((int) candidateOutput.dims,  (int) output_shape.size(),
    //           "Einsum op: The candidate output cannot be reshaped into the op's output");

    const MatShape candidate_output_dims = MatShape(candidateOutput.size.p, candidateOutput.size.p + candidateOutput.dims);
    const int candidate_output_rank = candidate_output_dims.size();

    // This vector holds the shape of the candidate_output after removing the dims that have
    // been reduced in the final output
    MatShape candidate_output_shape_without_reduced_dims;
    candidate_output_shape_without_reduced_dims.reserve(candidate_output_rank);  // reserve upper bound

    // Identify the permutation required by the op's output
    std::vector<size_t> output_permutation;
    output_permutation.resize(output_rank, 0);
    size_t output_iter = 0;


    for (size_t iter = 0, end = ordered_subscript_indices_in_candidate.size(); iter < end; ++iter)
    {
        auto output_index = subscript_indices_to_output_indices[ordered_subscript_indices_in_candidate[iter]];

        // If output_index is -1, then this dimension does not show up in the op's output and has been reduced along the way
        if (output_index != -1)
        {
            output_permutation[output_index] = output_iter++;
            candidate_output_shape_without_reduced_dims.push_back(candidate_output_dims[iter]);
        } else {
            // This dim doesn't show up in the op's output and hence we check if the dim has been reduced in the candidate output
            CV_CheckEQ(candidate_output_dims[iter], 1,
            "Not all dimensions to be reduced have been reduced in the candidate output. Candidate output dims: "); //%d", candidateOutput.size));
        }
    }

    // Transpose to the required final output order
    // (Identify no-op transposes and prevent triggering the transpose)

    if (IsTransposeRequired(candidate_output_shape_without_reduced_dims.size(), output_permutation))
    {
        auto candidate_output_transposed = Transpose(
                                            candidateOutput,
                                            candidate_output_shape_without_reduced_dims,
                                            output_permutation);
        return candidate_output_transposed;
    }
    return candidateOutput;
}

Mat LayerEinsumImpl::pairwiseOperandProcess(
    const Mat& left,
    const MatShape& leftShapeOverride,
    const Mat& right,
    const MatShape& rightShapeOverride,
    const MatShape& reduceDims,
    bool isFinalPair
)
{
    size_t matDimSize = left.total();
    size_t overrideDimSize = total(leftShapeOverride);

    CV_CheckEQ(matDimSize, overrideDimSize, "Override dims are not compatible with left tensor shape");

    matDimSize = right.total();
    overrideDimSize = total(rightShapeOverride);

    CV_CheckEQ(matDimSize, overrideDimSize, "Override dims are not compatible with right tensor shape");

    // Make copy as this may be overridden downstream
    const auto& leftDims = leftShapeOverride;
    const auto& rightDims = rightShapeOverride;

    int leftRank = static_cast<int>(leftDims.size());
    int rightRank = static_cast<int>(rightDims.size());

    Mat currentLeft;
    Mat currentRight;

    CV_CheckEQ(leftRank, rightRank, "Raks of pair-wise operands must be equal");

    // Following vectors hold:
    // lro: dim indices that are present in left, right, and reduce_dims
    // lo: dim indices that are present in left and reduce_dims
    // ro: dim indices that are present in right and reduce_dims
    std::vector<size_t> lro;
    lro.reserve(5);  // Reserve an arbitrary amount of space for this vector (not bound to see a tensor of rank > kTensorShapeSmallBufferElementsSize)

    std::vector<size_t> lo;
    lo.reserve(5);  // Reserve an arbitrary amount of space for this vector (not bound to see a tensor of rank > kTensorShapeSmallBufferElementsSize)

    std::vector<size_t> ro;
    ro.reserve(5);  // Reserve an arbitrary amount of space for this vector (not bound to see a tensor of rank > kTensorShapeSmallBufferElementsSize)

    // Maintain sizes to create reshaped "views"
    int lro_size = 1;
    int lo_size = 1;
    int ro_size = 1;
    int reduced_size = 1;

    size_t reduceDimsIter = 0;
    size_t reduceDimsSize = reduceDims.size();

    for (int i = 0; i < leftRank; ++i)
    {
        int leftDim = leftDims[i];
        int rightDim = rightDims[i];

        bool hasLeftDim = leftDim > 1;    // non-trivial dimension (dim_value != 1)
        bool hasRightDim = rightDim > 1;  // non-trivial dimension (dim_value != 1)

        if (reduceDimsIter < reduceDimsSize && reduceDims[reduceDimsIter] == i)
        {
            // This dimension is to be reduced after this pair-wise operation
            ++reduceDimsIter;
            if (hasLeftDim && hasRightDim){
                // Both inputs have non-trivial dim values along this dimension
                // Both the left and right operands have non-trivial dimension value along this axis
                CV_CheckEQ(leftDim, rightDim, "Einsum op: Input dimensions must be equal along an axis to be reduced across all inputs");
                reduced_size *= leftDim;

            } else if (hasLeftDim){
                // if the dim to be reduced is only in one of left and right, we can reduce right away
                Mat tensorToReduce = !currentLeft.empty() ? currentLeft : left;
                MatShape shapeToReduce = !currentLeft.empty() ? shape(currentLeft) : leftDims;
                currentLeft = reduceSum(tensorToReduce, shapeToReduce);

            } else if (hasRightDim){
                Mat tensorToReduce = !currentRight.empty() ? currentRight : right;
                MatShape shapeToReduce = !currentRight.empty() ? shape(currentRight) : rightDims;
                currentLeft = reduceSum(tensorToReduce, shapeToReduce);
            }

        } else {
            // This dimension is not reduced (i.e.) it appears in the output after processing these 2 operands
            // Both the left and right operands have non-trivial dimension value along this axis
            // They must be equal
            if (hasLeftDim && hasRightDim){
                CV_CheckEQ(leftDim, rightDim, "Input shapes do not align");
                lro.push_back(i);
                lro_size *= leftDim;

            } else if (hasLeftDim) {
                // The left operand has non-trivial dimension value
                lo.push_back(i);
                lo_size *= leftDim;

            } else {
                // The right operand may or may not have non-trivial dim value
                // If it has trivial dim value (1),
                // it will just form a trailing dimension for the right operand
                ro.push_back(i);
                ro_size *= rightDim;
            }
        }
    }


    // Permutate the left operand so that the axes order go like this: [lro, lo, reduce_dims, ro]
    MatShape reshaped_dims;
    std::vector<size_t> left_permutation;
    left_permutation.reserve(lro.size() + lo.size() + reduceDims.size() + ro.size());
    left_permutation.insert(left_permutation.end(), lro.begin(), lro.end());
    left_permutation.insert(left_permutation.end(), lo.begin(), lo.end());
    //  left_permutation.insert(left_permutation.end(), reduce_dims.begin(), reduce_dims.end());

    for (auto& a : reduceDims)
    {
        left_permutation.push_back(a);
    }
    left_permutation.insert(left_permutation.end(), ro.begin(), ro.end());

    if (IsTransposeRequired(!currentLeft.empty() ? currentLeft.dims : leftDims.size(),
                                        left_permutation))
    {
        if (!currentLeft.empty() && IsTransposeReshapeForEinsum(left_permutation,
                                                                shape(currentLeft),
                                                                reshaped_dims))
        {
            // This can be done because curent_* tensors (if they exist) and output tensors are
            // intermediate tensors and cannot be input tensors to the Einsum node itself
            // (which are immutable).
            currentLeft = currentLeft.reshape(1, reshaped_dims.size(), reshaped_dims.data());
        } else {
            // Covered by ExplicitEinsumAsTensorContraction, DiagonalWithMatmul, ...
            currentLeft = Transpose(!currentLeft.empty() ? currentLeft: left,
                                    !currentLeft.empty() ? shape(currentLeft) : leftDims,
                                    left_permutation);
        }
    }

    // Permutate the right operand so that the axes order go like this: [lro, reduce_dims, ro, lo]
    std::vector<size_t> right_permutation;
    right_permutation.reserve(lro.size() + lo.size() + reduceDims.size() + ro.size());
    right_permutation.insert(right_permutation.end(), lro.begin(), lro.end());
    //  right_permutation.insert(right_permutation.end(), reduce_dims.begin(), reduce_dims.end());
    for (auto& a : reduceDims) {
        right_permutation.push_back(a);
    }
    right_permutation.insert(right_permutation.end(), ro.begin(), ro.end());
    right_permutation.insert(right_permutation.end(), lo.begin(), lo.end());

    if (IsTransposeRequired(!currentRight.empty() ? currentRight.dims: rightDims.size(),
                                        right_permutation))
    {
        if (!currentRight.empty() && IsTransposeReshapeForEinsum(right_permutation,
                                                                shape(currentRight),
                                                                reshaped_dims))
        {
            currentRight = currentRight.reshape(1, reshaped_dims.size(), reshaped_dims.data());
        } else {
            currentRight = Transpose(!currentRight.empty() ? currentRight : right,
                                    !currentRight.empty() ? shape(currentRight) : rightDims,
                                    right_permutation);
        }
    }

    // Calculate output size
    // Output shape will be determined by rules of MatMul:
    // because we are multiplying two tensors of shapes [lro, lo, reduce_dims] , [lro, reduce_dims, ro]
    // [dim_value of `lro` dims,
    //  dim_value of `lo` dims,
    // `1` for each of the `reduce_dims`,
    // dim_value of `ro` dims]
    MatShape outputDims;
    outputDims.reserve(lro.size() + lo.size() + reduceDims.size() + ro.size());
    for (size_t i = 0; i < lro.size(); ++i)
    {
        outputDims.emplace_back(leftDims[lro[i]]);
    }

    for (size_t i = 0; i < lo.size(); ++i)
    {
        outputDims.emplace_back(leftDims[lo[i]]);
    }

    for (size_t i = 0; i < reduceDims.size(); ++i)
    {
        outputDims.emplace_back(1);  // reduced dimensions will have a value 1 in it
    }

    for (size_t i = 0; i < ro.size(); ++i) {
        outputDims.emplace_back(rightDims[ro[i]]);
    }

    MatShape currentSubscriptOrder;
    // Calculate output permutation
    // After the MatMul op, the because the two operands have been permutated,
    // the output is permutated as well with respect to the original ordering of the axes.
    // The permutated order will be the dims in: [lro, lo, reduced_dims, ro]
    // Hence invert the permutation by a permutation that puts the axes in the same ordering
    std::vector<size_t> outputPermutation;
    if (!isFinalPair) {  // If this is not the final pair, we need to permutate the result to match the pre-fixed order for the next iteration
        outputPermutation.resize(lro.size() + lo.size() + reduceDims.size() + ro.size(), 0);
        size_t iter = 0;
        for (size_t i = 0; i < lro.size(); ++i)
        {
            outputPermutation[lro[i]] = iter++;
        }

        for (size_t i = 0; i < lo.size(); ++i)
        {
            outputPermutation[lo[i]] = iter++;
        }

        for (size_t i = 0; i < reduceDims.size(); ++i)
        {
            outputPermutation[reduceDims[i]] = iter++;
        }

        for (size_t i = 0; i < ro.size(); ++i)
        {
            outputPermutation[ro[i]] = iter++;
        }

    } else {
        currentSubscriptOrder.reserve(lro.size() + lo.size() + reduceDims.size() + ro.size());
        currentSubscriptOrder.insert(currentSubscriptOrder.end(), lro.begin(), lro.end());
        currentSubscriptOrder.insert(currentSubscriptOrder.end(), lo.begin(), lo.end());
        currentSubscriptOrder.insert(currentSubscriptOrder.end(), reduceDims.begin(), reduceDims.end());
        currentSubscriptOrder.insert(currentSubscriptOrder.end(), ro.begin(), ro.end());
    }

    Mat output = batchwiseMatMul(
        !currentLeft.empty() ? currentLeft : left,
        MatShape({static_cast<int>(lro_size), static_cast<int>(lo_size), static_cast<int>(reduced_size)}),
        !currentRight.empty() ? currentRight : right,
        MatShape({static_cast<int>(lro_size), static_cast<int>(reduced_size), static_cast<int>(ro_size)})
        );

    //reshape
    output = output.reshape(1, outputDims.size(), outputDims.data());

    if (!isFinalPair)
    {  // This is not the final pair - so bring the axes order to what the inputs conformed to
        if (IsTransposeRequired(outputDims.size(), outputPermutation))
        {
            if (IsTransposeReshapeForEinsum(outputPermutation,
                                            outputDims,
                                            reshaped_dims))
            {
                // See note following the previous call of function IsTransposeReshapeForEinsum.
                // Covered by ExplicitEinsumAsTensorContractionReshapeFinal.
                output = output.reshape(1, reshaped_dims.size(), reshaped_dims.data());
            }
            else {
                output = Transpose(
                    output,
                    outputDims,
                    outputPermutation);
            }
        }
    } else {  // This is the final pair - Transpose directly to the output ordering required and copy the contents to the op's output
        // not sure if this finalize shape is needed at all
        output = FinalizeOutput(output, currentSubscriptOrder);
    }
    return output;
};

Mat LayerEinsumImpl::batchwiseMatMul(
    const Mat& input1,
    const MatShape& input1ShapeOverride,
    const Mat& input2,
    const MatShape& input2ShapeOverride)
{
    // Sanity checks before the actual MatMul
    CV_CheckType(input1.type(), input2.type(), "Data types of the inputs must match for MatMul");
    CV_CheckEQ(input1ShapeOverride.size(), (size_t) 3, "Only 1 batch dimension is allowed for MatMul");
    CV_CheckEQ(input2ShapeOverride.size(), (size_t) 3, "Only 1 batch dimension is allowed for MatMul");
    CV_CheckEQ((size_t) input1ShapeOverride[0], (size_t) input2ShapeOverride[0], "Batch dimension should match for MatMul;");
    CV_CheckEQ((size_t) input1ShapeOverride[2], (size_t) input2ShapeOverride[1], "Incompatible matrix dimensions for matMul");

    int batches = input1ShapeOverride[0];
    int M = input1ShapeOverride[1];
    int K = input1ShapeOverride[2];
    int N = input2ShapeOverride[2];

    Mat reshapedInput1 = input1;
    Mat reshapedInput2 = input2;


    Mat output;
    if (batches > 1)
    {
        // create tmpout with type like input1
        output = Mat({batches, M, N}, input1.type());

        reshapedInput2 = reshapedInput2.reshape(1, input2ShapeOverride);
        reshapedInput1 = reshapedInput1.reshape(1, input1ShapeOverride);

        fastGemmBatch(false, false, 1.0, reshapedInput1, reshapedInput2, 0.0, output, opt);
    } else {

        // input1 should of size MxK
        // check if input1 needs reshape, if need reshape
        if (input1.dims > 2 || input1.size[0] != M || (input1.dims > 1 && input1.size[1] != K) || input1.dims == 1)
        {
            int shape[] = {M, K};
            reshapedInput1 = input1.reshape(1, 2, shape);
        }

        // input2 should be of size KxN
        // check if input2 needs reshape, if needs reshape
        if (input2.dims > 2 || input2.size[0] != K || (input2.dims > 1 &&  input2.size[1] != N) || input2.dims == 1)
        {
            int shape2[] = {K, N};
            reshapedInput2 = input2.reshape(1, 2, shape2);
        }


        output = Mat(M, N, reshapedInput1.type());
        if ((reshapedInput1.dims == 0 && reshapedInput2.dims == 0)  ||
            (reshapedInput1.dims == 0 && reshapedInput2.dims != 0) ||
            (reshapedInput1.dims != 0 && reshapedInput2.dims == 0))
        {
            output = reshapedInput1.mul(reshapedInput2); // fastGemm does not support 0D * 0D multiplication
        } else {
            fastGemm(false, false, 1.0, reshapedInput1, reshapedInput2, 0.0, output, opt);
        }

        output = output.reshape(1, {1, M, N});
    }
    return output;
};
Ptr<EinsumLayer> EinsumLayer::create(const LayerParams& params)
{
    return makePtr<LayerEinsumImpl>(params);
}

}} // namespace cv::dnn
