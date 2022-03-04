/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../precomp.hpp"

#include <opencv2/core/utils/fp_control_utils.hpp>

#include <limits>
#include <set>
#include <map>
#include <algorithm>
#include <iostream>
#include <fstream>

#include "THDiskFile.h"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

using namespace TH;

//#ifdef NDEBUG
static bool dbgPrint = false;
//#else
//static bool dbgPrint = true;
//#endif

enum LuaType
{
    TYPE_NIL      = 0,
    TYPE_NUMBER   = 1,
    TYPE_STRING   = 2,
    TYPE_TABLE    = 3,
    TYPE_TORCH    = 4,
    TYPE_BOOLEAN  = 5,
    TYPE_FUNCTION = 6,
    TYPE_RECUR_FUNCTION = 8,
    LEGACY_TYPE_RECUR_FUNCTION = 7
};

// We use OpenCV's types to manage CV_ELEM_SIZE.
enum TorchType
{
    TYPE_DOUBLE = CV_64F,
    TYPE_FLOAT  = CV_32F,
    TYPE_BYTE   = CV_8U,
    TYPE_CHAR   = CV_8S,
    TYPE_SHORT  = CV_16S,
    TYPE_INT    = CV_32S,
    TYPE_LONG   = CV_32SC2
};

template<typename T>
static String toString(const T &v)
{
    std::ostringstream ss;
    ss << v;
    return ss.str();
}

static inline bool startsWith(const String &str, const char *substr)
{
    return str.find(substr) == 0;
}

static inline bool endsWith(const String &str, const char *substr)
{
    return str.rfind(substr) == str.length() - strlen(substr);
}

struct TorchImporter
{
    FPDenormalsIgnoreHintScope fp_denormals_ignore_scope;

    typedef std::map<String, std::pair<int, Mat> > TensorsMap;
    Net net;

    cv::Ptr<THFile> file;
    std::set<int> readedIndexes;
    std::map<int, Mat> storages;
    std::map<int, Mat> tensors;
    // Stack with numbers of unconnected layers per scope (Sequential, ConcatTable etc.)
    std::vector<int> numUnconnectedLayers;

    struct Module
    {
        String thName, apiType;
        dnn::LayerParams params;
        std::vector<cv::Ptr<Module> > modules;

        Module(const String &_thName, const String &_apiType = String())
            : thName(_thName), apiType(_apiType) {}
    };

    Module *rootModule;
    Module *curModule;
    int moduleCounter;
    bool testPhase;

    TorchImporter(String filename, bool isBinary, bool evaluate)
    {
        CV_TRACE_FUNCTION();

        rootModule = curModule = NULL;
        moduleCounter = 0;
        testPhase = evaluate;

        file = cv::Ptr<THFile>(THDiskFile_new(filename, "r", 0), THFile_free);
        CV_Assert(file && THFile_isOpened(file));

        if (isBinary)
            THFile_binary(file);
        else
            THFile_ascii(file);
    }

    /* Simple readers */

    inline int readInt()
    {
        return THFile_readIntScalar(file);
    }

    inline long readLong()
    {
        return THFile_readLongScalar(file);
    }

    inline bool readBool()
    {
        return readInt() != 0;
    }

    inline double readDouble()
    {
        return THFile_readDoubleScalar(file);
    }

    inline String readString()
    {
        int size = THFile_readIntScalar(file);
        String str(size, '\0');
        THFile_readCharRaw(file, const_cast<char*>(str.c_str()), size);
        return str;
    }

    inline String readTorchClassName()
    {
        String version = readString();
        return startsWith(version, "V ") ? readString() : version;
    }

    inline void readFunction()
    {
        readString();
        readObject();
    }

    void readTable(int index = -1)
    {
        index = (index < 0) ? readInt() : index;

        if (readedIndexes.count(index))
            return;

        readedIndexes.insert(index);

        int size = readInt();

        for (int i = 0; i < size; i++)
        {
            readObject(); //key
            readObject(); //value
        }
    }

    /* Special readers */

    static inline int parseTorchType(const String &str, const char *suffix, const char *prefix = "torch.")
    {
        if (startsWith(str, prefix) && endsWith(str, suffix))
        {
           String typeStr = str.substr(strlen(prefix), str.length() - strlen(prefix) - strlen(suffix));

           if (typeStr == "Double")
               return TYPE_DOUBLE;
           else if (typeStr == "Float" || typeStr == "Cuda")
               return TYPE_FLOAT;
           else if (typeStr == "Byte")
               return TYPE_BYTE;
           else if (typeStr == "Char")
               return TYPE_CHAR;
           else if (typeStr == "Short")
               return TYPE_SHORT;
           else if (typeStr == "Int")
               return TYPE_INT;
           else if (typeStr == "Long")
               return TYPE_LONG;
           else
               CV_Error(Error::StsNotImplemented, "Unknown type \"" + typeStr + "\" of torch class \"" + str + "\"");
        }

        return -1;
    }

    static int parseTensorType(const String &className)
    {
        return parseTorchType(className, "Tensor");
    }

    static int parseStorageType(const String &className)
    {
        return parseTorchType(className, "Storage");
    }

    void readTorchStorage(int index, int type = -1)
    {
        long size = readLong();
        Mat storageMat;

        switch (type)
        {
        case TYPE_FLOAT:
            storageMat.create(1, size, CV_32F);
            THFile_readFloatRaw(file, (float*)storageMat.data, size);
            break;
        case TYPE_DOUBLE:
            storageMat.create(1, size, CV_64F);
            THFile_readDoubleRaw(file, (double*)storageMat.data, size);
            break;
        case TYPE_CHAR:
            storageMat.create(1, size, CV_8S);
            THFile_readByteRaw(file, (uchar*)storageMat.data, size);
            break;
        case TYPE_BYTE:
            storageMat.create(1, size, CV_8U);
            THFile_readByteRaw(file, (uchar*)storageMat.data, size);
            break;
        case TYPE_SHORT:
            storageMat.create(1, size, CV_16S);
            THFile_readShortRaw(file, (short*)storageMat.data, size);
            break;
        case TYPE_INT:
            storageMat.create(1, size, CV_32S);
            THFile_readIntRaw(file, (int*)storageMat.data, size);
            break;
        case TYPE_LONG:
        {
            storageMat.create(1, size, CV_64F);   //handle LongStorage as CV_64F Mat
            double *buf = storageMat.ptr<double>();
            THFile_readLongRaw(file, (int64*)buf, size);

            for (size_t i = (size_t)size; i-- > 0; )
                buf[i] = ((int64*)buf)[i];
            break;
        }
        default:
            CV_Error(Error::StsInternal, "");
            break;
        }

        storages.insert(std::make_pair(index, storageMat));
    }

    void readTorchTable(Dict &scalarParams, TensorsMap &tensorParams)
    {
        int luaType = readInt();
        int index = readInt();

        CV_Assert(luaType == TYPE_TABLE && readedIndexes.count(index) == 0);
        readedIndexes.insert(index);

        long fpos;
        int numPairs = readInt();

        for (int i = 0; i < numPairs; i++)
        {
            fpos = THFile_position(file);
            int ktype = readInt();

            if (ktype != TYPE_STRING) //skip non-string fields
            {
                THFile_seek(file, fpos);
                readObject(); //key
                readObject(); //value
                continue;
            }

            String key = readString();
            if (dbgPrint)
                std::cout << i << "th key: " << key << "\n";

            fpos = THFile_position(file);
            int vtype = readInt();

            if (vtype == TYPE_TORCH)
            {
                int index = readInt();
                int numModules = curModule->modules.size();
                readTorchObject(index);

                if (tensors.count(index)) //tensor was read
                {
                    tensorParams.insert(std::make_pair(key, std::make_pair(index, tensors[index])));
                }
                else if (storages.count(index)) //storage was read
                {
                    Mat &matStorage = storages[index];
                    Mat matCasted;
                    matStorage.convertTo(matCasted, CV_64F);

                    DictValue scalar = DictValue::arrayReal(matCasted.ptr<double>(), matCasted.total());
                    scalarParams.set(key, scalar);
                }
                else
                {
                    // Only tensors and scalars are supported for table fields.
                    // i.e. nn.Inception has field `transfer` which is an
                    // activation layer. So we remove added modules as readTorchObject(index).
                    while (curModule->modules.size() > numModules)
                        curModule->modules.pop_back();
                }
            }
            else if (vtype == TYPE_NUMBER)
            {
                scalarParams.set(key, readDouble());
            }
            else if (vtype == TYPE_STRING)
            {
                scalarParams.set(key, readString());
            }
            else if (vtype == TYPE_BOOLEAN)
            {
                scalarParams.set(key, readBool());
            }
            else
            {
                THFile_seek(file, fpos);
                readObject();
            }
        }

        //Debug output
        if (dbgPrint)
        {
            std::cout << "scalarParams:\n";
            std::cout << scalarParams;

            std::cout << "#" << tensorParams.size() << " tensorParams:\n";
            std::map<String,std::pair<int, Mat> >::const_iterator it;
            for (it = tensorParams.begin(); it != tensorParams.end(); it++)
                std::cout << it->first << ": Tensor " << it->second.second.size << "\n";
        }
    }

    void readTorchTensor(int indexTensor, int typeTensor)
    {
        int ndims = readInt();
        AutoBuffer<int64, 4> sizes(ndims);
        AutoBuffer<int64, 4> steps(ndims);
        THFile_readLongRaw(file, sizes.data(), ndims);
        THFile_readLongRaw(file, steps.data(), ndims);
        long offset = readLong() - 1;

        //read Storage
        int typeidx = readInt();
        CV_Assert(typeidx == TYPE_TORCH || (typeidx == TYPE_NIL && ndims == 0));

        if (typeidx == TYPE_NIL)
        {
            tensors.insert(std::make_pair(indexTensor, Mat()));
            return;
        }

        int indexStorage = readInt();
        if (readedIndexes.count(indexStorage) == 0)
        {
            String className = readTorchClassName();
            int typeStorage = parseStorageType(className);
            CV_Assert(typeStorage >= 0 && typeTensor == typeStorage);
            readTorchStorage(indexStorage, typeStorage);
            typeTensor = storages[indexStorage].type();
            readedIndexes.insert(indexStorage);
        }

        //small check
        size_t requireElems = (size_t)offset + (size_t)steps[0] * (size_t)sizes[0];
        size_t storageElems = storages[indexStorage].total();
        if (requireElems > storageElems)
            CV_Error(Error::StsBadSize, "Storage has insufficient number of elements for requested Tensor");

        //convert sizes
        AutoBuffer<int, 4> isizes(ndims);
        AutoBuffer<size_t, 4> ssteps(ndims);
        for (int i = ndims - 1; i >= 0; i--)
        {
            isizes[i] = (int)sizes[i];
            ssteps[i] = (size_t)steps[i] * CV_ELEM_SIZE(typeTensor);
        }

        //allocate Blob
        Mat srcMat(ndims, isizes.data(), typeTensor , storages[indexStorage].ptr() + offset*CV_ELEM_SIZE(typeTensor), ssteps.data());
        int dstType = CV_32F;

        Mat blob;
        srcMat.convertTo(blob, dstType);

        tensors.insert(std::make_pair(indexTensor, blob));
    }

    static bool isNNClass(const String &className, String &nnName)
    {
        const char *prefixes[] = {"nn.", "cunn.", "cudnn.", "fbcunn.", NULL};

        for (int i = 0; prefixes[i]; i++)
        {
            if (startsWith(className, prefixes[i]))
            {
                nnName = className.substr(strlen(prefixes[i]));
                return true;
            }
        }

        return false;
    }

    static void convertTorchKernelsParams(const Dict &torchParams, cv::dnn::LayerParams &layerParams)
    {
        layerParams.set("kernel_h", torchParams.get<int>("kH"));
        layerParams.set("kernel_w", torchParams.get<int>("kW"));
        layerParams.set("stride_h", torchParams.get<int>("dH"));
        layerParams.set("stride_w", torchParams.get<int>("dW"));
        layerParams.set("pad_h", torchParams.get<int>("padH", 0));
        layerParams.set("pad_w", torchParams.get<int>("padW", 0));
    }

    void readTorchObject(int index)
    {
        if(readedIndexes.count(index))
            return;

        String className = readTorchClassName();
        String nnName;

        if (dbgPrint)
            std::cout << "Class: " << className << std::endl;

        int type;
        if ( (type = parseTensorType(className)) >= 0 ) //is Tensor
        {
            readTorchTensor(index, type);
        }
        else if ( (type = parseStorageType(className)) >= 0 ) //is Storage
        {
            readTorchStorage(index, type);
        }
        else if (isNNClass(className, nnName))
        {
            Dict scalarParams;
            TensorsMap tensorParams;

            cv::Ptr<Module> newModule(new Module(nnName));
            cv::dnn::LayerParams &layerParams = newModule->params;

            layerParams.set("torch_index", index);

            if (nnName == "Sequential" || nnName == "Parallel" ||
                nnName == "Concat" || nnName == "ConcatTable" || nnName == "JoinTable" ||
                nnName == "DepthConcat" || nnName == "Inception")
            {
                Module *parentModule = curModule;
                curModule->modules.push_back(newModule);
                curModule = newModule;
                readTorchTable(scalarParams, tensorParams);
                curModule = parentModule;

                if (nnName == "Parallel")
                {
                    layerParams.set("inputDimension", scalarParams.get<int>("inputDimension"));
                    layerParams.set("outputDimension", scalarParams.get<int>("outputDimension"));
                }
                else if (nnName == "Concat" || nnName == "JoinTable" || nnName == "DepthConcat")
                {
                    layerParams.set("dimension", scalarParams.get<int>("dimension"));
                }
            }
            else if (nnName == "SpatialConvolution" || nnName == "SpatialConvolutionMM")
            {
                newModule->apiType = "Convolution";
                readTorchTable(scalarParams, tensorParams);

                CV_Assert(tensorParams.count("weight"));
                layerParams.blobs.push_back(tensorParams["weight"].second);

                bool bias = tensorParams.count("bias") != 0;
                layerParams.set("bias_term", bias);
                if (bias)
                    layerParams.blobs.push_back(tensorParams["bias"].second);

                layerParams.set("num_output", scalarParams.get<int>("nOutputPlane"));
                convertTorchKernelsParams(scalarParams, layerParams);

                if (nnName == "SpatialConvolutionMM")
                {
                    // Split weights from a [ outCh x inCh*kH*kW ] 2D matrix
                    // onto a 4D [ outCh x inCh x kH x kW ] blob.
                    CV_Assert(layerParams.blobs[0].dims == 2);
                    const int kernel = layerParams.blobs[0].size[1];  // inCh * kH * kW
                    MatShape kernelShape(4);
                    kernelShape[0] = layerParams.blobs[0].size[0];  // outCh.
                    kernelShape[2] = layerParams.get<int>("kernel_h");
                    kernelShape[3] = layerParams.get<int>("kernel_w");
                    kernelShape[1] = kernel / (kernelShape[2] * kernelShape[3]);  // inCh.
                    layerParams.blobs[0] = layerParams.blobs[0].reshape(1, kernelShape);
                }
                curModule->modules.push_back(newModule);
            }
            else if (nnName == "SpatialLPPooling")
            {
                // nn.Sequential {
                //     [input -> (1) -> (2) -> output]
                //     (1): nn.Sequential {
                //       [input -> (1) -> (2) -> (3) -> (4) -> output]
                //       (1): nn.Power
                //       (2): nn.SpatialAveragePooling(...)
                //       (3): nn.MulConstant
                //       (4): nn.Power
                //     }
                //     (2): nn.Sigmoid
                // }
                // nn.SpatialLPPooling is just a table so we skip it.
                readTorchTable(scalarParams, tensorParams);
            }
            else if (nnName == "SpatialMaxPooling" || nnName == "SpatialAveragePooling")
            {
                newModule->apiType = "Pooling";
                readTorchTable(scalarParams, tensorParams);

                if (nnName == "SpatialMaxPooling") {
                    layerParams.set("pool", "MAX");
                    layerParams.set("indices_blob_id", tensorParams["indices"].first);
                }
                if (nnName == "SpatialAveragePooling")
                {
                    layerParams.set("pool", "AVE");
                    layerParams.set("ave_pool_padded_area", scalarParams.has("count_include_pad") &&
                                                            scalarParams.get<bool>("count_include_pad"));
                }
                convertTorchKernelsParams(scalarParams, layerParams);

                CV_Assert(scalarParams.has("ceil_mode"));
                layerParams.set("ceil_mode", scalarParams.get<bool>("ceil_mode"));

                curModule->modules.push_back(newModule);
            }
            else if (nnName == "Linear")
            {
                newModule->apiType = "InnerProduct";
                readTorchTable(scalarParams, tensorParams);

                CV_Assert(tensorParams.count("weight"));
                Mat weightBlob = tensorParams["weight"].second;
                layerParams.blobs.push_back(weightBlob);

                bool bias = tensorParams.count("bias") != 0;
                if (bias)
                    layerParams.blobs.push_back(tensorParams["bias"].second);
                layerParams.set("bias_term", bias);

                layerParams.set("num_output", weightBlob.size[0]);
                curModule->modules.push_back(newModule);
            }
            else if (nnName == "Reshape" || nnName == "View")
            {
                newModule->apiType = "Reshape";

                readTorchTable(scalarParams, tensorParams);
                CV_Assert(scalarParams.has("size"));

                DictValue dimParam = scalarParams.get("size");
                layerParams.set("dim", dimParam);

                int axis = (int)scalarParams.get<bool>("batchMode", true);
                layerParams.set("axis", axis);

                curModule->modules.push_back(newModule);
            }
            else if (nnName == "ReLU")
            {
                curModule->modules.push_back(cv::Ptr<Module>(new Module(nnName, "ReLU")));
                readObject();
            }
            else if (nnName == "Tanh")
            {
                curModule->modules.push_back(cv::Ptr<Module>(new Module(nnName, "TanH")));
                readObject();
            }
            else if (nnName == "Sigmoid")
            {
                curModule->modules.push_back(cv::Ptr<Module>(new Module(nnName, "Sigmoid")));
                readObject();
            }
            else if (nnName == "SpatialBatchNormalization" || nnName == "InstanceNormalization" ||
                     nnName == "BatchNormalization")
            {
                newModule->apiType = "BatchNorm";
                readTorchTable(scalarParams, tensorParams);

                CV_Assert(scalarParams.has("eps"));
                float eps = float(scalarParams.get<double>("eps"));
                layerParams.set("eps", eps);

                if (tensorParams.count("running_mean"))
                {
                    layerParams.blobs.push_back(tensorParams["running_mean"].second);
                }
                else
                {
                    CV_Assert(scalarParams.has("nOutput"));
                    layerParams.blobs.push_back(Mat::zeros(1, scalarParams.get<int>("nOutput"), CV_32F));
                }

                if (tensorParams.count("running_var"))
                {
                    layerParams.blobs.push_back(tensorParams["running_var"].second);
                }
                else if (tensorParams.count("running_std"))
                {
                    layerParams.blobs.push_back(tensorParams["running_std"].second);
                    pow(layerParams.blobs.back(), -2, layerParams.blobs.back());
                    subtract(layerParams.blobs.back(), eps, layerParams.blobs.back());
                }
                else
                {
                    CV_Assert(scalarParams.has("nOutput"));
                    layerParams.blobs.push_back(Mat::ones(1, scalarParams.get<int>("nOutput"), CV_32F));
                }

                if (tensorParams.count("weight"))
                {
                    layerParams.set("has_weight", true);
                    layerParams.blobs.push_back(tensorParams["weight"].second);
                }

                if (tensorParams.count("bias"))
                {
                    layerParams.set("has_bias", true);
                    layerParams.blobs.push_back(tensorParams["bias"].second);
                }

                bool trainPhase = scalarParams.get<bool>("train", false);
                if (nnName == "InstanceNormalization" || (trainPhase && !testPhase))
                {
                    cv::Ptr<Module> mvnModule(new Module(nnName));
                    mvnModule->apiType = "MVN";
                    curModule->modules.push_back(mvnModule);

                    layerParams.blobs[0].setTo(0);  // batch norm's mean
                    layerParams.blobs[1].setTo(1);  // batch norm's std
                }

                curModule->modules.push_back(newModule);
            }
            else if (nnName == "PReLU")
            {
                readTorchTable(scalarParams, tensorParams);

                CV_Assert(tensorParams.count("weight"));

                size_t outputChannels = static_cast<int>(scalarParams.get<double>("nOutputPlane"));
                if (outputChannels) {

                    CV_Assert(tensorParams["weight"].second.total() == outputChannels);
                    layerParams.blobs.push_back(tensorParams["weight"].second);

                    newModule->apiType = "ChannelsPReLU";
                }
                else {
                    CV_Assert(tensorParams["weight"].second.total() == 1);
                    float negative_slope = *tensorParams["weight"].second.ptr<float>();
                    layerParams.set("negative_slope", negative_slope);

                    newModule->apiType = "ReLU";
                }

                curModule->modules.push_back(newModule);
            }
            else if (nnName == "SpatialDropout" || nnName == "Dropout")
            {
                readTorchTable(scalarParams, tensorParams);
                CV_Assert(scalarParams.has("p"));

                if (scalarParams.has("v2") && scalarParams.get<bool>("v2"))
                {
                    newModule->apiType = "Identity";
                }
                else
                {
                    float scale = 1 -  scalarParams.get<double>("p");

                    CV_Assert(scale > 0);

                    newModule->apiType = "Power";
                    layerParams.set("scale", scale);
                }
                curModule->modules.push_back(newModule);
            }
            // TotalVariation layer is from fast-neural-style project: https://github.com/jcjohnson/fast-neural-style
            // It's a loss function that has an Identity forward.
            else if (nnName == "Identity" || nnName == "TotalVariation")
            {
                readTorchTable(scalarParams, tensorParams);
                newModule->apiType = "Identity";
                curModule->modules.push_back(newModule);
            }
            else if (nnName == "Normalize")
            {
                readTorchTable(scalarParams, tensorParams);
                CV_Assert(scalarParams.has("p"));

                layerParams.set("p", scalarParams.get<float>("p"));
                if (scalarParams.has("eps"))
                    layerParams.set("eps", scalarParams.get<float>("eps"));

                newModule->apiType = "Normalize";
                curModule->modules.push_back(newModule);
            }
            else if (nnName == "Padding")
            {
                readTorchTable(scalarParams, tensorParams);
                newModule->apiType = "Padding";

                CV_Assert(scalarParams.has("pad") && scalarParams.has("dim"));
                if (scalarParams.has("index") && scalarParams.get<int>("index") != 1)
                    CV_Error(Error::StsNotImplemented, "Padding with offset is not implemented");

                if (scalarParams.has("value"))
                    layerParams.set("value", scalarParams.get<float>("value"));

                if (scalarParams.has("nInputDim"))
                    layerParams.set("input_dims", scalarParams.get<int>("nInputDim"));

                int dim = scalarParams.get<int>("dim") - 1;  // In Lua we start from 1.
                int pad = scalarParams.get<int>("pad");

                std::vector<int> paddings((dim + 1) * 2, 0);
                if (pad > 0)
                    paddings[dim * 2 + 1] = pad;  // Pad after (right).
                else
                    paddings[dim * 2] = -pad;  // Pad before (left).
                layerParams.set("paddings", DictValue::arrayInt<int*>(&paddings[0], paddings.size()));

                curModule->modules.push_back(newModule);
            }
            else if (nnName == "CAddTable")
            {
                curModule->modules.push_back(newModule);
                readObject();
            }
            else if (nnName == "SpatialDilatedConvolution")
            {
                readTorchTable(scalarParams, tensorParams);
                newModule->apiType = "Convolution";
                CV_Assert(scalarParams.has("padW") &&
                          scalarParams.has("padH")&&
                          scalarParams.has("dW")&&
                          scalarParams.has("dH")&&
                          scalarParams.has("dilationW")&&
                          scalarParams.has("dilationH")&&
                          scalarParams.has("kW")&&
                          scalarParams.has("kH")&&
                          scalarParams.has("nOutputPlane"));

                layerParams.set("kernel_w", static_cast<int>(scalarParams.get<double>("kW")));
                layerParams.set("kernel_h", static_cast<int>(scalarParams.get<double>("kH")));
                layerParams.set("pad_w", static_cast<int>(scalarParams.get<double>("padW")));
                layerParams.set("pad_h", static_cast<int>(scalarParams.get<double>("padH")));
                layerParams.set("stride_w", static_cast<int>(scalarParams.get<double>("dW")));
                layerParams.set("stride_h", static_cast<int>(scalarParams.get<double>("dH")));
                layerParams.set("dilation_w", static_cast<int>(scalarParams.get<double>("dilationW")));
                layerParams.set("dilation_h", static_cast<int>(scalarParams.get<double>("dilationH")));
                layerParams.set("num_output", static_cast<int>(scalarParams.get<double>("nOutputPlane")));

                layerParams.blobs.push_back(tensorParams["weight"].second);

                bool bias = tensorParams.count("bias");
                layerParams.set("bias_term", bias);
                if (bias)
                    layerParams.blobs.push_back(tensorParams["bias"].second);

                curModule->modules.push_back(newModule);
            }
            else if (nnName == "SpatialFullConvolution")
            {
                readTorchTable(scalarParams, tensorParams);
                newModule->apiType = "Deconvolution";
                CV_Assert(scalarParams.has("padW") &&
                          scalarParams.has("padH")&&
                          scalarParams.has("dW")&&
                          scalarParams.has("dH")&&
                          scalarParams.has("adjW")&&
                          scalarParams.has("adjH")&&
                          scalarParams.has("kW")&&
                          scalarParams.has("kH")&&
                          scalarParams.has("nOutputPlane"));

                layerParams.set("kernel_w", static_cast<int>(scalarParams.get<double>("kW")));
                layerParams.set("kernel_h", static_cast<int>(scalarParams.get<double>("kH")));
                layerParams.set("pad_w", static_cast<int>(scalarParams.get<double>("padW")));
                layerParams.set("pad_h", static_cast<int>(scalarParams.get<double>("padH")));
                layerParams.set("stride_w", static_cast<int>(scalarParams.get<double>("dW")));
                layerParams.set("stride_h", static_cast<int>(scalarParams.get<double>("dH")));
                layerParams.set("adj_w", static_cast<int>(scalarParams.get<double>("adjW")));
                layerParams.set("adj_h", static_cast<int>(scalarParams.get<double>("adjH")));
                layerParams.set("num_output", static_cast<int>(scalarParams.get<double>("nOutputPlane")));

                layerParams.blobs.push_back(tensorParams["weight"].second);

                bool bias = tensorParams.count("bias");
                layerParams.set("bias_term", bias);
                if (bias)
                    layerParams.blobs.push_back(tensorParams["bias"].second);

                curModule->modules.push_back(newModule);
            }
            else if (nnName == "SpatialMaxUnpooling")
            {
                readTorchTable(scalarParams, tensorParams);
                CV_Assert(tensorParams.count("indices"));

                layerParams.set("indices_blob_id", tensorParams["indices"].first);
                curModule->modules.push_back(newModule);
            }
            else if (nnName == "LogSoftMax" || nnName == "SoftMax")
            {
                newModule->apiType = "Softmax";
                layerParams.set("log_softmax", nnName == "LogSoftMax");
                curModule->modules.push_back(newModule);
            }
            else if (nnName == "SpatialCrossMapLRN")
            {
                newModule->apiType = "LRN";
                readTorchTable(scalarParams, tensorParams);

                CV_Assert(scalarParams.has("alpha"));
                CV_Assert(scalarParams.has("beta"));
                CV_Assert(scalarParams.has("k"));
                CV_Assert(scalarParams.has("size"));

                layerParams.set("norm_region", "ACROSS_CHANNELS");
                layerParams.set("alpha", scalarParams.get<float>("alpha"));
                layerParams.set("beta", scalarParams.get<float>("beta"));
                layerParams.set("bias", scalarParams.get<float>("k"));
                layerParams.set("local_size", scalarParams.get<int>("size"));
                layerParams.set("norm_by_size", true);

                curModule->modules.push_back(newModule);
            }
            else if (nnName == "Square" || nnName == "Sqrt" || nnName == "Power")
            {
                readTorchTable(scalarParams, tensorParams);

                float power;
                if (nnName == "Square") power = 2.0f;
                else if (nnName == "Sqrt") power = 0.5f;
                else if (nnName == "Power") power = scalarParams.get<float>("pow", 1.0f);

                newModule->apiType = "Power";
                layerParams.set("power", power);
                curModule->modules.push_back(newModule);
            }
            else if (nnName == "MulConstant")
            {
                readTorchTable(scalarParams, tensorParams);
                CV_Assert(scalarParams.has("constant_scalar"));
                newModule->apiType = "Power";
                layerParams.set("scale", scalarParams.get<float>("constant_scalar"));
                curModule->modules.push_back(newModule);
            }
            else if (nnName == "SpatialZeroPadding" || nnName == "SpatialReflectionPadding")
            {
                readTorchTable(scalarParams, tensorParams);
                CV_Assert_N(scalarParams.has("pad_l"), scalarParams.has("pad_r"),
                            scalarParams.has("pad_t"), scalarParams.has("pad_b"));
                int padTop = scalarParams.get<int>("pad_t");
                int padLeft = scalarParams.get<int>("pad_l");
                int padRight = scalarParams.get<int>("pad_r");
                int padBottom = scalarParams.get<int>("pad_b");
                if (padTop < 0 || padLeft < 0 || padRight < 0 || padBottom < 0)
                    CV_Error(Error::StsNotImplemented, "SpatialZeroPadding in cropping mode is not implemented");

                newModule->apiType = "Padding";

                // Torch's SpatialZeroPadding works with 3- or 4-dimensional input.
                // So we add parameter input_dims=3 to ignore batch dimension if it will be.
                std::vector<int> paddings(6, 0);  // CHW
                paddings[2] = padTop;
                paddings[3] = padBottom;
                paddings[4] = padLeft;
                paddings[5] = padRight;
                layerParams.set("paddings", DictValue::arrayInt<int*>(&paddings[0], paddings.size()));
                layerParams.set("input_dims", 3);

                if (nnName == "SpatialReflectionPadding")
                    layerParams.set("type", "reflect");

                curModule->modules.push_back(newModule);
            }
            else if (nnName == "ShaveImage")
            {
                // ShaveImage layer is from fast-neural-style project: https://github.com/jcjohnson/fast-neural-style
                // It may be mapped to Slice layer.
                readTorchTable(scalarParams, tensorParams);
                CV_Assert(scalarParams.has("size"));
                int size = scalarParams.get<int>("size");

                int begins[] = {0, 0, size, size};
                int ends[] = {-1, -1, -size - 1, -size - 1};

                newModule->apiType = "Slice";
                layerParams.set("begin", DictValue::arrayInt<int*>(&begins[0], 4));
                layerParams.set("end", DictValue::arrayInt<int*>(&ends[0], 4));
                curModule->modules.push_back(newModule);
            }
            else if (nnName == "SpatialUpSamplingNearest")
            {
                readTorchTable(scalarParams, tensorParams);
                CV_Assert(scalarParams.has("scale_factor"));
                int scale_factor = scalarParams.get<int>("scale_factor");
                newModule->apiType = "Resize";
                layerParams.set("interpolation", "nearest");
                layerParams.set("zoom_factor", scale_factor);
                curModule->modules.push_back(newModule);
            }
            else
            {
                // Importer does not know how to map Torch's layer type to an OpenCV's one.
                // However we parse all the parameters to let user create a custom layer.
                readTorchTable(scalarParams, tensorParams);
                for (std::map<String, DictValue>::const_iterator it = scalarParams.begin();
                     it != scalarParams.end(); ++it)
                {
                    layerParams.set(it->first, it->second);
                }
                for (std::map<String, std::pair<int, Mat> >::iterator it = tensorParams.begin();
                     it != tensorParams.end(); ++it)
                {
                    layerParams.blobs.push_back(it->second.second);
                }
                newModule->apiType = nnName;
                curModule->modules.push_back(newModule);
            }
        }
        else
        {
            CV_Error(Error::StsNotImplemented, "Unsupported Torch class \"" + className + "\"");
        }

        readedIndexes.insert(index);
    }

    void readObject()
    {
        int typeidx = readInt();

        if (typeidx == TYPE_TORCH)
        {
            int index = readInt();
            readTorchObject(index);
            readedIndexes.insert(index);
        }
        else if (typeidx == TYPE_NIL)
            return;
        else if (typeidx == TYPE_NUMBER)
            readDouble();
        else if (typeidx == TYPE_BOOLEAN)
            readBool();
        else if (typeidx == TYPE_STRING)
            readString();
        else if (typeidx == TYPE_TABLE)
            readTable();
        else
            CV_Error(Error::StsNotImplemented, "Unsupported Lua type");
    }

    inline String generateLayerName(const String &label = String())
    {
        return "l" + toString(++this->moduleCounter) + "_" + label;
    }

    int fill(Module *module, std::vector<std::pair<int, Module*> >& addedModules, int prevLayerId = 0, int prevOutNum = 0)
    {
        if (module == NULL)
            return prevLayerId;

        if (module->apiType.length())
        {
            int newLayerId = net.addLayer(generateLayerName(module->apiType), module->apiType, module->params);
            net.connect(prevLayerId, prevOutNum, newLayerId, 0);
            addedModules.push_back(std::make_pair(newLayerId, module));
            return newLayerId;
        }
        else
        {
            if (module->thName == "Sequential" || module->thName == "Inception")
            {
                for (size_t i = 0; i < module->modules.size(); i++)
                {
                    prevLayerId = fill(module->modules[i], addedModules, prevLayerId, prevOutNum);
                    prevOutNum = 0;
                }
                return prevLayerId;
            }
            else if (module->thName == "Concat")
            {
                int newId, mergeId;
                LayerParams mergeParams;
                mergeParams.set("axis", module->params.get<int>("dimension") - 1);

                std::vector<int> branchIds;
                for (int i = 0; i < (int)module->modules.size(); i++)
                {
                    newId = fill(module->modules[i], addedModules, prevLayerId, prevOutNum);
                    branchIds.push_back(newId);
                }

                moduleCounter += 1;  // Skip split layer creation. See https://github.com/opencv/opencv/pull/9384.
                mergeId = net.addLayer(generateLayerName("torchMerge"), "Concat", mergeParams);

                for (int i = 0; i < branchIds.size(); i++)
                {
                    net.connect(branchIds[i], 0, mergeId, i);
                }

                addedModules.push_back(std::make_pair(mergeId, module));
                return mergeId;
            }
            else if (module->thName == "DepthConcat")
            {
                int newId, mergeId;
                LayerParams mergeParams;
                mergeParams.set("axis", module->params.get<int>("dimension") - 1);
                mergeParams.set("padding", true);

                std::vector<int> branchIds;
                for (int i = 0; i < (int)module->modules.size(); i++)
                {
                    newId = fill(module->modules[i], addedModules, prevLayerId, prevOutNum);
                    branchIds.push_back(newId);
                }

                mergeId = net.addLayer(generateLayerName("torchMerge"), "Concat", mergeParams);

                for (int i = 0; i < branchIds.size(); i++)
                {
                    net.connect(branchIds[i], 0, mergeId, i);
                }

                addedModules.push_back(std::make_pair(mergeId, module));
                return mergeId;
            }
            else if (module->thName == "Parallel")
            {
                int newId, splitId, mergeId, reshapeId;

                LayerParams splitParams, mergeParams, reshapeParams;
                splitParams.set("axis", module->params.get<int>("inputDimension") - 1);
                mergeParams.set("axis", module->params.get<int>("outputDimension") - 1);
                reshapeParams.set("axis", splitParams.get<int>("axis"));
                reshapeParams.set("num_axes", 1);

                splitId = net.addLayer(generateLayerName("torchSplit"), "Slice", splitParams);
                reshapeId = net.addLayer(generateLayerName("torchReshape"), "Reshape", reshapeParams);
                net.connect(prevLayerId, prevOutNum, splitId, 0);

                std::vector<int> branchIds;
                for (int i = 0; i < (int)module->modules.size(); i++)
                {
                    net.connect(splitId, i, reshapeId, i);
                    newId = fill(module->modules[i], addedModules, reshapeId, i);
                    branchIds.push_back(newId);
                }

                mergeId = net.addLayer(generateLayerName("torchMerge"), "Concat", mergeParams);

                for (int i = 0; i < branchIds.size(); i++)
                {
                    net.connect(branchIds[i], 0, mergeId, i);
                }

                addedModules.push_back(std::make_pair(mergeId, module));
                return mergeId;
            }
            else if (module->thName == "ConcatTable") {
                int newId = -1;
                moduleCounter += 1;  // Skip split layer creation. See https://github.com/opencv/opencv/pull/9384.
                for (int i = 0; i < (int)module->modules.size(); i++)
                {
                    newId = fill(module->modules[i], addedModules, prevLayerId, prevOutNum);
                }
                numUnconnectedLayers.push_back(module->modules.size());
                return newId;
            }
            else if (module->thName == "JoinTable") {
                std::vector<int> ids = net.getUnconnectedOutLayers();

                int mergeId;
                LayerParams mergeParams;
                mergeParams.set("axis", module->params.get<int>("dimension") - 1);

                mergeId = net.addLayer(generateLayerName("torchMerge"), "Concat", mergeParams);
                addedModules.push_back(std::make_pair(mergeId, module));

                // Connect to the last number of unconnected layers.
                CV_Assert(!numUnconnectedLayers.empty());
                const int numInputs = numUnconnectedLayers.back();
                numUnconnectedLayers.pop_back();
                CV_Assert(numInputs <= ids.size());
                for (int i = 0; i < numInputs; i++)
                {
                    net.connect(ids[ids.size() - numInputs + i], 0, mergeId, i);
                }

                return mergeId;
            }
            else if (module->thName == "CAddTable") {
                String name = generateLayerName("torchCAddTable");
                std::vector<int> ids = net.getUnconnectedOutLayers();
                LayerParams params;
                params.set("operation", "sum");


                int id = net.addLayer(name, "Eltwise", params);

                // Connect to the last number of unconnected layers.
                CV_Assert(!numUnconnectedLayers.empty());
                const int numInputs = numUnconnectedLayers.back();
                numUnconnectedLayers.pop_back();
                CV_Assert(numInputs <= ids.size());
                for (int i = 0; i < numInputs; i++)
                {
                    net.connect(ids[ids.size() - numInputs + i], 0, id, i);
                }

                addedModules.push_back(std::make_pair(id, module));
                return id;
            }
            else if (module->thName == "SpatialMaxUnpooling") {
                CV_Assert(module->params.has("indices_blob_id"));
                int indicesBlobId = module->params.get<int>("indices_blob_id");
                std::pair<int, Module*> poolingLayer;
                poolingLayer.first = -1;

                for(int i = 0; i < addedModules.size(); i++)
                {
                    if (addedModules[i].second->apiType == "Pooling" &&
                        addedModules[i].second->params.has("indices_blob_id") &&
                        addedModules[i].second->params.get<int>("indices_blob_id") == indicesBlobId)
                    {
                        poolingLayer = addedModules[i];
                        break;
                    }
                }

                module->params.set("pool_k_h", poolingLayer.second->params.get<int>("kernel_h"));
                module->params.set("pool_k_w", poolingLayer.second->params.get<int>("kernel_w"));
                module->params.set("pool_stride_h", poolingLayer.second->params.get<int>("stride_h"));
                module->params.set("pool_stride_w", poolingLayer.second->params.get<int>("stride_w"));
                module->params.set("pool_pad_h", poolingLayer.second->params.get<int>("pad_h"));
                module->params.set("pool_pad_w", poolingLayer.second->params.get<int>("pad_w"));

                String name = generateLayerName("torchMaxUnpooling");
                int id = net.addLayer(name, "MaxUnpool", module->params);
                net.connect(prevLayerId, 0, id, 0);

                CV_Assert(poolingLayer.first != -1);
                net.connect(poolingLayer.first, 1, id, 1);

                return id;
            }
        }

        CV_Error(Error::StsInternal, "Unexpected torch container: " + module->thName);
        return -1;
    }

    void populateNet(Net net_)
    {
        CV_TRACE_FUNCTION();

        CV_Assert(rootModule == NULL);
        cv::Ptr<Module> rootModule_ = cv::makePtr<Module>("Sequential");
        rootModule = rootModule_.get();
        curModule = rootModule;

        THFile_seek(file, 0);
        readObject();

        net = net_;
        std::vector<std::pair<int, Module*> > addedModules;
        fill(rootModule, addedModules);

        rootModule = NULL;
        curModule = NULL;
    }
};

Mat readTorchBlob(const String &filename, bool isBinary)
{
    TorchImporter importer(filename, isBinary, true);
    importer.readObject();
    CV_Assert(importer.tensors.size() == 1);

    return importer.tensors.begin()->second;
}

Net readNetFromTorch(const String &model, bool isBinary, bool evaluate)
{
    CV_TRACE_FUNCTION();

    TorchImporter importer(model, isBinary, evaluate);
    Net net;
    importer.populateNet(net);
    return net;
}

CV__DNN_INLINE_NS_END
}} // namespace
