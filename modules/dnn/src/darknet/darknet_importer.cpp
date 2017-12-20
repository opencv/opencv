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
//                        (3-clause BSD License)
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of the contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../precomp.hpp"

#include <iostream>
#include <algorithm>
#include <vector>
#include <map>

#include "darknet_io.hpp"


namespace cv {
namespace dnn {
CV__DNN_EXPERIMENTAL_NS_BEGIN

namespace
{

class DarknetImporter
{
    darknet::NetParameter net;

public:

    DarknetImporter() {}

    DarknetImporter(const char *cfgFile, const char *darknetModel)
    {
        CV_TRACE_FUNCTION();

        ReadNetParamsFromCfgFileOrDie(cfgFile, &net);

        if (darknetModel && darknetModel[0])
            ReadNetParamsFromBinaryFileOrDie(darknetModel, &net);
    }

    struct BlobNote
    {
        BlobNote(const std::string &_name, int _layerId, int _outNum) :
            name(_name), layerId(_layerId), outNum(_outNum) {}

        std::string name;
        int layerId, outNum;
    };

    std::vector<BlobNote> addedBlobs;
    std::map<String, int> layerCounter;

    void populateNet(Net dstNet)
    {
        CV_TRACE_FUNCTION();

        int layersSize = net.layer_size();
        layerCounter.clear();
        addedBlobs.clear();
        addedBlobs.reserve(layersSize + 1);

        //setup input layer names
        {
            std::vector<String> netInputs(net.input_size());
            for (int inNum = 0; inNum < net.input_size(); inNum++)
            {
                addedBlobs.push_back(BlobNote(net.input(inNum), 0, inNum));
                netInputs[inNum] = net.input(inNum);
            }
            dstNet.setInputsNames(netInputs);
        }

        for (int li = 0; li < layersSize; li++)
        {
            const darknet::LayerParameter &layer = net.layer(li);
            String name = layer.name();
            String type = layer.type();
            LayerParams layerParams = layer.getLayerParams();

            int repetitions = layerCounter[name]++;
            if (repetitions)
                name += cv::format("_%d", repetitions);

            int id = dstNet.addLayer(name, type, layerParams);

            // iterate many bottoms layers (for example for: route -1, -4)
            for (int inNum = 0; inNum < layer.bottom_size(); inNum++)
                addInput(layer.bottom(inNum), id, inNum, dstNet, layer.name());

            for (int outNum = 0; outNum < layer.top_size(); outNum++)
                addOutput(layer, id, outNum);
        }

        addedBlobs.clear();
    }

    void addOutput(const darknet::LayerParameter &layer, int layerId, int outNum)
    {
        const std::string &name = layer.top(outNum);

        bool haveDups = false;
        for (int idx = (int)addedBlobs.size() - 1; idx >= 0; idx--)
        {
            if (addedBlobs[idx].name == name)
            {
                haveDups = true;
                break;
            }
        }

        if (haveDups)
        {
            bool isInplace = layer.bottom_size() > outNum && layer.bottom(outNum) == name;
            if (!isInplace)
                CV_Error(Error::StsBadArg, "Duplicate blobs produced by multiple sources");
        }

        addedBlobs.push_back(BlobNote(name, layerId, outNum));
    }

    void addInput(const std::string &name, int layerId, int inNum, Net &dstNet, std::string nn)
    {
        int idx;
        for (idx = (int)addedBlobs.size() - 1; idx >= 0; idx--)
        {
            if (addedBlobs[idx].name == name)
                break;
        }

        if (idx < 0)
        {
            CV_Error(Error::StsObjectNotFound, "Can't find output blob \"" + name + "\"");
            return;
        }

        dstNet.connect(addedBlobs[idx].layerId, addedBlobs[idx].outNum, layerId, inNum);
    }
};

}

Net readNetFromDarknet(const String &cfgFile, const String &darknetModel /*= String()*/)
{
    DarknetImporter darknetImporter(cfgFile.c_str(), darknetModel.c_str());
    Net net;
    darknetImporter.populateNet(net);
    return net;
}

CV__DNN_EXPERIMENTAL_NS_END
}} // namespace
