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

/*M///////////////////////////////////////////////////////////////////////////////////////
//MIT License
//
//Copyright (c) 2017 Joseph Redmon
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.
//
//M*/

#ifndef __OPENCV_DNN_DARKNET_IO_HPP__
#define __OPENCV_DNN_DARKNET_IO_HPP__

#include <opencv2/dnn/dnn.hpp>

namespace cv {
    namespace dnn {
        namespace darknet {

            class LayerParameter {
                std::string layer_name, layer_type;
                std::vector<std::string> bottom_indexes;
                cv::dnn::LayerParams layerParams;
            public:
                friend class setLayersParams;
                cv::dnn::LayerParams getLayerParams() const { return layerParams; }
                std::string name() const { return layer_name; }
                std::string type() const { return layer_type; }
                int bottom_size() const { return bottom_indexes.size(); }
                std::string bottom(const int index) const { return bottom_indexes.at(index); }
                int top_size() const { return 1; }
                std::string top(const int index) const { return layer_name; }
            };

            class NetParameter {
            public:
                int width, height, channels;
                std::vector<LayerParameter> layers;
                std::vector<int> out_channels_vec;

                std::map<int, std::map<std::string, std::string> > layers_cfg;
                std::map<std::string, std::string> net_cfg;

                NetParameter() : width(0), height(0), channels(0) {}

                int layer_size() const { return layers.size(); }

                int input_size() const { return 1; }
                std::string input(const int index) const { return "data"; }
                LayerParameter layer(const int index) const { return layers.at(index); }
            };
        }

        // Read parameters from a file into a NetParameter message.
        void ReadNetParamsFromCfgFileOrDie(const char *cfgFile, darknet::NetParameter *net);
        void ReadNetParamsFromBinaryFileOrDie(const char *darknetModel, darknet::NetParameter *net);

    }
}
#endif
