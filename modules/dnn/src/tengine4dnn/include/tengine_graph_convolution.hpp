/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2020, OPEN AI LAB
 * Author: qtang@openailab.com
 */

#ifndef TENGINE_GRAPH_CONVOLUTION_HPP
#define TENGINE_GRAPH_CONVOLUTION_HPP

#define FLOAT_TO_REALSIZE (4)

namespace cv
{
namespace dnn
{
bool tengine_forward(float *input_, int inch, int group, int in_h, int in_w,
                        float *output_, int out_b, int outch, int out_h, int out_w,
                        float *kernel_,int kernel_s , int kernel_h, int kernel_w,
                        float *teg_bias, int stride_h,int stride_w,
                        int pad_h, int pad_w,  int dilation_h, int dilation_w,
                        size_t wstep, const std::string padMode) ;
}
}
#endif /* TENGINE_GRAPH_CONVOLUTION_HPP */
