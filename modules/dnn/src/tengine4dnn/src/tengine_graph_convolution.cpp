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

#include "../../precomp.hpp"
#include <iostream>
#include <vector>

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "../include/tengine_graph_convolution.hpp"

#ifdef HAVE_TENGINE

#include "tengine_c_api.h"


namespace cv
{
namespace dnn
{
static int create_input_node(teng_graph_t graph, const char* node_name, int inch, int in_h, int in_w)
{
    node_t node     = teng_create_graph_node(graph, node_name, "InputOp");
    tensor_t tensor = teng_create_graph_tensor(graph, node_name, TENGINE_DT_FP32);
    teng_set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    int dims[4] = {1, inch, in_h, in_w};
    teng_set_tensor_shape(tensor, dims, 4);

    teng_release_graph_tensor(tensor);
    teng_release_graph_node(node);

    return 0;
}

static int create_conv_node(teng_graph_t graph, const char* node_name, const char* input_name, int in_h, int in_w, int out_h, int out_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h0, int pad_h1, int pad_w0, int pad_w1, int inch, int outch, int group,
    int dilation_h, int dilation_w, int activation, std::string padMode)
{
    node_t conv_node      = teng_create_graph_node(graph, node_name, "Convolution");
    tensor_t input_tensor = teng_get_graph_tensor(graph, input_name);

    if (input_tensor == NULL)
    {
        CV_LOG_WARNING(NULL,"Tengine: input_tensor is NULL." );
        return -1;
    }

    teng_set_node_input_tensor(conv_node, 0, input_tensor);
    teng_release_graph_tensor(input_tensor);

    /* output */
    tensor_t output_tensor = teng_create_graph_tensor(graph, node_name, TENGINE_DT_FP32);

    teng_set_node_output_tensor(conv_node, 0, output_tensor, TENSOR_TYPE_VAR);
    teng_release_graph_tensor(output_tensor);

    /* weight */
    std::string weight_name(node_name);
    weight_name += "/weight";

    node_t w_node = teng_create_graph_node(graph, weight_name.c_str(), "Const");
    tensor_t w_tensor = teng_create_graph_tensor(graph, weight_name.c_str(), TENGINE_DT_FP32);
    teng_set_node_output_tensor(w_node, 0, w_tensor, TENSOR_TYPE_CONST);
    teng_set_node_input_tensor(conv_node, 1, w_tensor);
    int w_dims[] = {outch, inch / group, kernel_h, kernel_w};

    teng_set_tensor_shape(w_tensor, w_dims, 4);

    teng_release_graph_node(w_node);
    teng_release_graph_tensor(w_tensor);

    /* bias */
    std::string bias_name(node_name);
    bias_name += "/bias";

    node_t b_node = teng_create_graph_node(graph, bias_name.c_str(), "Const");
    tensor_t b_tensor = teng_create_graph_tensor(graph, bias_name.c_str(), TENGINE_DT_FP32);
    teng_set_node_output_tensor(b_node, 0, b_tensor, TENSOR_TYPE_CONST);
    int b_dims[] = {outch};

    teng_set_tensor_shape(b_tensor, b_dims, 1);

    teng_set_node_input_tensor(conv_node, 2, b_tensor);
    teng_release_graph_node(b_node);
    teng_release_graph_tensor(b_tensor);

    if (!padMode.empty())
    {
        if (padMode == "SAME")
        {
            int out_h_temp = (in_h-kernel_h + 2*pad_h0)/stride_h + 1;
            int out_w_temp = (in_w-kernel_w + 2*pad_w0)/stride_w + 1;

            if (out_h_temp < out_h)
                pad_h1 += 1;
            if (out_w_temp < out_w)
                pad_w1 += 1;
        }
    }

    /* attr */
    teng_set_node_attr_int(conv_node, "kernel_h", &kernel_h);
    teng_set_node_attr_int(conv_node, "kernel_w", &kernel_w);
    teng_set_node_attr_int(conv_node, "stride_h", &stride_h);
    teng_set_node_attr_int(conv_node, "stride_w", &stride_w);
    teng_set_node_attr_int(conv_node, "pad_h0", &pad_h0);
    teng_set_node_attr_int(conv_node, "pad_w0", &pad_w0);
    teng_set_node_attr_int(conv_node, "pad_h1", &pad_h1);
    teng_set_node_attr_int(conv_node, "pad_w1", &pad_w1);
    teng_set_node_attr_int(conv_node, "output_channel", &outch);
    teng_set_node_attr_int(conv_node, "input_channel", &inch);
    teng_set_node_attr_int(conv_node, "group", &group);
    teng_set_node_attr_int(conv_node, "dilation_h", &dilation_h);
    teng_set_node_attr_int(conv_node, "dilation_w", &dilation_w);
  //  set_node_attr_int(conv_node, "activation", &activation);

    teng_release_graph_node(conv_node);

    return 0;
}

static teng_graph_t create_conv_graph(const char* layer_name, float* input_data, int inch, int group, int in_h, int in_w,
                        float* output_data, int outch, int out_h, int out_w,
                        int kernel_h, int kernel_w,
                        int stride_h,int stride_w,
                        int pad_h0, int pad_h1, int pad_w0, int pad_w1, int dilation_h, int dilation_w, int activation,
                        float* teg_weight, float* teg_bias, std::string padMode, int nstripes)
{
    node_t    conv_node     = NULL;

    tensor_t  input_tensor  = NULL;
    tensor_t  output_tensor = NULL;
    tensor_t  weight_tensor = NULL;
    tensor_t  bias_tensor   = NULL;

    /* create graph for convolution */
    int in_size  = in_h * in_w * inch;
    int out_size  = out_h * out_w * outch;
    int weight_size = outch * (inch / group) * kernel_w * kernel_h;
    int bias_size = outch;

    int buf_size  = 0;
    int input_num = 0;

    /* create graph */
    teng_graph_t graph = teng_create_graph(NULL, NULL, NULL);
    bool ok = true;

    if(graph == NULL)
    {
        CV_LOG_WARNING(NULL,"Tengine: create_graph failed." );
        ok = false;
    }

    const char* input_name = "data";
    const char* conv_name  = layer_name;

    if (ok && create_input_node(graph, input_name, inch, in_h, in_w) < 0)
    {
        CV_LOG_WARNING(NULL,"Tengine: create_input_node failed." );
        ok = false;
    }

    if (ok && create_conv_node(graph, conv_name, input_name, in_h, in_w, out_h, out_w, kernel_h, kernel_w,
        stride_h, stride_w, pad_h0, pad_h1, pad_w0, pad_w1, inch, outch, group, dilation_h, dilation_w, activation, padMode) < 0)
    {
        CV_LOG_WARNING(NULL,"Tengine: create conv node failed." );
        ok = false;
    }

    /* set input/output node */
    const char* inputs_name[]  = {input_name};
    const char* outputs_name[] = {conv_name};

    if (ok && teng_set_graph_input_node(graph, inputs_name, sizeof(inputs_name) / sizeof(char*)) < 0)
    {
        CV_LOG_WARNING(NULL,"Tengine: set inputs failed." );
        ok = false;
    }

    if (ok && teng_set_graph_output_node(graph, outputs_name, sizeof(outputs_name) / sizeof(char*)) < 0)
    {
        CV_LOG_WARNING(NULL,"Tengine: set outputs failed." );
        ok = false;
    }

    /* set input data */
    if (ok)
    {
        input_tensor = teng_get_graph_input_tensor(graph, 0, 0);
        buf_size     = teng_get_tensor_buffer_size(input_tensor);
        if (buf_size != in_size * FLOAT_TO_REALSIZE)
        {
            CV_LOG_WARNING(NULL,"Tengine: Input data size check failed.");
            ok = false;
        }
    }

    if (ok)
    {
        teng_set_tensor_buffer(input_tensor, (float *)input_data, buf_size);
        teng_release_graph_tensor(input_tensor);

        /* create convolution node */
        /* set weight node */
        conv_node     = teng_get_graph_node(graph, conv_name);
        weight_tensor = teng_get_node_input_tensor(conv_node, 1);
        buf_size      = teng_get_tensor_buffer_size(weight_tensor);

        if (buf_size != weight_size * FLOAT_TO_REALSIZE)
        {
            CV_LOG_WARNING(NULL,"Tengine: Input weight size check failed.");
            ok = false;
        }
    }

    if (ok)
    {
        teng_set_tensor_buffer(weight_tensor, teg_weight, buf_size);

        /* set bias node */
        input_num = teng_get_node_input_number(conv_node);
        if (input_num > 2)
        {
            bias_tensor = teng_get_node_input_tensor(conv_node, 2);
            buf_size    = teng_get_tensor_buffer_size(bias_tensor);
            if (buf_size != bias_size * FLOAT_TO_REALSIZE)
            {
                CV_LOG_WARNING(NULL,"Tengine: Input bias size check failed.");
                ok = false;
            }
            else teng_set_tensor_buffer(bias_tensor, teg_bias, buf_size);
        }
    }

    /* prerun */
    if (ok && teng_prerun_graph_multithread(graph, TENGINE_CLUSTER_BIG, nstripes) < 0)
    {
        CV_LOG_WARNING(NULL, "Tengine: prerun_graph failed.");
        ok = false;
    }

    if (ok)
    {
        /* set output data */
        output_tensor = teng_get_node_output_tensor(conv_node, 0);
        int ret = teng_set_tensor_buffer(output_tensor, output_data, out_size * FLOAT_TO_REALSIZE);
        if(ret)
        {
            CV_LOG_WARNING(NULL,"Tengine: Set output tensor buffer failed." );
            ok = false;
        }
    }

    if (false == ok)
    {
        teng_destroy_graph(graph) ;
        return NULL ;
    }
    return graph;
}
static bool tengine_init_flag = false;
teng_graph_t tengine_init(const char* layer_name, float* input_, int inch, int group, int in_h, int in_w,
                        float *output_, int out_b, int outch, int out_h, int out_w,
                        float *kernel_, int kernel_s ,int kernel_h, int kernel_w,
                        float *teg_bias, int stride_h, int stride_w,
                        int pad_h0, int pad_h1, int pad_w0, int pad_w1, int dilation_h, int dilation_w,
                        size_t wstep, const std::string padMode, teng_graph_t &graph, int nstripes)
{
    std::vector<float> teg_weight_vec;
    float *teg_weight = NULL;
    int kernel_inwh = (inch / group) * kernel_w * kernel_h;
    // Do not using the activation fuse mode, just convolution only.
    int activation = -1;

    if (!(kernel_s == 2 && kernel_h == kernel_w
        && dilation_h == dilation_w && stride_h == stride_w
        && out_b == 1 && pad_h0 < 10 && pad_h1 < 10 && pad_w0 < 10 && pad_w1 < 10)) // just for Conv2D
    {
       // printf("return : just for Conv2D\n");
        return NULL;
    }

    {
      /*   printf("Tengine(%s): input (1 x %d x %d x %d),output (%d x %d x %d x %d), kernel (%d x %d), stride (%d x %d), dilation (%d x %d), pad (%d x %d).\n",
               layer_name, inch, in_h, in_w,
               out_b, outch, out_h, out_w,
               kernel_w, kernel_h,
               stride_w, stride_h,
               dilation_w, dilation_h,
               pad_h0, pad_h1, pad_w0, pad_w1);
     */
        // weight
        if (kernel_inwh != wstep)
        {
            teg_weight_vec.resize(kernel_inwh * outch);
            teg_weight = &teg_weight_vec[0];
            for (int i=0; i<outch; i++)
            {
                memcpy(teg_weight+i*kernel_inwh, kernel_+i*wstep, kernel_inwh*FLOAT_TO_REALSIZE);
            }
        }
        else
        {
            teg_weight = kernel_;
        }

        /* initial the resource of tengine */
        if(false == tengine_init_flag)
        {
            init_tengine();
            tengine_init_flag = true;
        }

        /* create the convolution graph */
        graph = create_conv_graph(layer_name, input_, inch, group, in_h, in_w,
                                    output_, outch, out_h, out_w,
                                    kernel_h, kernel_w, stride_h,stride_w,
                                    pad_h0, pad_h1, pad_w0, pad_w1, dilation_h, dilation_w, activation,
                                    teg_weight, teg_bias, padMode, nstripes);
        if(NULL == graph )
        {
            return NULL;
        }
    }
    return graph ;
}

bool tengine_forward(teng_graph_t &graph)
{
    /* run */
    if(teng_run_graph(graph, 1) < 0)
    {
        CV_LOG_WARNING(NULL,"Tengine: run_graph failed.");
        return false ;
    }
    return true;
}
bool tengine_release(teng_graph_t &graph)
{
    teng_postrun_graph(graph);
    teng_destroy_graph(graph);
    return true;
}
}
}
#endif
