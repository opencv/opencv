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
// Copyright (c) 2016-2017 Fabian David Tschopp, all rights reserved.
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

#define Dtype float
#define Dtype4 float4

__kernel void DecodeBBoxesCORNER(const int nthreads,
                                 __global const Dtype* loc_data,
                                 __global const Dtype* prior_data,
                                 const int variance_encoded_in_target,
                                 const int num_priors,
                                 const int share_location,
                                 const int num_loc_classes,
                                 const int background_label_id,
                                 const int clip_bbox,
                                 const int locPredTransposed,
                                 __global Dtype* bbox_data)
{
    for (int index = get_global_id(0); index < nthreads; index += get_global_size(0))
    {
        Dtype bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax;
        const int i = index % 4;
        const int p = ((index / 4 / num_loc_classes) % num_priors) * 4;

        const int c = (index / 4) % num_loc_classes;
        int label = share_location ? -1 : c;
        if (label == background_label_id)
            return; // Ignore background class.

        Dtype4 loc_vec = vload4(0, loc_data + index - i);
        Dtype4 bbox_vec, prior_variance;
        if (variance_encoded_in_target)
        {
            bbox_vec = loc_vec;
        } else {
            const int start_index = num_priors * 4 + p;
            prior_variance = vload4(0, prior_data + start_index);
            bbox_vec = loc_vec * prior_variance;
        }

        if (locPredTransposed)
        {
            bbox_ymin = bbox_vec.x;
            bbox_xmin = bbox_vec.y;
            bbox_ymax = bbox_vec.z;
            bbox_xmax = bbox_vec.w;
        } else {
            bbox_xmin = bbox_vec.x;
            bbox_ymin = bbox_vec.y;
            bbox_xmax = bbox_vec.z;
            bbox_ymax = bbox_vec.w;
        }

        Dtype4 prior_vec = vload4(0, prior_data + p);
        Dtype val;
        switch (i)
        {
            case 0:
                val = prior_vec.x + bbox_xmin;
                break;
            case 1:
                val = prior_vec.y + bbox_ymin;
                break;
            case 2:
                val = prior_vec.z + bbox_xmax;
                break;
            case 3:
                val = prior_vec.w + bbox_ymax;
                break;
        }

        if (clip_bbox)
            val = max(min(val, (Dtype)1.), (Dtype)0.);

        bbox_data[index] = val;
    }
}

__kernel void DecodeBBoxesCENTER_SIZE(const int nthreads,
                                      __global const Dtype* loc_data,
                                      __global const Dtype* prior_data,
                                      const int variance_encoded_in_target,
                                      const int num_priors,
                                      const int share_location,
                                      const int num_loc_classes,
                                      const int background_label_id,
                                      const int clip_bbox,
                                      const int locPredTransposed,
                                      __global Dtype* bbox_data)
{
    for (int index = get_global_id(0); index < nthreads; index += get_global_size(0))
    {
        Dtype bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax;
        const int i = index % 4;
        const int p = ((index / 4 / num_loc_classes) % num_priors) * 4;

        const int c = (index / 4) % num_loc_classes;
        int label = share_location ? -1 : c;
        if (label == background_label_id)
            return; // Ignore background class.

        Dtype4 loc_vec = vload4(0, loc_data + index - i);
        Dtype4 bbox_vec, prior_variance;
        if (variance_encoded_in_target)
        {
            bbox_vec = loc_vec;
        } else {
            const int start_index = num_priors * 4 + p;
            prior_variance = vload4(0, prior_data + start_index);
            bbox_vec = loc_vec * prior_variance;
        }

        if (locPredTransposed)
        {
            bbox_ymin = bbox_vec.x;
            bbox_xmin = bbox_vec.y;
            bbox_ymax = bbox_vec.z;
            bbox_xmax = bbox_vec.w;
        } else {
            bbox_xmin = bbox_vec.x;
            bbox_ymin = bbox_vec.y;
            bbox_xmax = bbox_vec.z;
            bbox_ymax = bbox_vec.w;
        }

        Dtype4 prior_vec = vload4(0, prior_data + p);
        Dtype prior_width = prior_vec.z - prior_vec.x;
        Dtype prior_height = prior_vec.w - prior_vec.y;
        Dtype prior_center_x = (prior_vec.x + prior_vec.z) * .5;
        Dtype prior_center_y = (prior_vec.y + prior_vec.w) * .5;

        Dtype decode_bbox_center_x, decode_bbox_center_y;
        Dtype decode_bbox_width, decode_bbox_height;
        decode_bbox_center_x = bbox_xmin * prior_width + prior_center_x;
        decode_bbox_center_y = bbox_ymin * prior_height + prior_center_y;
        decode_bbox_width = exp(bbox_xmax) * prior_width;
        decode_bbox_height = exp(bbox_ymax) * prior_height;

        Dtype val;
        switch (i)
        {
            case 0:
                val = decode_bbox_center_x - decode_bbox_width * .5;
                break;
            case 1:
                val = decode_bbox_center_y - decode_bbox_height * .5;
                break;
            case 2:
                val = decode_bbox_center_x + decode_bbox_width * .5;
                break;
            case 3:
                val = decode_bbox_center_y + decode_bbox_height * .5;
                break;
        }

        if (clip_bbox)
            val = max(min(val, (Dtype)1.), (Dtype)0.);

        bbox_data[index] = val;
    }
}
