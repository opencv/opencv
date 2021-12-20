//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Weta Digital, Ltd and Contributors to the OpenEXR Project.
//

#include "ImfDeepCompositing.h"

#include "ImfNamespace.h"
#include <algorithm>
#include <vector>

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using std::sort;
using std::vector;
DeepCompositing::DeepCompositing()
{
}

DeepCompositing::~DeepCompositing()
{
}

void 
DeepCompositing::composite_pixel (float outputs[],
                                  const float* inputs[],
                                  const char*channel_names[],
                                  int num_channels,
                                  int num_samples,
                                  int sources)
{
    for(int i=0;i<num_channels;i++) outputs[i]=0.0;
    // no samples? do nothing
   if(num_samples==0)
   {
       return;
   }
   
   vector<int> sort_order;
   if(sources>1)
   {
       sort_order.resize(num_samples);
       for(int i=0;i<num_samples;i++) sort_order[i]=i;
       sort(&sort_order[0],inputs,channel_names,num_channels,num_samples,sources);
   }
   
   
   for(int i=0;i<num_samples;i++)
   {
       int s=(sources>1) ? sort_order[i] : i;
       float alpha=outputs[2]; 
       if(alpha>=1.0f) return;
       
       for(int c=0;c<num_channels;c++)
       {
           outputs[c]+=(1.0f-alpha)*inputs[c][s];
       }
   }   
}

struct sort_helper
{
    const float ** inputs;
    bool operator() (int a,int b) 
    {
        if(inputs[0][a] < inputs[0][b]) return true;
        if(inputs[0][a] > inputs[0][b]) return false;
        if(inputs[1][a] < inputs[1][b]) return true;
        if(inputs[1][a] > inputs[1][b]) return false;
        return a<b;
    }
    sort_helper(const float ** i) : inputs(i) {}
};

void
DeepCompositing::sort(int order[], const float* inputs[], const char* channel_names[], int num_channels, int num_samples, int sources)
{
  std::sort(order+0,order+num_samples,sort_helper(inputs));
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
