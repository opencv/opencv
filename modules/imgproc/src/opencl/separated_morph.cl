
// complile-time definitions
// THRESHOLD, RADIUS, POW_RADIUS

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

#define IS_SET(value) value[0] > 200

#define WITHIN(value, iter) value + pown((float)(iter), 2) < POW_RADIUS

__kernel void
HorizontalFilter(__read_only image2d_t inputImage,
                 __write_only image2d_t outputImage, unsigned int _radius) {
  const int2 currentPose = {get_global_id(0), get_global_id(1)};
  // iterate
  for (int ii = 0; ii != _radius; ++ii) {
    const int2 offset = {ii, 0};
    if (IS_SET(read_imageui(inputImage, sampler, currentPose + offset)) ||
        IS_SET(read_imageui(inputImage, sampler, currentPose - offset))) {
      write_imagef(outputImage, currentPose, (float4)(pown((float)(ii), 2), 0, 0, 0));
      return;
    }
  }
  write_imagef(outputImage, currentPose, (float4)(POW_RADIUS + 1, 0, 0, 0));
}

__kernel void
VerticalFilter(__read_only image2d_t inputImage,
               __write_only image2d_t outputImage, unsigned int _radius) {
  const int2 currentPose = (int2)(get_global_id(0), get_global_id(1));
  // iterate
  for (unsigned int ii = 0; ii != RADIUS; ++ii) {
    const int2 offset = {0, ii};
    if (WITHIN(read_imagef(inputImage, sampler, currentPose + offset)[0],
               ii) ||
        WITHIN(read_imagef(inputImage, sampler, currentPose - offset)[0],
               ii)) {
      write_imageui(outputImage, currentPose, (uint4)(0, 0, 0, 0));
      return;
    }
  }
  write_imageui(outputImage, currentPose, (uint4)(255, 0, 0, 0));
}