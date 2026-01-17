#!/usr/bin/env python3
"""
@file resize_demo.py
@brief Demonstration of cv2.resize() function with different interpolation methods
@author OpenCV Documentation

This script demonstrates how to use cv2.resize() with various interpolation methods
and helps understand the differences between them.
"""

import cv2 as cv
import numpy as np
import sys
import argparse
import time

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='This program demonstrates different interpolation methods in cv2.resize()')
    parser.add_argument('--input', type=str, default='lena.jpg',
                        help='Path to input image')
    args = parser.parse_args()
    
    # Load the source image
    src = cv.imread(cv.samples.findFile(args.input))
    
    if src is None:
        print(f'Could not open or find the image: {args.input}')
        print('Usage: python resize_demo.py --input [image_path]')
        return -1
    
    print(f'Original image size: {src.shape[1]}x{src.shape[0]}')
    
    # Display original
    cv.namedWindow('Original', cv.WINDOW_AUTOSIZE)
    cv.imshow('Original', src)
    
    # 1. Resize by specifying output size (upscale by 2x)
    new_size = (src.shape[1] * 2, src.shape[0] * 2)
    dst_size = cv.resize(src, new_size, interpolation=cv.INTER_LINEAR)
    print(f'\n1. Resize by size (2x upscale with INTER_LINEAR):')
    print(f'   Output size: {dst_size.shape[1]}x{dst_size.shape[0]}')
    cv.namedWindow('2x Upscale (INTER_LINEAR)', cv.WINDOW_AUTOSIZE)
    cv.imshow('2x Upscale (INTER_LINEAR)', dst_size)
    
    # 2. Resize by scale factor (downscale by 0.5x)
    dst_scale = cv.resize(src, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    print(f'\n2. Resize by scale factor (0.5x downscale with INTER_AREA):')
    print(f'   Output size: {dst_scale.shape[1]}x{dst_scale.shape[0]}')
    cv.namedWindow('0.5x Downscale (INTER_AREA)', cv.WINDOW_AUTOSIZE)
    cv.imshow('0.5x Downscale (INTER_AREA)', dst_scale)
    
    # 3. Compare different interpolation methods for upscaling
    upscale_size = (src.shape[1] * 2, src.shape[0] * 2)
    
    dst_nearest = cv.resize(src, upscale_size, interpolation=cv.INTER_NEAREST)
    dst_linear = cv.resize(src, upscale_size, interpolation=cv.INTER_LINEAR)
    dst_cubic = cv.resize(src, upscale_size, interpolation=cv.INTER_CUBIC)
    dst_lanczos = cv.resize(src, upscale_size, interpolation=cv.INTER_LANCZOS4)
    
    print('\n3. Comparing interpolation methods (2x upscale):')
    print('   INTER_NEAREST - Fastest, lowest quality')
    print('   INTER_LINEAR  - Good balance (default)')
    print('   INTER_CUBIC   - Better quality, slower')
    print('   INTER_LANCZOS4 - Best quality, slowest')
    
    cv.namedWindow('INTER_NEAREST', cv.WINDOW_AUTOSIZE)
    cv.namedWindow('INTER_LINEAR', cv.WINDOW_AUTOSIZE)
    cv.namedWindow('INTER_CUBIC', cv.WINDOW_AUTOSIZE)
    cv.namedWindow('INTER_LANCZOS4', cv.WINDOW_AUTOSIZE)
    
    cv.imshow('INTER_NEAREST', dst_nearest)
    cv.imshow('INTER_LINEAR', dst_linear)
    cv.imshow('INTER_CUBIC', dst_cubic)
    cv.imshow('INTER_LANCZOS4', dst_lanczos)
    
    # 4. Compare different interpolation methods for downscaling
    downscale_size = (src.shape[1] // 2, src.shape[0] // 2)
    
    dst_down_nearest = cv.resize(src, downscale_size, interpolation=cv.INTER_NEAREST)
    dst_down_linear = cv.resize(src, downscale_size, interpolation=cv.INTER_LINEAR)
    dst_down_area = cv.resize(src, downscale_size, interpolation=cv.INTER_AREA)
    
    print('\n4. Comparing interpolation methods (0.5x downscale):')
    print('   INTER_NEAREST - Fastest, can cause artifacts')
    print('   INTER_LINEAR  - Good quality')
    print('   INTER_AREA    - Best for downscaling (recommended)')
    
    cv.namedWindow('Downscale INTER_NEAREST', cv.WINDOW_AUTOSIZE)
    cv.namedWindow('Downscale INTER_LINEAR', cv.WINDOW_AUTOSIZE)
    cv.namedWindow('Downscale INTER_AREA', cv.WINDOW_AUTOSIZE)
    
    cv.imshow('Downscale INTER_NEAREST', dst_down_nearest)
    cv.imshow('Downscale INTER_LINEAR', dst_down_linear)
    cv.imshow('Downscale INTER_AREA', dst_down_area)
    
    # 5. Demonstration of aspect ratio change
    dst_aspect = cv.resize(src, (src.shape[1] * 2, src.shape[0]), 
                           interpolation=cv.INTER_LINEAR)
    print(f'\n5. Non-uniform scaling (width 2x, height 1x):')
    print(f'   Output size: {dst_aspect.shape[1]}x{dst_aspect.shape[0]}')
    cv.namedWindow('Non-uniform Scale', cv.WINDOW_AUTOSIZE)
    cv.imshow('Non-uniform Scale', dst_aspect)
    
    # 6. Demonstrate timing comparison
    print('\n6. Performance comparison (1000 iterations):')
    iterations = 1000
    
    # Time INTER_NEAREST
    start = time.time()
    for _ in range(iterations):
        result = cv.resize(src, upscale_size, interpolation=cv.INTER_NEAREST)
    nearest_time = time.time() - start
    print(f'   INTER_NEAREST:  {nearest_time:.3f}s ({nearest_time/iterations*1000:.2f}ms per resize)')
    
    # Time INTER_LINEAR
    start = time.time()
    for _ in range(iterations):
        result = cv.resize(src, upscale_size, interpolation=cv.INTER_LINEAR)
    linear_time = time.time() - start
    print(f'   INTER_LINEAR:   {linear_time:.3f}s ({linear_time/iterations*1000:.2f}ms per resize)')
    
    # Time INTER_CUBIC
    start = time.time()
    for _ in range(iterations):
        result = cv.resize(src, upscale_size, interpolation=cv.INTER_CUBIC)
    cubic_time = time.time() - start
    print(f'   INTER_CUBIC:    {cubic_time:.3f}s ({cubic_time/iterations*1000:.2f}ms per resize)')
    
    # Time INTER_LANCZOS4
    start = time.time()
    for _ in range(iterations):
        result = cv.resize(src, upscale_size, interpolation=cv.INTER_LANCZOS4)
    lanczos_time = time.time() - start
    print(f'   INTER_LANCZOS4: {lanczos_time:.3f}s ({lanczos_time/iterations*1000:.2f}ms per resize)')
    
    print('\nPress any key to exit...')
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
