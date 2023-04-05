/* -*- C++ -*-
 * Copyright 2019-2021 LibRaw LLC (info@libraw.org)
 *
 LibRaw is free software; you can redistribute it and/or modify
 it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

 */

#include "../../internal/libraw_cxx_defs.h"

const double LibRaw_constants::xyz_rgb[3][3] = {
    {0.4124564, 0.3575761, 0.1804375},
    {0.2126729, 0.7151522, 0.0721750},
    {0.0193339, 0.1191920, 0.9503041}};

const float LibRaw_constants::d65_white[3] = {0.95047f, 1.0f, 1.08883f};

const double LibRaw_constants::xyzd50_srgb[3][3] = {
    {0.436083, 0.385083, 0.143055},
    {0.222507, 0.716888, 0.060608},
    {0.013930, 0.097097, 0.714022}};
const double LibRaw_constants::rgb_rgb[3][3] = {
    {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
const double LibRaw_constants::adobe_rgb[3][3] = {
    {0.715146, 0.284856, 0.000000},
    {0.000000, 1.000000, 0.000000},
    {0.000000, 0.041166, 0.958839}};
const double LibRaw_constants::wide_rgb[3][3] = {
    {0.593087, 0.404710, 0.002206},
    {0.095413, 0.843149, 0.061439},
    {0.011621, 0.069091, 0.919288}};
const double LibRaw_constants::prophoto_rgb[3][3] = {
    {0.529317, 0.330092, 0.140588},
    {0.098368, 0.873465, 0.028169},
    {0.016879, 0.117663, 0.865457}};
const double LibRaw_constants::aces_rgb[3][3] = {
// Coffin's values (commented out) are not adapted from ACES "D60-like" WP to D65
//     {0.432996, 0.375380, 0.189317},
//     {0.089427, 0.816523, 0.102989},
//     {0.019165, 0.118150, 0.941914}};
    {0.43968015, 0.38295299, 0.17736686},
    {0.08978964, 0.81343316, 0.09677734},
    {0.01754827, 0.11156156, 0.87089017}};
const double LibRaw_constants::dcip3d65_rgb[3][3] = {
    {0.822488, 0.177511, 0.000000},
    {0.033200, 0.966800, 0.000000},
    {0.017089, 0.072411, 0.910499}};
const double LibRaw_constants::rec2020_rgb[3][3] = {
    {0.627452, 0.329249, 0.043299},
    {0.069109, 0.919531, 0.011360},
    {0.016398, 0.088030, 0.895572}};
