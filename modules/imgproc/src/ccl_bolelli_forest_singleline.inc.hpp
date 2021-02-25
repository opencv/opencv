/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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
// 2021 Federico Bolelli <federico.bolelli@unimore.it>
// 2021 Stefano Allegretti <stefano.allegretti@unimore.it>
// 2021 Costantino Grana <costantino.grana@unimore.it>
//
// This file has been automatically generated using GRAPHGEN (https://github.com/prittt/GRAPHGEN)
// and taken from the YACCLAB repo (https://github.com/prittt/YACCLAB).
//M*/
//
sl_tree_0: if ((c+=2) >= w - 2) { if (c > w - 2) { goto sl_break_0_0; } else { goto sl_break_1_0; } }
        if (CONDITION_O) {
            if (CONDITION_P) {
                ACTION_2
                goto sl_tree_1;
            }
            else {
                ACTION_2
                goto sl_tree_0;
            }
        }
        else {
            NODE_372:
            if (CONDITION_P) {
                ACTION_2
                goto sl_tree_1;
            }
            else {
                ACTION_1
                goto sl_tree_0;
            }
        }
sl_tree_1: if ((c+=2) >= w - 2) { if (c > w - 2) { goto sl_break_0_1; } else { goto sl_break_1_1; } }
        if (CONDITION_O) {
            if (CONDITION_P) {
                ACTION_6
                goto sl_tree_1;
            }
            else {
                ACTION_6
                goto sl_tree_0;
            }
        }
        else{
            goto NODE_372;
        }
sl_break_0_0:
        if (CONDITION_O) {
            ACTION_2
        }
        else {
            ACTION_1
        }
    goto end_sl;
sl_break_0_1:
        if (CONDITION_O) {
            ACTION_6
        }
        else {
            ACTION_1
        }
    goto end_sl;
sl_break_1_0:
        if (CONDITION_O) {
            if (CONDITION_P) {
                ACTION_2
            }
            else {
                ACTION_2
            }
        }
        else {
            NODE_375:
            if (CONDITION_P) {
                ACTION_2
            }
            else {
                ACTION_1
            }
        }
    goto end_sl;
sl_break_1_1:
        if (CONDITION_O) {
            if (CONDITION_P) {
                ACTION_6
            }
            else {
                ACTION_6
            }
        }
        else{
            goto NODE_375;
        }
    goto end_sl;
end_sl:;
