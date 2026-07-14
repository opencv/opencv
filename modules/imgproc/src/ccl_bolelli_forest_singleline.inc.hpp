// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// 2021 Federico Bolelli <federico.bolelli@unimore.it>
// 2021 Stefano Allegretti <stefano.allegretti@unimore.it>
// 2021 Costantino Grana <costantino.grana@unimore.it>
//
// This file has been automatically generated using GRAPHGEN (https://github.com/prittt/GRAPHGEN)
// and taken from the YACCLAB repository (https://github.com/prittt/YACCLAB).
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
