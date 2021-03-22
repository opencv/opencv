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
fl_tree_0: if ((c+=2) >= w - 2) { if (c > w - 2) { goto fl_break_0_0; } else { goto fl_break_1_0; } }
        if (CONDITION_O) {
            NODE_253:
            if (CONDITION_P) {
                ACTION_2
                goto fl_tree_1;
            }
            else {
                ACTION_2
                goto fl_tree_2;
            }
        }
        else {
            if (CONDITION_S){
                goto NODE_253;
            }
            else {
                NODE_255:
                if (CONDITION_P) {
                    ACTION_2
                    goto fl_tree_1;
                }
                else {
                    if (CONDITION_T) {
                        ACTION_2
                        goto fl_tree_1;
                    }
                    else {
                        ACTION_1
                        goto fl_tree_0;
                    }
                }
            }
        }
fl_tree_1: if ((c+=2) >= w - 2) { if (c > w - 2) { goto fl_break_0_1; } else { goto fl_break_1_1; } }
        if (CONDITION_O) {
            NODE_257:
            if (CONDITION_P) {
                ACTION_6
                goto fl_tree_1;
            }
            else {
                ACTION_6
                goto fl_tree_2;
            }
        }
        else {
            if (CONDITION_S){
                goto NODE_257;
            }
            else{
                goto NODE_255;
            }
        }
fl_tree_2: if ((c+=2) >= w - 2) { if (c > w - 2) { goto fl_break_0_2; } else { goto fl_break_1_2; } }
        if (CONDITION_O) {
            if (CONDITION_R){
                goto NODE_257;
            }
            else{
                goto NODE_253;
            }
        }
        else {
            if (CONDITION_S) {
                if (CONDITION_P) {
                    if (CONDITION_R) {
                        ACTION_6
                        goto fl_tree_1;
                    }
                    else {
                        ACTION_2
                        goto fl_tree_1;
                    }
                }
                else {
                    if (CONDITION_R) {
                        ACTION_6
                        goto fl_tree_2;
                    }
                    else {
                        ACTION_2
                        goto fl_tree_2;
                    }
                }
            }
            else{
                goto NODE_255;
            }
        }
fl_break_0_0:
        if (CONDITION_O) {
            ACTION_2
        }
        else {
            if (CONDITION_S) {
                ACTION_2
            }
            else {
                ACTION_1
            }
        }
    goto end_fl;
fl_break_0_1:
        if (CONDITION_O) {
            ACTION_6
        }
        else {
            if (CONDITION_S) {
                ACTION_6
            }
            else {
                ACTION_1
            }
        }
    goto end_fl;
fl_break_0_2:
        if (CONDITION_O) {
            NODE_266:
            if (CONDITION_R) {
                ACTION_6
            }
            else {
                ACTION_2
            }
        }
        else {
            if (CONDITION_S){
                goto NODE_266;
            }
            else {
                ACTION_1
            }
        }
    goto end_fl;
fl_break_1_0:
        if (CONDITION_O) {
            NODE_268:
            if (CONDITION_P) {
                ACTION_2
            }
            else {
                ACTION_2
            }
        }
        else {
            if (CONDITION_S){
                goto NODE_268;
            }
            else {
                NODE_270:
                if (CONDITION_P) {
                    ACTION_2
                }
                else {
                    if (CONDITION_T) {
                        ACTION_2
                    }
                    else {
                        ACTION_1
                    }
                }
            }
        }
    goto end_fl;
fl_break_1_1:
        if (CONDITION_O) {
            NODE_272:
            if (CONDITION_P) {
                ACTION_6
            }
            else {
                ACTION_6
            }
        }
        else {
            if (CONDITION_S){
                goto NODE_272;
            }
            else{
                goto NODE_270;
            }
        }
    goto end_fl;
fl_break_1_2:
        if (CONDITION_O) {
            if (CONDITION_R){
                goto NODE_272;
            }
            else{
                goto NODE_268;
            }
        }
        else {
            if (CONDITION_S) {
                if (CONDITION_P){
                    goto NODE_266;
                }
                else{
                    goto NODE_266;
                }
            }
            else{
                goto NODE_270;
            }
        }
    goto end_fl;
end_fl:;
