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
ll_tree_0: if ((c+=2) >= w - 2) { if (c > w - 2) { goto ll_break_0_0; } else { goto ll_break_1_0; } }
        if (CONDITION_O) {
            if (CONDITION_J) {
                ACTION_4
                goto ll_tree_6;
            }
            else {
                if (CONDITION_P) {
                    NODE_277:
                    if (CONDITION_K) {
                        if (CONDITION_I) {
                            NODE_279:
                            if (CONDITION_D) {
                                ACTION_5
                                goto ll_tree_4;
                            }
                            else {
                                ACTION_10
                                goto ll_tree_4;
                            }
                        }
                        else {
                            ACTION_5
                            goto ll_tree_4;
                        }
                    }
                    else {
                        if (CONDITION_I) {
                            ACTION_4
                            goto ll_tree_3;
                        }
                        else {
                            ACTION_2
                            goto ll_tree_2;
                        }
                    }
                }
                else {
                    if (CONDITION_I) {
                        ACTION_4
                        goto ll_tree_0;
                    }
                    else {
                        ACTION_2
                        goto ll_tree_0;
                    }
                }
            }
        }
        else {
            NODE_282:
            if (CONDITION_P) {
                if (CONDITION_J) {
                    ACTION_4
                    goto ll_tree_5;
                }
                else{
                    goto NODE_277;
                }
            }
            else {
                ACTION_1
                goto ll_tree_1;
            }
        }
ll_tree_1: if ((c+=2) >= w - 2) { if (c > w - 2) { goto ll_break_0_1; } else { goto ll_break_1_1; } }
        if (CONDITION_O) {
            if (CONDITION_J) {
                if (CONDITION_I) {
                    ACTION_4
                    goto ll_tree_6;
                }
                else {
                    if (CONDITION_H) {
                        NODE_287:
                        if (CONDITION_C) {
                            ACTION_4
                            goto ll_tree_6;
                        }
                        else {
                            ACTION_7
                            goto ll_tree_6;
                        }
                    }
                    else {
                        ACTION_4
                        goto ll_tree_6;
                    }
                }
            }
            else {
                if (CONDITION_P) {
                    if (CONDITION_K) {
                        if (CONDITION_I){
                            goto NODE_279;
                        }
                        else {
                            if (CONDITION_H) {
                                NODE_292:
                                if (CONDITION_D) {
                                    if (CONDITION_C) {
                                        ACTION_5
                                        goto ll_tree_4;
                                    }
                                    else {
                                        ACTION_8
                                        goto ll_tree_4;
                                    }
                                }
                                else {
                                    ACTION_8
                                    goto ll_tree_4;
                                }
                            }
                            else {
                                ACTION_5
                                goto ll_tree_4;
                            }
                        }
                    }
                    else {
                        if (CONDITION_I) {
                            ACTION_4
                            goto ll_tree_3;
                        }
                        else {
                            if (CONDITION_H) {
                                ACTION_3
                                goto ll_tree_2;
                            }
                            else {
                                ACTION_2
                                goto ll_tree_2;
                            }
                        }
                    }
                }
                else {
                    if (CONDITION_I) {
                        ACTION_4
                        goto ll_tree_0;
                    }
                    else {
                        if (CONDITION_H) {
                            ACTION_3
                            goto ll_tree_0;
                        }
                        else {
                            ACTION_2
                            goto ll_tree_0;
                        }
                    }
                }
            }
        }
        else{
            goto NODE_282;
        }
ll_tree_2: if ((c+=2) >= w - 2) { if (c > w - 2) { goto ll_break_0_2; } else { goto ll_break_1_2; } }
        if (CONDITION_O) {
            if (CONDITION_J) {
                ACTION_11
                goto ll_tree_6;
            }
            else {
                if (CONDITION_P) {
                    if (CONDITION_K) {
                        ACTION_12
                        goto ll_tree_4;
                    }
                    else {
                        ACTION_6
                        goto ll_tree_7;
                    }
                }
                else {
                    ACTION_6
                    goto ll_tree_0;
                }
            }
        }
        else {
            NODE_301:
            if (CONDITION_P) {
                if (CONDITION_J) {
                    ACTION_4
                    goto ll_tree_5;
                }
                else {
                    if (CONDITION_K) {
                        ACTION_5
                        goto ll_tree_4;
                    }
                    else {
                        ACTION_2
                        goto ll_tree_2;
                    }
                }
            }
            else {
                ACTION_1
                goto ll_tree_1;
            }
        }
ll_tree_3: if ((c+=2) >= w - 2) { if (c > w - 2) { goto ll_break_0_2; } else { goto ll_break_1_3; } }
        if (CONDITION_O) {
            if (CONDITION_J) {
                if (CONDITION_C) {
                    NODE_306:
                    if (CONDITION_B) {
                        ACTION_4
                        goto ll_tree_6;
                    }
                    else {
                        ACTION_7
                        goto ll_tree_6;
                    }
                }
                else {
                    ACTION_11
                    goto ll_tree_6;
                }
            }
            else {
                if (CONDITION_P) {
                    if (CONDITION_K) {
                        if (CONDITION_D) {
                            if (CONDITION_C) {
                                NODE_311:
                                if (CONDITION_B) {
                                    ACTION_5
                                    goto ll_tree_4;
                                }
                                else {
                                    ACTION_12
                                    goto ll_tree_4;
                                }
                            }
                            else {
                                ACTION_12
                                goto ll_tree_4;
                            }
                        }
                        else {
                            ACTION_12
                            goto ll_tree_4;
                        }
                    }
                    else {
                        ACTION_6
                        goto ll_tree_7;
                    }
                }
                else {
                    ACTION_6
                    goto ll_tree_0;
                }
            }
        }
        else{
            goto NODE_301;
        }
ll_tree_4: if ((c+=2) >= w - 2) { if (c > w - 2) { goto ll_break_0_2; } else { goto ll_break_1_4; } }
        if (CONDITION_O) {
            if (CONDITION_J) {
                ACTION_4
                goto ll_tree_6;
            }
            else {
                if (CONDITION_P) {
                    if (CONDITION_K) {
                        if (CONDITION_D) {
                            ACTION_5
                            goto ll_tree_4;
                        }
                        else {
                            ACTION_12
                            goto ll_tree_4;
                        }
                    }
                    else {
                        ACTION_6
                        goto ll_tree_7;
                    }
                }
                else {
                    ACTION_6
                    goto ll_tree_0;
                }
            }
        }
        else {
            if (CONDITION_P) {
                if (CONDITION_J) {
                    ACTION_4
                    goto ll_tree_5;
                }
                else {
                    if (CONDITION_K){
                        goto NODE_279;
                    }
                    else {
                        ACTION_4
                        goto ll_tree_3;
                    }
                }
            }
            else {
                ACTION_1
                goto ll_tree_1;
            }
        }
ll_tree_5: if ((c+=2) >= w - 2) { if (c > w - 2) { goto ll_break_0_2; } else { goto ll_break_1_5; } }
        if (CONDITION_O) {
            NODE_319:
            if (CONDITION_J) {
                if (CONDITION_I) {
                    ACTION_4
                    goto ll_tree_6;
                }
                else {
                    if (CONDITION_C) {
                        ACTION_4
                        goto ll_tree_6;
                    }
                    else {
                        ACTION_11
                        goto ll_tree_6;
                    }
                }
            }
            else {
                if (CONDITION_P) {
                    if (CONDITION_K) {
                        if (CONDITION_D) {
                            if (CONDITION_I) {
                                ACTION_5
                                goto ll_tree_4;
                            }
                            else {
                                if (CONDITION_C) {
                                    ACTION_5
                                    goto ll_tree_4;
                                }
                                else {
                                    ACTION_12
                                    goto ll_tree_4;
                                }
                            }
                        }
                        else {
                            ACTION_12
                            goto ll_tree_4;
                        }
                    }
                    else {
                        ACTION_6
                        goto ll_tree_7;
                    }
                }
                else {
                    ACTION_6
                    goto ll_tree_0;
                }
            }
        }
        else{
            goto NODE_282;
        }
ll_tree_6: if ((c+=2) >= w - 2) { if (c > w - 2) { goto ll_break_0_3; } else { goto ll_break_1_6; } }
        if (CONDITION_O) {
            if (CONDITION_N){
                goto NODE_319;
            }
            else {
                if (CONDITION_J) {
                    if (CONDITION_I) {
                        ACTION_4
                        goto ll_tree_6;
                    }
                    else{
                        goto NODE_287;
                    }
                }
                else {
                    if (CONDITION_P) {
                        if (CONDITION_K) {
                            if (CONDITION_I){
                                goto NODE_279;
                            }
                            else{
                                goto NODE_292;
                            }
                        }
                        else {
                            if (CONDITION_I) {
                                ACTION_4
                                goto ll_tree_3;
                            }
                            else {
                                ACTION_3
                                goto ll_tree_2;
                            }
                        }
                    }
                    else {
                        if (CONDITION_I) {
                            ACTION_4
                            goto ll_tree_0;
                        }
                        else {
                            ACTION_3
                            goto ll_tree_0;
                        }
                    }
                }
            }
        }
        else{
            goto NODE_282;
        }
ll_tree_7: if ((c+=2) >= w - 2) { if (c > w - 2) { goto ll_break_0_2; } else { goto ll_break_1_7; } }
        if (CONDITION_O) {
            if (CONDITION_J) {
                if (CONDITION_C) {
                    if (CONDITION_G){
                        goto NODE_306;
                    }
                    else {
                        ACTION_11
                        goto ll_tree_6;
                    }
                }
                else {
                    ACTION_11
                    goto ll_tree_6;
                }
            }
            else {
                if (CONDITION_P) {
                    if (CONDITION_K) {
                        if (CONDITION_D) {
                            if (CONDITION_C) {
                                if (CONDITION_G){
                                    goto NODE_311;
                                }
                                else {
                                    ACTION_12
                                    goto ll_tree_4;
                                }
                            }
                            else {
                                ACTION_12
                                goto ll_tree_4;
                            }
                        }
                        else {
                            ACTION_12
                            goto ll_tree_4;
                        }
                    }
                    else {
                        ACTION_6
                        goto ll_tree_7;
                    }
                }
                else {
                    ACTION_6
                    goto ll_tree_0;
                }
            }
        }
        else{
            goto NODE_301;
        }
ll_break_0_0:
        if (CONDITION_O) {
            NODE_343:
            if (CONDITION_I) {
                ACTION_4
            }
            else {
                ACTION_2
            }
        }
        else {
            ACTION_1
        }
    goto ll_end;
ll_break_0_1:
        if (CONDITION_O) {
            NODE_344:
            if (CONDITION_I) {
                ACTION_4
            }
            else {
                if (CONDITION_H) {
                    ACTION_3
                }
                else {
                    ACTION_2
                }
            }
        }
        else {
            ACTION_1
        }
    goto ll_end;
ll_break_0_2:
        if (CONDITION_O) {
            ACTION_6
        }
        else {
            ACTION_1
        }
    goto ll_end;
ll_break_0_3:
        if (CONDITION_O) {
            if (CONDITION_N) {
                ACTION_6
            }
            else {
                NODE_347:
                if (CONDITION_I) {
                    ACTION_4
                }
                else {
                    ACTION_3
                }
            }
        }
        else {
            ACTION_1
        }
    goto ll_end;
ll_break_1_0:
        if (CONDITION_O) {
            NODE_348:
            if (CONDITION_J) {
                ACTION_4
            }
            else{
                goto NODE_343;
            }
        }
        else {
            NODE_349:
            if (CONDITION_P){
                goto NODE_348;
            }
            else {
                ACTION_1
            }
        }
    goto ll_end;
ll_break_1_1:
        if (CONDITION_O) {
            if (CONDITION_J) {
                if (CONDITION_I) {
                    ACTION_4
                }
                else {
                    if (CONDITION_H) {
                        NODE_353:
                        if (CONDITION_C) {
                            ACTION_4
                        }
                        else {
                            ACTION_7
                        }
                    }
                    else {
                        ACTION_4
                    }
                }
            }
            else{
                goto NODE_344;
            }
        }
        else{
            goto NODE_349;
        }
    goto ll_end;
ll_break_1_2:
        if (CONDITION_O) {
            if (CONDITION_J) {
                ACTION_11
            }
            else {
                ACTION_6
            }
        }
        else {
            NODE_355:
            if (CONDITION_P) {
                if (CONDITION_J) {
                    ACTION_4
                }
                else {
                    ACTION_2
                }
            }
            else {
                ACTION_1
            }
        }
    goto ll_end;
ll_break_1_3:
        if (CONDITION_O) {
            if (CONDITION_J) {
                if (CONDITION_C) {
                    NODE_359:
                    if (CONDITION_B) {
                        ACTION_4
                    }
                    else {
                        ACTION_7
                    }
                }
                else {
                    ACTION_11
                }
            }
            else {
                ACTION_6
            }
        }
        else{
            goto NODE_355;
        }
    goto ll_end;
ll_break_1_4:
        if (CONDITION_O) {
            if (CONDITION_J) {
                ACTION_4
            }
            else {
                ACTION_6
            }
        }
        else {
            if (CONDITION_P) {
                ACTION_4
            }
            else {
                ACTION_1
            }
        }
    goto ll_end;
ll_break_1_5:
        if (CONDITION_O) {
            NODE_362:
            if (CONDITION_J) {
                if (CONDITION_I) {
                    ACTION_4
                }
                else {
                    if (CONDITION_C) {
                        ACTION_4
                    }
                    else {
                        ACTION_11
                    }
                }
            }
            else {
                ACTION_6
            }
        }
        else{
            goto NODE_349;
        }
    goto ll_end;
ll_break_1_6:
        if (CONDITION_O) {
            if (CONDITION_N){
                goto NODE_362;
            }
            else {
                if (CONDITION_J) {
                    if (CONDITION_I) {
                        ACTION_4
                    }
                    else{
                        goto NODE_353;
                    }
                }
                else{
                    goto NODE_347;
                }
            }
        }
        else{
            goto NODE_349;
        }
    goto ll_end;
ll_break_1_7:
        if (CONDITION_O) {
            if (CONDITION_J) {
                if (CONDITION_C) {
                    if (CONDITION_G){
                        goto NODE_359;
                    }
                    else {
                        ACTION_11
                    }
                }
                else {
                    ACTION_11
                }
            }
            else {
                ACTION_6
            }
        }
        else{
            goto NODE_355;
        }
    goto ll_end;
ll_end:;
