/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#include "precomp.hpp"
#include "opencv2/flann.hpp"

namespace cvflann
{
    /** Global variable indicating the distance metric to be used.
     * \deprecated Provided for backward compatibility
    */
    flann_distance_t flann_distance_type_ = FLANN_DIST_L2;
    CV_DEPRECATED flann_distance_t flann_distance_type() { return flann_distance_type_; }

    /**
     * Set distance type to used
     * \deprecated
     */
    void set_distance_type(flann_distance_t distance_type, int /*order*/)
    {
        printf("[WARNING] The cvflann::set_distance_type function is deperecated, "
            "use cv::flann::GenericIndex<Distance> instead.\n");
        if (distance_type != FLANN_DIST_L1 && distance_type != FLANN_DIST_L2) {
            printf("[ERROR] cvflann::set_distance_type only provides backwards compatibility "
            "for the L1 and L2 distances. "
            "For other distance types you must use cv::flann::GenericIndex<Distance>\n");
        }
        flann_distance_type_ = distance_type;
    }

}
