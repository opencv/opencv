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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// Authors:
//  * Ozan Tonkal, ozantonkal@gmail.com
//  * Anatoly Baksheev, Itseez Inc.  myname.mysurname <> mycompany.com
//
//M*/

#ifndef __OPENCV_VIZ_WIDGET_ACCESSOR_HPP__
#define __OPENCV_VIZ_WIDGET_ACCESSOR_HPP__

#include <opencv2/core/cvdef.h>
#include <vtkSmartPointer.h>
#include <vtkProp.h>

namespace cv
{
    namespace viz
    {

//! @addtogroup viz_widget
//! @{

        class Widget;

        /** @brief This class is for users who want to develop their own widgets using VTK library API. :
        */
        struct CV_EXPORTS WidgetAccessor
        {
            /** @brief Returns vtkProp of a given widget.

            @param widget Widget whose vtkProp is to be returned.

            @note vtkProp has to be down cast appropriately to be modified.
                @code
                vtkActor * actor = vtkActor::SafeDownCast(viz::WidgetAccessor::getProp(widget));
                @endcode
             */
            static vtkSmartPointer<vtkProp> getProp(const Widget &widget);
            /** @brief Sets vtkProp of a given widget.

            @param widget Widget whose vtkProp is to be set. @param prop A vtkProp.
             */
            static void setProp(Widget &widget, vtkSmartPointer<vtkProp> prop);
        };

//! @}

    }
}

#endif
