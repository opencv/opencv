// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_VERSION_HPP
#define OPENCV_VERSION_HPP

#define CV_VERSION_MAJOR    4
#define CV_VERSION_MINOR    9
#define CV_VERSION_REVISION 0
#define CV_VERSION_STATUS   ""

#define CVAUX_STR_EXP(__A)  #__A
#define CVAUX_STR(__A)      CVAUX_STR_EXP(__A)

#define CVAUX_STRW_EXP(__A)  L ## #__A
#define CVAUX_STRW(__A)      CVAUX_STRW_EXP(__A)

#define CV_VERSION          CVAUX_STR(CV_VERSION_MAJOR) "." CVAUX_STR(CV_VERSION_MINOR) "." CVAUX_STR(CV_VERSION_REVISION) CV_VERSION_STATUS

/* old  style version constants*/
#define CV_MAJOR_VERSION    CV_VERSION_MAJOR
#define CV_MINOR_VERSION    CV_VERSION_MINOR
#define CV_SUBMINOR_VERSION CV_VERSION_REVISION

#endif // OPENCV_VERSION_HPP
