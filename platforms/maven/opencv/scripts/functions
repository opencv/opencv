#!/bin/bash
###############################################################
#
# Defines some common functions.
#
# Kerry Billingham <contact [At] AvionicEngineers.{com]>
#
##############################################################
majorHashDefine="#define CV_VERSION_MAJOR"
minorHashDefine="#define CV_VERSION_MINOR"
revisionHashDefine="#define CV_VERSION_REVISION"
statusHashDefine="#define CV_VERSION_STATUS"
versionHeader="../../../../modules/core/include/opencv2/core/version.hpp"

function extract_version() {
    minorVersion=$(grep "${minorHashDefine}" $versionHeader | grep -o ".$")
    majorVersion=$(grep "${majorHashDefine}" $versionHeader | grep -o ".$")
    revision=$(grep "${revisionHashDefine}" $versionHeader | grep -o ".$")

    REPLY="${majorVersion}.${minorVersion}.${revision}"
}
