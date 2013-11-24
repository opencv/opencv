#ifndef __BP_OPENCV_ENGINE_H__
#define __BP_OPENCV_ENGINE_H__

#include "EngineCommon.h"
#include "IOpenCVEngine.h"
#include <binder/IInterface.h>
#include <binder/Parcel.h>
#include <utils/String16.h>

class BnOpenCVEngine: public android::BnInterface<IOpenCVEngine>
{
public:
    android::status_t onTransact(uint32_t code,
                 const android::Parcel &data,
                 android::Parcel *reply,
                 uint32_t flags);
    virtual ~BnOpenCVEngine();

};

#endif
