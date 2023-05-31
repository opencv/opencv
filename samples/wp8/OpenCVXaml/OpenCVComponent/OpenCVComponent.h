#pragma once

#include <ppltasks.h>
#include <collection.h>

namespace OpenCVComponent
{
    public ref class OpenCVLib sealed
    {
    public:
        OpenCVLib();
        Windows::Foundation::IAsyncOperation<Windows::Foundation::Collections::IVectorView<int>^>^ ProcessAsync(Windows::Foundation::Collections::IVector<int>^ input, int width, int height);
    };
}