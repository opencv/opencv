//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

//
// SuspensionManager.h
// Declaration of the SuspensionManager class
//

#pragma once

#include <ppltasks.h>

namespace SDKSample
{
    namespace Common
    {
        /// <summary>
        /// SuspensionManager captures global session state to simplify process lifetime management
        /// for an application.  Note that session state will be automatically cleared under a variety
        /// of conditions and should only be used to store information that would be convenient to
        /// carry across sessions, but that should be disacarded when an application crashes or is
        /// upgraded.
        /// </summary>
        ref class SuspensionManager sealed
        {
        internal:
            static void RegisterFrame(Windows::UI::Xaml::Controls::Frame^ frame, Platform::String^ sessionStateKey);
            static void UnregisterFrame(Windows::UI::Xaml::Controls::Frame^ frame);
            static Concurrency::task<void> SaveAsync(void);
            static Concurrency::task<void> RestoreAsync(void);
            static property Windows::Foundation::Collections::IMap<Platform::String^, Platform::Object^>^ SessionState
            {
                Windows::Foundation::Collections::IMap<Platform::String^, Platform::Object^>^ get(void);
            };
            static Windows::Foundation::Collections::IMap<Platform::String^, Platform::Object^>^ SessionStateForFrame(
                Windows::UI::Xaml::Controls::Frame^ frame);

        private:
            static void RestoreFrameNavigationState(Windows::UI::Xaml::Controls::Frame^ frame);
            static void SaveFrameNavigationState(Windows::UI::Xaml::Controls::Frame^ frame);
        };
    }
}
