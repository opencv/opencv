//
// App.xaml.h
// Declaration of the App class.
//

#pragma once

#include "App.g.h"

namespace highgui_xaml
{
    /// <summary>
    /// Provides application-specific behavior to supplement the default Application class.
    /// </summary>
    ref class App sealed
    {
    public:
        App();

        virtual void OnLaunched(Windows::ApplicationModel::Activation::LaunchActivatedEventArgs^ e) override;

    private:
#if WINAPI_FAMILY==WINAPI_FAMILY_PHONE_APP
        Windows::UI::Xaml::Media::Animation::TransitionCollection^ _transitions;
        Windows::Foundation::EventRegistrationToken _firstNavigatedToken;

        void RootFrame_FirstNavigated(Platform::Object^ sender, Windows::UI::Xaml::Navigation::NavigationEventArgs^ e);
#endif

        void OnSuspending(Platform::Object^ sender, Windows::ApplicationModel::SuspendingEventArgs^ e);
        void OnResuming(Platform::Object ^sender, Platform::Object ^args);
    };
}
