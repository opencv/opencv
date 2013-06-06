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
// App.xaml.cpp
// Implementation of the App.xaml class.
//

#include "pch.h"
#include "MainPage.xaml.h"
#include "Common\SuspensionManager.h"

using namespace SDKSample;
using namespace SDKSample::Common;

using namespace Concurrency;
using namespace Platform;
using namespace Windows::ApplicationModel;
using namespace Windows::ApplicationModel::Activation;
using namespace Windows::Foundation;
using namespace Windows::Foundation::Collections;
using namespace Windows::UI::Core;
using namespace Windows::UI::Xaml;
using namespace Windows::UI::Xaml::Controls;
using namespace Windows::UI::Xaml::Controls::Primitives;
using namespace Windows::UI::Xaml::Data;
using namespace Windows::UI::Xaml::Input;
using namespace Windows::UI::Xaml::Interop;
using namespace Windows::UI::Xaml::Media;
using namespace Windows::UI::Xaml::Navigation;

/// <summary>
/// Initializes the singleton application object.  This is the first line of authored code
/// executed, and as such is the logical equivalent of main() or WinMain().
/// </summary>
App::App()
{
    InitializeComponent();
    this->Suspending += ref new SuspendingEventHandler(this, &SDKSample::App::OnSuspending);
}

/// <summary>
/// Invoked when the application is launched normally by the end user.  Other entry points will
/// be used when the application is launched to open a specific file, to display search results,
/// and so forth.
/// </summary>
/// <param name="pArgs">Details about the launch request and process.</param>
void App::OnLaunched(LaunchActivatedEventArgs^ pArgs)
{
    this->LaunchArgs = pArgs;

    // Do not repeat app initialization when already running, just ensure that
    // the window is active
    if (pArgs->PreviousExecutionState == ApplicationExecutionState::Running)
    {
        Window::Current->Activate();
        return;
    }

    // Create a Frame to act as the navigation context and associate it with
    // a SuspensionManager key
    auto rootFrame = ref new Frame();
    SuspensionManager::RegisterFrame(rootFrame, "AppFrame");

    auto prerequisite = task<void>([](){});
    if (pArgs->PreviousExecutionState == ApplicationExecutionState::Terminated)
    {
        // Restore the saved session state only when appropriate, scheduling the
        // final launch steps after the restore is complete
        prerequisite = SuspensionManager::RestoreAsync();
    }
    prerequisite.then([=]()
    {
        // When the navigation stack isn't restored navigate to the first page,
        // configuring the new page by passing required information as a navigation
        // parameter
        if (rootFrame->Content == nullptr)
        {
            if (!rootFrame->Navigate(TypeName(MainPage::typeid)))
            {
                throw ref new FailureException("Failed to create initial page");
            }
        }

        // Place the frame in the current Window and ensure that it is active
        Window::Current->Content = rootFrame;
        Window::Current->Activate();
    }, task_continuation_context::use_current());
}

/// <summary>
/// Invoked when application execution is being suspended.  Application state is saved
/// without knowing whether the application will be terminated or resumed with the contents
/// of memory still intact.
/// </summary>
/// <param name="sender">The source of the suspend request.</param>
/// <param name="e">Details about the suspend request.</param>
void App::OnSuspending(Object^ sender, SuspendingEventArgs^ e)
{
    (void) sender;	// Unused parameter

    auto deferral = e->SuspendingOperation->GetDeferral();
    SuspensionManager::SaveAsync().then([=]()
    {
        deferral->Complete();
    });
}
