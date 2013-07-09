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
// MainPage.xaml.h
// Declaration of the MainPage.xaml class.
//

#pragma once

#include "pch.h"
#include "MainPage.g.h"
#include "Common\LayoutAwarePage.h" // Required by generated header
#include "Constants.h"

namespace SDKSample
{
    public enum class NotifyType
    {
        StatusMessage,
        ErrorMessage
    };

    public ref class MainPageSizeChangedEventArgs sealed
    {
    public:
        property Windows::UI::ViewManagement::ApplicationViewState ViewState
        {
            Windows::UI::ViewManagement::ApplicationViewState get()
            {
                return viewState;
            }

            void set(Windows::UI::ViewManagement::ApplicationViewState value)
            {
                viewState = value;
            }
        }

    private:
        Windows::UI::ViewManagement::ApplicationViewState viewState;
    };

    public ref class MainPage sealed
    {
    public:
        MainPage();

    protected:
        virtual void LoadState(Platform::Object^ navigationParameter,
            Windows::Foundation::Collections::IMap<Platform::String^, Platform::Object^>^ pageState) override;
        virtual void SaveState(Windows::Foundation::Collections::IMap<Platform::String^, Platform::Object^>^ pageState) override;

    internal:
        property bool AutoSizeInputSectionWhenSnapped
        {
            bool get()
            {
                return autoSizeInputSectionWhenSnapped;
            }

            void set(bool value)
            {
                autoSizeInputSectionWhenSnapped = value;
            }
        }

        property Windows::ApplicationModel::Activation::LaunchActivatedEventArgs^ LaunchArgs
       {
            Windows::ApplicationModel::Activation::LaunchActivatedEventArgs^ get()
            {
                return safe_cast<App^>(App::Current)->LaunchArgs;
            }
        }

        void NotifyUser(Platform::String^ strMessage, NotifyType type);
        void LoadScenario(Platform::String^ scenarioName);
        event Windows::Foundation::EventHandler<Platform::Object^>^ ScenarioLoaded;
        event Windows::Foundation::EventHandler<MainPageSizeChangedEventArgs^>^ MainPageResized;

    private:
        void PopulateScenarios();
        void InvalidateSize();
        void InvalidateViewState();

        Platform::Collections::Vector<Object^>^ ScenarioList;
        Windows::UI::Xaml::Controls::Frame^ HiddenFrame;
        void Footer_Click(Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
        bool autoSizeInputSectionWhenSnapped;

        void MainPage_SizeChanged(Object^ sender, Windows::UI::Xaml::SizeChangedEventArgs^ e);
        void Scenarios_SelectionChanged(Object^ sender, Windows::UI::Xaml::Controls::SelectionChangedEventArgs^ e);

    internal:
        static MainPage^ Current;

    };
}
