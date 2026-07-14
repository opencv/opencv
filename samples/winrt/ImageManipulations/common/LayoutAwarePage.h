//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#pragma once

#include <collection.h>

namespace SDKSample
{
    namespace Common
    {
        /// <summary>
        /// Typical implementation of Page that provides several important conveniences:
        /// <list type="bullet">
        /// <item>
        /// <description>Application view state to visual state mapping</description>
        /// </item>
        /// <item>
        /// <description>GoBack, GoForward, and GoHome event handlers</description>
        /// </item>
        /// <item>
        /// <description>Mouse and keyboard shortcuts for navigation</description>
        /// </item>
        /// <item>
        /// <description>State management for navigation and process lifetime management</description>
        /// </item>
        /// <item>
        /// <description>A default view model</description>
        /// </item>
        /// </list>
        /// </summary>
        [Windows::Foundation::Metadata::WebHostHidden]
        public ref class LayoutAwarePage : Windows::UI::Xaml::Controls::Page
        {
        internal:
            LayoutAwarePage();

        public:
            void StartLayoutUpdates(Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
            void StopLayoutUpdates(Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
            void InvalidateVisualState();
            static property Windows::UI::Xaml::DependencyProperty^ DefaultViewModelProperty
            {
                Windows::UI::Xaml::DependencyProperty^ get();
            };
            property Windows::Foundation::Collections::IObservableMap<Platform::String^, Platform::Object^>^ DefaultViewModel
            {
                Windows::Foundation::Collections::IObservableMap<Platform::String^, Platform::Object^>^ get();
                void set(Windows::Foundation::Collections::IObservableMap<Platform::String^, Platform::Object^>^ value);
            }

        protected:
            virtual void GoHome(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
            virtual void GoBack(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
            virtual void GoForward(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
            virtual Platform::String^ DetermineVisualState(Windows::UI::ViewManagement::ApplicationViewState viewState);
            virtual void OnNavigatedTo(Windows::UI::Xaml::Navigation::NavigationEventArgs^ e) override;
            virtual void OnNavigatedFrom(Windows::UI::Xaml::Navigation::NavigationEventArgs^ e) override;
            virtual void LoadState(Platform::Object^ navigationParameter,
                Windows::Foundation::Collections::IMap<Platform::String^, Platform::Object^>^ pageState);
            virtual void SaveState(Windows::Foundation::Collections::IMap<Platform::String^, Platform::Object^>^ pageState);

        private:
            Platform::String^ _pageKey;
            bool _navigationShortcutsRegistered;
            Platform::Collections::Map<Platform::String^, Platform::Object^>^ _defaultViewModel;
            Windows::Foundation::EventRegistrationToken _windowSizeEventToken,
                _acceleratorKeyEventToken, _pointerPressedEventToken;
            Platform::Collections::Vector<Windows::UI::Xaml::Controls::Control^>^ _layoutAwareControls;
            void WindowSizeChanged(Platform::Object^ sender, Windows::UI::Core::WindowSizeChangedEventArgs^ e);
            void OnLoaded(Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
            void OnUnloaded(Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);

            void CoreDispatcher_AcceleratorKeyActivated(Windows::UI::Core::CoreDispatcher^ sender,
                Windows::UI::Core::AcceleratorKeyEventArgs^ args);
            void CoreWindow_PointerPressed(Windows::UI::Core::CoreWindow^ sender,
                Windows::UI::Core::PointerEventArgs^ args);
            LayoutAwarePage^ _this; // Strong reference to self, cleaned up in OnUnload
        };
    }
}
