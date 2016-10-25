//
// MainPage.xaml.h
// Declaration of the MainPage class.
//

#pragma once

#include "MainPage.g.h"

namespace PhoneTutorial
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public ref class MainPage sealed
    {
    public:
        MainPage();

    protected:
        virtual void OnNavigatedTo(Windows::UI::Xaml::Navigation::NavigationEventArgs^ e) override;
    private:

        Windows::UI::Xaml::Media::Imaging::WriteableBitmap^ m_bitmap;
        void Process_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
        void Reset_Click(Platform::Object^ sender, Windows::UI::Xaml::RoutedEventArgs^ e);
        void LoadImage();
    };
}
