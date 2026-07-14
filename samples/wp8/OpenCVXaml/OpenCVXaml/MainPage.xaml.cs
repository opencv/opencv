using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Navigation;
using Microsoft.Phone.Controls;
using Microsoft.Phone.Shell;
using OpenCVXaml.Resources;
using System.Windows.Media.Imaging;
using OpenCVComponent;

namespace OpenCVXaml
{
    public partial class MainPage : PhoneApplicationPage
    {
        private OpenCVLib m_opencv = new OpenCVLib();

        // Constructor
        public MainPage()
        {
            InitializeComponent();

            // Sample code to localize the ApplicationBar
            //BuildLocalizedApplicationBar();
        }

        private async void Button_Click(object sender, RoutedEventArgs e)
        {
            if (Preview.Source != null)
            {
                ProcessButton.IsEnabled = false;

                // Get WriteableBitmap. ImageToModify is defined in MainPage.xaml
                WriteableBitmap bitmap = new WriteableBitmap(Preview.Source as BitmapSource);

                // call OpenCVLib to convert pixels to grayscale. This is an asynchronous call.
                var pixels =  await m_opencv.ProcessAsync(bitmap.Pixels, bitmap.PixelWidth, bitmap.PixelHeight);

                // copy the pixels into the WriteableBitmap
                for (int x = 0; x < bitmap.Pixels.Length; x++)
                {
                    bitmap.Pixels[x] = pixels[x];
                }

                // Set Image object, defined in XAML, to the modified bitmap.
                Preview.Source = bitmap;

                ProcessButton.IsEnabled = true;
            }
        }

        // Sample code for building a localized ApplicationBar
        //private void BuildLocalizedApplicationBar()
        //{
        //    // Set the page's ApplicationBar to a new instance of ApplicationBar.
        //    ApplicationBar = new ApplicationBar();

        //    // Create a new button and set the text value to the localized string from AppResources.
        //    ApplicationBarIconButton appBarButton = new ApplicationBarIconButton(new Uri("/Assets/AppBar/appbar.add.rest.png", UriKind.Relative));
        //    appBarButton.Text = AppResources.AppBarButtonText;
        //    ApplicationBar.Buttons.Add(appBarButton);

        //    // Create a new menu item with the localized string from AppResources.
        //    ApplicationBarMenuItem appBarMenuItem = new ApplicationBarMenuItem(AppResources.AppBarMenuItemText);
        //    ApplicationBar.MenuItems.Add(appBarMenuItem);
        //}
    }
}