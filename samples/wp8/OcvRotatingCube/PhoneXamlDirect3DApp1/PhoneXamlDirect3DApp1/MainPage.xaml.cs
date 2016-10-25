using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Navigation;
using Microsoft.Phone.Controls;
using Microsoft.Phone.Shell;
using PhoneXamlDirect3DApp1Comp;
using Microsoft.Phone.Tasks;
using System.Windows.Media.Imaging;
using System.Threading;
using System.Windows.Resources;
using System.IO;
using  System.Runtime.InteropServices.WindowsRuntime;
using Microsoft.Xna.Framework.Media;

namespace PhoneXamlDirect3DApp1
{
    public partial class MainPage : PhoneApplicationPage
    {
        private Direct3DInterop m_d3dInterop = new Direct3DInterop();
        WriteableBitmap m_bmp;
        bool m_bInitialized = false;

        // Constructor
        public MainPage()
        {
            InitializeComponent();
        }

        private void DrawingSurface_Loaded(object sender, RoutedEventArgs e)
        {
            // Set window bounds in dips
            m_d3dInterop.WindowBounds = new Windows.Foundation.Size(
                (float)DrawingSurface.ActualWidth,
                (float)DrawingSurface.ActualHeight
                );

            // Set native resolution in pixels
            m_d3dInterop.NativeResolution = new Windows.Foundation.Size(
                (float)Math.Floor(DrawingSurface.ActualWidth * Application.Current.Host.Content.ScaleFactor / 100.0f + 0.5f),
                (float)Math.Floor(DrawingSurface.ActualHeight * Application.Current.Host.Content.ScaleFactor / 100.0f + 0.5f)
                );

            // Set render resolution to the full native resolution
            m_d3dInterop.RenderResolution = m_d3dInterop.NativeResolution;

            // Hook-up native component to DrawingSurface
            DrawingSurface.SetContentProvider(m_d3dInterop.CreateContentProvider());
            DrawingSurface.SetManipulationHandler(m_d3dInterop);


            Deployment.Current.Dispatcher.BeginInvoke(() =>
                {
                    StreamResourceInfo resourceInfo = Application.GetResourceStream(new Uri("Assets/Lena.png", UriKind.Relative));
                    BitmapImage bitmap = new BitmapImage();
                    bitmap.SetSource(resourceInfo.Stream);
                    m_bmp = new WriteableBitmap(bitmap);
                    m_d3dInterop.CreateTexture(m_bmp.Pixels, m_bmp.PixelWidth, m_bmp.PixelHeight, OCVFilterType.ePreview);
                    m_bInitialized = true;
                });
        }


        private void RadioButton_Checked(object sender, RoutedEventArgs e)
        {
            if (!m_bInitialized)
            {
                return;
            }

            RadioButton rb = sender as RadioButton;
            switch (rb.Name)
            {
                case "Normal":
                    m_d3dInterop.CreateTexture(m_bmp.Pixels, m_bmp.PixelWidth, m_bmp.PixelHeight, OCVFilterType.ePreview);
                    break;

                case "Gray":
                    m_d3dInterop.CreateTexture(m_bmp.Pixels, m_bmp.PixelWidth, m_bmp.PixelHeight, OCVFilterType.eGray);
                    break;

                case "Canny":
                    m_d3dInterop.CreateTexture(m_bmp.Pixels, m_bmp.PixelWidth, m_bmp.PixelHeight, OCVFilterType.eCanny);
                    break;

                case "Sepia":
                    m_d3dInterop.CreateTexture(m_bmp.Pixels, m_bmp.PixelWidth, m_bmp.PixelHeight, OCVFilterType.eSepia);
                    break;
            }
        }
    }
}