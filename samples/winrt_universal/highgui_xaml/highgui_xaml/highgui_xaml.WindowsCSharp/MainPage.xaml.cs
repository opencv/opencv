// Copyright (c) 2015, Microsoft Open Technologies, Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// - Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// - Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// - Neither the name of Microsoft Open Technologies, Inc. nor the names
//   of its contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
// ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices.WindowsRuntime;
using System.Threading.Tasks;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.System.Threading;
using Windows.UI.Core;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.UI.Xaml.Controls.Primitives;
using Windows.UI.Xaml.Data;
using Windows.UI.Xaml.Input;
using Windows.UI.Xaml.Media;
using Windows.UI.Xaml.Navigation;
using System.Threading;
using Windows.Foundation.Metadata;
using Windows.UI.ViewManagement;
using cvRT;
using static Windows.System.Threading.ThreadPool;
using Point = cvRT.Point;
using Rect = cvRT.Rect;

// The Blank Page item template is documented at http://go.microsoft.com/fwlink/?LinkId=402352&clcid=0x409

namespace highgui_xaml.WindowsCSharp
{
    /// <summary>
    /// An empty page that can be used on its own or navigated to within a Frame.
    /// </summary>
    public sealed partial class MainPage : Page
    {
        /// <summary>
        /// 
        /// </summary>
        private readonly VideoIo _videoIo;

        private int _processingMethodIndex = 0;

        /// <summary>
        /// 
        /// </summary>
        public MainPage()
        {
            this.InitializeComponent();

            _videoIo = new VideoIo();
            _videoIo.Initialize();
            _videoIo.SetImage(OCVImage);

            RunAsync(asyncAction => CvMainThread());
        }

        /// <summary>
        /// User input on image processing method.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void comboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            var cb = (ComboBox)sender;
            _processingMethodIndex = cb.SelectedIndex;
            var selectedItem = ((ComboBoxItem)cb.SelectedItem).Content.ToString();
            Debug.WriteLine(selectedItem);
        }

        /// <summary>
        /// Worker thread for image processing.
        /// </summary>
        public void CvMainThread()
        {
            var faceCascade = new CascadeClassifier();
            var eyesCascade = new CascadeClassifier();

            faceCascade.load("haarcascade_frontalface_alt.xml");
            eyesCascade.load("haarcascade_eye_tree_eyeglasses.xml");
            
            var srcFrame = new Mat();
            var dstFrame = new Mat();

            var imgProc = new ImgProc();

            _videoIo.StartCapture();

            while (true)
            {
                _videoIo.GetFrame(srcFrame);

                switch (_processingMethodIndex)
                {
                    // passthrough
                    case 0:
                        break;
                    
                    // gray
                    case 1:
                        imgProc.cvtColor(srcFrame, dstFrame, ColorConversionCodes.COLOR_RGBA2GRAY);
                        imgProc.cvtColor(dstFrame, srcFrame, ColorConversionCodes.COLOR_GRAY2RGB);
                        break;
                    
                    // canny
                    case 3:
                        imgProc.cvtColor(srcFrame, dstFrame, cvRT.ColorConversionCodes.COLOR_RGBA2GRAY);
                        imgProc.GaussianBlur(dstFrame, dstFrame, new cvRT.Size(7, 7), 1.5, 1.5);
                        imgProc.Canny(dstFrame, dstFrame, 0, 30, 3);
                        imgProc.cvtColor(dstFrame, srcFrame, ColorConversionCodes.COLOR_GRAY2RGB);
                        break;

                    // contour
                    case 4:
                    {
                        var contours = new VectorOfVectorOfPoint();
                        var hierarchy = new VectorOfVec4i();
                        var color = new Scalar(255, 255, 255, 255);

                        imgProc.Canny(srcFrame, dstFrame, 100, 100 * 2, 3);
                        imgProc.FindContours(dstFrame, contours, hierarchy, ContourRetrievalAlgorithm.RETR_TREE, ContourApproximationModes.CHAIN_APPROX_SIMPLE, new Point(0, 0));

                        srcFrame.Set(new Scalar(0, 0, 0, 0));

                        for (var i = 0 ; i < contours.Count();  i++)
                        {
                            imgProc.DrawContours(srcFrame, contours, i, color, 2, 8, hierarchy, 0, new Point(0, 0));
                        }
                            
                        break;
                    }

                    // face detect
                    case 5:
                    {
                        imgProc.cvtColor(srcFrame, dstFrame, ColorConversionCodes.COLOR_RGBA2GRAY);
                        imgProc.EqualizeHist(dstFrame, dstFrame);

                        // Faces in the frame.
                        var faces = new List<Rect>();

                        try
                        {
                            faces = new List<Rect>();
                            faceCascade.detectMultiScale(dstFrame, faces, 1.1, 2, (int)(0 | CV_HAAR.SCALE_IMAGE), new cvRT.Size(30, 30));
                        }
                        catch (Exception ex)
                        {
                            Debug.WriteLine("Exception {0}", ex.Message);
                        }

                        // For each face, detect the eyes
                        foreach (var face in faces)
                        {
                            // Draw ellipse for the face.
                            var faceCenter = new Point(face.X + face.Width / 2, face.Y + face.Height / 2);
                            imgProc.Ellipse(srcFrame, faceCenter, new cvRT.Size(face.Width / 2, face.Height / 2), 0, 0, 360, new Scalar(255, 0, 255, 0), 4, 8, 0);

                            // Detect the eyes for the face
                            var faceRoi = dstFrame.RectOfInterest(face);
                            var eyes = new List<Rect>();
                            eyesCascade.detectMultiScale(faceRoi, eyes, 1.1, 2, (int) (0 | CASCADE_FLAG.CASCADE_SCALE_IMAGE),new cvRT.Size(30, 30));
                            
                            // Draw the eyes
                            foreach (var eye in eyes)
                            {
                                var eyeCenter = new Point(face.X + eye.X + eye.Width/2, face.Y + eye.Y + eye.Height/2);
                                var radius = (int) Math.Round((eye.Width + eye.Height) * 0.25);
                                imgProc.Circle(srcFrame, eyeCenter, radius, new Scalar(255, 0, 0, 0), 4, 8, 0);
                            }
                       }

                       break;
                    }

                    default:
                        break;
                }

                _videoIo.ShowFrame(srcFrame);
            }
        }
    }
}