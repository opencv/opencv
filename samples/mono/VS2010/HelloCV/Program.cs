using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using org.opencv.core;
using System.IO;

namespace HelloCV
{
    /// <summary>
    /// Demonstrates usage of Mono / .NET bindings for OpenCV.
    /// opencv-mono.dll and IKVM.OpenJDK.Core can be found in OpenCV/mono.
    /// opencv_javaNNN.dll can be found in OpenCV/java.
    /// opencv_javaNNN.dll must either be in the output path or reachable
    /// through PATH environment variable.
    /// </summary>
    class Program
    {
        static Program()
        {
            // In case opencv_javaNNN.dll is NOT located in the output path extend
            // the process local PATH environment variable before accessing any 
            // OpenCV type.
            /*
            var path = Environment.GetEnvironmentVariable("PATH");
            var openCVInstallPath = "path/to/opencv";
            Environment.SetEnvironmentVariable(
                "PATH", 
                path + Path.PathSeparator + Path.Combine(openCVInstallPath, "java")
            );
            */
        }

        static void Main(string[] args)
        {
            var mat = Mat.eye(3, 3, CvType.CV_8UC1);
            Console.WriteLine("m = {0}", mat);
            Console.ReadLine();
        }
    }
}
