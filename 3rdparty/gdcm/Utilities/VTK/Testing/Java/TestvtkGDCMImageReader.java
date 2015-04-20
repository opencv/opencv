/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
import gdcm.*;
import vtk.*;

/*
 */
public class TestvtkGDCMImageReader
{
  static {
    System.loadLibrary("vtkCommonJava");
    System.loadLibrary("vtkFilteringJava");
    System.loadLibrary("vtkIOJava");
    System.loadLibrary("vtkgdcmJava");
  }

  public static void main(String[] args)
    {
    long nfiles = Testing.GetNumberOfFileNames();
    Trace.DebugOff();
    Trace.WarningOff();

    for( long i = 0; i < nfiles; ++i )
      {
      String filename = Testing.GetFileName( i );
      //System.out.println("Success reading: " + filename );
      vtkGDCMImageReader reader = new vtkGDCMImageReader();
      reader.SetFileName( filename );
      reader.Update();
      }
    }
}
