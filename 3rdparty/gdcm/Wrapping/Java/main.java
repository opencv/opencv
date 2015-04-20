/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// This example illustrates how C++ classes can be used from Java using SWIG.
// The Java class gets mapped onto the C++ class and behaves as if it is a Java class.

public class main {
  static {
    try {
        System.loadLibrary("gdcm");
    } catch (UnsatisfiedLinkError e) {
      System.err.println("Native code library failed to load. See the chapter on Dynamic Linking Problems in the SWIG Java documentation for help.\n" + e);
      System.exit(1);
    }
  }

  public static void main(String argv[])
  {
    // ----- Object creation -----
    System.out.println( "Creating some objects:" );
    Reader r = new Reader();
    UIDGenerator uid = new UIDGenerator();
    String s = uid.Generate();
    System.out.println( uid.Generate() );
    System.out.println( "Goodbye" );
  }
}
