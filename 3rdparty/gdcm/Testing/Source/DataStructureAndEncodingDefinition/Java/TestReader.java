/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*
 */
import gdcm.*;
import java.util.Properties;
import java.util.Enumeration;

public class TestReader
{
  public static void main(String[] args) throws Exception
    {
/*
    System.out.println("PATH : "
      + System.getProperty("java.library.path"));
    Properties p = System.getProperties();
    Enumeration keys = p.keys();
    while (keys.hasMoreElements()) {
      String key = (String)keys.nextElement();
      String value = (String)p.get(key);
      System.out.println(key + ": " + value);
    }
*/

    long nfiles = Testing.GetNumberOfFileNames();
    Trace.DebugOff();
    Trace.WarningOff();

    for( long i = 0; i < nfiles; ++i )
      {
      String filename = Testing.GetFileName( i );
      //System.out.println("Success reading: " + filename );
      Reader reader = new Reader();
      reader.SetFileName( filename );
      if ( !reader.Read() )
        {
        throw new Exception("Could not read: " + filename );
        }
      String ref = Testing.GetMediaStorageFromFile(filename);
      if( ref == null )
        {
        throw new Exception("Missing ref for: " + filename );
        }
      MediaStorage ms = new MediaStorage();
      ms.SetFromFile( reader.GetFile() );
      if( ms.IsUndefined() && !"".equals( ref ) )
        {
        // gdcm-CR-DCMTK-16-NonSamplePerPix.dcm is empty
        throw new Exception("ref is undefined for: " + filename + " should be " + ref );
        }
      MediaStorage.MSType ref_mstype = MediaStorage.GetMSType( ref );
      if( !"".equals( ref ) && ms.GetType() != ref_mstype )
        {
        throw new Exception("incompatible type: " + ref + " vs " + ms.GetString() + " for " + filename );
        }
      }
    }
}
