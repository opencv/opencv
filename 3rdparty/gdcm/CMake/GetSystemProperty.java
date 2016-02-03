/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
import java.util.Properties;
import java.util.Enumeration;

/*
 * Java only allows setting properties from the command line not reading them
 * Let's create a small app for this specific task then:
 *
 * namely:
 */
public class GetSystemProperty {
  public static void main(String args[]) {
    if( args.length == 0 ) {
      Properties p = System.getProperties();
      Enumeration keys = p.keys();
      while (keys.hasMoreElements()) {
        String key = (String)keys.nextElement();
        String value = (String)p.get(key);
        System.out.println(key + " : " + value);
      }
    }
    else {
      for (String key: args) {
        System.out.println(System.getProperty( key ));
      }
    }
  }
}
