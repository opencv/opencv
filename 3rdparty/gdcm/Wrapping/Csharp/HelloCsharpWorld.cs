/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

using System;
using gdcm;

public class HelloWorld
{
  public static int Main(string[] args)
    {
    System.Console.WriteLine("Hello World !");
    //gdcm.Reader reader2;
    string filename = args[0];
    System.Console.WriteLine( "Reading: " + filename );
    Reader reader = new Reader();
    reader.SetFileName( filename );
    bool ret = reader.Read();
    if( !ret )
      {
      //throw new Exception("Could not read: " + filename );
      return 1;
      }
    //std::cout << reader.GetFile()
    Tag t = new Tag(0x10,0x10);
    System.Console.WriteLine( "out:" + t.toString() );
    System.Console.WriteLine( "out:" + reader.GetFile().GetDataSet().toString() );

    Anonymizer ano = new Anonymizer();
    ano.SetFile( reader.GetFile() );
    ano.RemovePrivateTags();
    ano.RemoveGroupLength();
    ano.Replace( t, "GDCM^Csharp^Test^Hello^World" );

    Writer writer = new Writer();
    writer.SetFileName( "testcs.dcm" );
    writer.SetFile( ano.GetFile() );
    ret = writer.Write();
    if( !ret )
      {
      //throw new Exception("Could not read: " + filename );
      return 1;
      }

    return 0;
    }
}
