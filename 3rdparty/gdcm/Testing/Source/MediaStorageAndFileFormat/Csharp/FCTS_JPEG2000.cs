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

public class FCTS_JPEG2000
{
  public static int Main(string[] args)
    {
    string filename = args[0];
    string outfilename = args[1];

    using( var sfcts = FileChangeTransferSyntax.New() )
      {
      FileChangeTransferSyntax fcts = sfcts.__ref__();
      //SimpleSubjectWatcher watcher = new SimpleSubjectWatcher(fcts, "FileChangeTransferSyntax");
      gdcm.TransferSyntax ts = new TransferSyntax( TransferSyntax.TSType.JPEG2000 );
      fcts.SetTransferSyntax( ts );
      ImageCodec ic = fcts.GetCodec();
      JPEG2000Codec jpeg2000 = JPEG2000Codec.Cast( ic );
      jpeg2000.SetReversible( false );
      jpeg2000.SetQuality(0, 55);

      fcts.SetInputFileName( filename );
      fcts.SetOutputFileName( outfilename );
      if( !fcts.Change() )
        {
        return 1;
        }
      }

    return 0;
    }
}
