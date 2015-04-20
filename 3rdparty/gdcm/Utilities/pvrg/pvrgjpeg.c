/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

/* simple wrapper around the fake pvrg 'main' function, so that we can still
 * build the library and a separate executable
 */

/* forward declaration */
int pvrgmain(int argc, char *argv[]);

int main(int argc, char *argv[])
{
  return pvrgmain(argc, argv);
}
