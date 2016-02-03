/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMMESHPRIMITIVE_H
#define GDCMMESHPRIMITIVE_H

#include <gdcmObject.h>
#include <gdcmDataElement.h>

namespace gdcm
{

/**
  * \brief  This class defines surface mesh primitives.
  * It is designed from surface mesh primitives macro.
  *
  * \see  PS 3.3 C.27.4
  */
class GDCM_EXPORT MeshPrimitive : public Object
{
public:

  typedef std::vector< DataElement > PrimitivesData;

  /**
    * \brief  This enumeration defines primitive types.
    *
    * \see  PS 3.3 C.27.4.1
    */
    typedef enum {
        VERTEX = 0,
        EDGE,
        TRIANGLE,
        TRIANGLE_STRIP,
        TRIANGLE_FAN,
        LINE,
        FACET,
        MPType_END
        } MPType;

    static const char * GetMPTypeString(const MPType type);

    static MPType GetMPType(const char * type);

    MeshPrimitive();

    virtual ~MeshPrimitive();

    MPType GetPrimitiveType() const;
    void SetPrimitiveType(const MPType type);

    const DataElement & GetPrimitiveData() const;
    DataElement & GetPrimitiveData();
    void SetPrimitiveData(DataElement const & de);

    const PrimitivesData & GetPrimitivesData() const;
    PrimitivesData & GetPrimitivesData();
    void SetPrimitivesData(PrimitivesData const & DEs);

    const DataElement & GetPrimitiveData(const unsigned int idx) const;
    DataElement & GetPrimitiveData(const unsigned int idx);
    void SetPrimitiveData(const unsigned int idx, DataElement const & de);
    void AddPrimitiveData(DataElement const & de);

    unsigned int GetNumberOfPrimitivesData() const;

protected:

    // Use to define tag where PrimitiveData will be put.
    MPType          PrimitiveType;

    // PrimitiveData contains point index list.
    // It shall have 1 or 1-n DataElement following PrimitiveType.
    PrimitivesData  PrimitiveData;
};

}

#endif // GDCMMESHPRIMITIVE_H
