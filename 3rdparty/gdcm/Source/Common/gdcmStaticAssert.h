/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSTATICASSERT_H
#define GDCMSTATICASSERT_H


// the following was shamelessly borowed from BOOST static assert:
namespace gdcm
{
  template <bool x>
  struct STATIC_ASSERTION_FAILURE;

  template <>
  struct STATIC_ASSERTION_FAILURE<true> { enum { value = 1 }; };

  template <int x>
  struct static_assert_test {};
}

#define GDCM_JOIN( X, Y ) GDCM_DO_JOIN( X, Y )
#define GDCM_DO_JOIN( X, Y ) GDCM_DO_JOIN2(X,Y)
#define GDCM_DO_JOIN2( X, Y ) X##Y

/// The GDCM_JOIN  + __LINE__ is needed to create a uniq identifier
#define GDCM_STATIC_ASSERT( B ) \
  typedef ::gdcm::static_assert_test<\
    sizeof(::gdcm::STATIC_ASSERTION_FAILURE< (bool)( B ) >)>\
      GDCM_JOIN(gdcm_static_assert_typedef_, __LINE__)


/* Example of use:
 *
 * template <class T>
 * struct must_not_be_instantiated
 * {
 * // this will be triggered if this type is instantiated
 * GDCM_STATIC_ASSERT(sizeof(T) == 0);
 * };
 *
 */
#endif // GDCMSTATICASSERT_H
