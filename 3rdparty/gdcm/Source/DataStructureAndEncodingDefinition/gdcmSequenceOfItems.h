/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMSEQUENCEOFITEMS_H
#define GDCMSEQUENCEOFITEMS_H

#include "gdcmValue.h"
#include "gdcmItem.h"

#include <vector>
#include <cstring> // strcmp

namespace gdcm
{

/**
 * \brief Class to represent a Sequence Of Items
 * (value representation : SQ)
 *  - a Value Representation for Data Elements that contains a sequence of Data Sets.
 *  - Sequence of Item allows for Nested Data Sets
 *
 * See PS 3.5, 7.4.6 Data Element Type Within a Sequence
 * \note
 * SEQUENCE OF ITEMS (VALUE REPRESENTATION SQ)
 * A Value Representation for Data Elements that contain a sequence of
 * Data Sets. Sequence of Items allows for Nested Data Sets.
 */
class GDCM_EXPORT SequenceOfItems : public Value
{
public:
  // Typdefs:
  typedef std::vector< Item > ItemVector;
  typedef ItemVector::size_type SizeType;
  typedef ItemVector::iterator Iterator;
  typedef ItemVector::const_iterator ConstIterator;
  Iterator Begin() { return Items.begin(); }
  Iterator End() { return Items.end(); }
  ConstIterator Begin() const { return Items.begin(); }
  ConstIterator End() const { return Items.end(); }

  /// \brief constructor (UndefinedLength by default)
  SequenceOfItems():SequenceLengthField(0xFFFFFFFF) { }
  //SequenceOfItems(VL const &vl = 0xFFFFFFFF):SequenceLengthField(vl),NType(type) { }

  /// \brief Returns the SQ length, as read from disk
  VL GetLength() const { return SequenceLengthField; }
  /// \brief Sets the actual SQ length
  void SetLength(VL length) {
    SequenceLengthField = length;
  }
  /// \brief Properly set the Sequence of Item to be undefined length
  void SetLengthToUndefined();
  /// return if Value Length if of undefined length
  bool IsUndefinedLength() const {
    return SequenceLengthField.IsUndefined();
  }

  template <typename TDE>
  VL ComputeLength() const;

  /// remove all items within the sequence
  void Clear();

  /// \brief Appends an Item to the already added ones
  void AddItem(Item const &item);

  /// Remove an Item as specified by its index, if index > size, false is returned
  /// Index starts at 1 not 0
  bool RemoveItemByIndex( const SizeType index );

  SizeType GetNumberOfItems() const {  return Items.size(); }
  void SetNumberOfItems(SizeType n) {  Items.resize(n); }

  /* WARNING: first item is #1 (see DICOM standard)
   *  Each Item shall be implicitly assigned an ordinal position starting with the value 1 for the
   * first Item in the Sequence, and incremented by 1 with each subsequent Item. The last Item in the
   * Sequence shall have an ordinal position equal to the number of Items in the Sequence.
   */
  const Item &GetItem(SizeType position) const;
  Item &GetItem(SizeType position);

  SequenceOfItems &operator=(const SequenceOfItems &val) {
    SequenceLengthField = val.SequenceLengthField;
    Items = val.Items;
    return *this;
    }

  template <typename TDE, typename TSwap>
  std::istream &Read(std::istream &is, bool readvalues = true)
    {
    (void)readvalues;
    const Tag seqDelItem(0xfffe,0xe0dd);
    if( SequenceLengthField.IsUndefined() )
      {
      Item item;
      while( item.Read<TDE,TSwap>(is) && item.GetTag() != seqDelItem )
        {
        //gdcmDebugMacro( "Item: " << item );
        assert( item.GetTag() != seqDelItem );
        Items.push_back( item );
        item.Clear();
        }
      //assert( item.GetTag() == seqDelItem && item.GetVL() == 0 );
      }
    else
      {
      Item item;
      VL l = 0;
      //is.seekg( SequenceLengthField, std::ios::cur ); return is;
      while( l != SequenceLengthField )
        {
        try
          {
          item.Read<TDE,TSwap>(is);
          }
        catch( Exception &ex )
          {
          if( strcmp( ex.GetDescription(), "Changed Length" ) == 0 )
            {
            VL newlength = l + item.template GetLength<TDE>();
            if( newlength > SequenceLengthField )
              {
              // BogugsItemAndSequenceLength.dcm
              gdcmWarningMacro( "SQ length is wrong" );
              SequenceLengthField = newlength;
              }
            }
          else
            {
            throw ex;
            }
          }
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
        if( item.GetTag() == seqDelItem )
          {
          gdcmWarningMacro( "SegDelItem found in defined length Sequence. Skipping" );
          assert( item.GetVL() == 0 );
          assert( item.GetNestedDataSet().Size() == 0 );
          // we need to pay attention that the length of the Sequence of Items will be wrong
          // this way. Indeed by not adding this item we are changing the size of this sqi
          }
        else // Not a seq del item marker
#endif
          {
          // By design we never load them. If we were to load those attribute
          // as normal item it would become very complex to convert a sequence
          // from defined length to undefined length with the risk to write two
          // seq del marker
          Items.push_back( item );
          }
        l += item.template GetLength<TDE>();
        if( l > SequenceLengthField )
          {
          gdcmDebugMacro( "Found: Length of Item larger than expected" )
          throw "Length of Item larger than expected";
          }
        assert( l <= SequenceLengthField );
        //std::cerr << "sqi debug len: " << is.tellg() << " " <<  l << " " <<  SequenceLengthField << std::endl;
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
        // MR_Philips_Intera_No_PrivateSequenceImplicitVR.dcm
        // (0x2005, 0x1080): for some reason computation of length fails...
        if( SequenceLengthField == 778 && l == 774 )
          {
          gdcmWarningMacro( "PMS: Super bad hack" );
          SequenceLengthField = l;
          throw Exception( "Wrong Length" );
          //l = SequenceLengthField;
          }
        // Bug_Philips_ItemTag_3F3F
        // (0x2005, 0x1080): Because we do not handle fully the bug at the item
        // level we need to check here too
        else if ( SequenceLengthField == 444 && l == 3*71 )
          {
          // This one is a double bug. Item length is wrong and impact SQ length
          gdcmWarningMacro( "PMS: Super bad hack" );
          l = SequenceLengthField;
          }
#endif
        }
      assert( l == SequenceLengthField );
      }
    return is;
    }

  template <typename TDE,typename TSwap>
  std::ostream const &Write(std::ostream &os) const
    {
    typename ItemVector::const_iterator it = Items.begin();
    for(;it != Items.end(); ++it)
      {
      it->Write<TDE,TSwap>(os);
      }
    if( SequenceLengthField.IsUndefined() )
      {
      // seq del item is not stored, write it !
      const Tag seqDelItem(0xfffe,0xe0dd);
      seqDelItem.Write<TSwap>(os);
      VL zero = 0;
      zero.Write<TSwap>(os);
      }

    return os;
    }

//protected:
  void Print(std::ostream &os) const {
    os << "\t(" << SequenceLengthField << ")\n";
    ItemVector::const_iterator it =
      Items.begin();
    for(;it != Items.end(); ++it)
      {
      os << "  " << *it;
      }
    if( SequenceLengthField.IsUndefined() )
      {
      const Tag seqDelItem(0xfffe,0xe0dd);
      VL zero = 0;
      os << seqDelItem;
      os << "\t" << zero;
      }
  }

  static SmartPointer<SequenceOfItems> New()
  {
     return new SequenceOfItems;
  }
  bool FindDataElement(const Tag &t) const;

  bool operator==(const Value &val) const
    {
    const SequenceOfItems &sqi = dynamic_cast<const SequenceOfItems&>(val);
    return SequenceLengthField == sqi.SequenceLengthField &&
      Items == sqi.Items;
    }

private:
public:
  /// \brief Total length of the Sequence (or 0xffffffff) if undefined
  VL SequenceLengthField;
  /// \brief Vector of Sequence Items
  ItemVector Items;
};

} // end namespace gdcm

#include "gdcmSequenceOfItems.txx"

#endif //GDCMSEQUENCEOFITEMS_H
