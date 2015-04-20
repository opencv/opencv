/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include <iostream>
#include <iomanip>
#include <vector>

#include <stdint.h>

struct AE
{
  char Internal[16+1];
  void Print(std::ostream &os) {
    os << Internal;
  }
};

typedef enum {
  DAY = 'D',
  WEEK = 'W',
  MONTH = 'M',
  YEAR = 'Y'
} DateFormat;

//template <int Format> struct ASValid;
//template <> struct ASValid<'Y'> { };
struct AS
{
  unsigned short N;
  unsigned char  Format;
  void Print(std::ostream &os) {
    if( N < 1000 &&
       (Format == 'D' ||
        Format == 'W' ||
        Format == 'M' ||
        Format == 'Y' ))
       {
    os << std::setw(3) << std::setfill('0') << N << Format;
    }
    else
    {
    os << "";
    }
  }
};

struct AT
{
  // Tag Internal;
};

struct CS
{
  char Internal[16];
};

struct DA
{
  unsigned short Year : 12;
  unsigned short Month : 4;
  unsigned short Day : 5;
  void Print(std::ostream &os)
  {
    os << std::setw(4) << std::setfill('0') << Year;
    os << std::setw(2) << std::setfill('0') << Month;
    os << std::setw(2) << std::setfill('0') << Day;
  }
};

struct date {
    unsigned short day   : 5;   // 1 to 31
         unsigned short month : 4;   // 1 to 12
         unsigned short year : 12;  /* 0 to 9999 (technically :11 should be enough for a couple of years...) */
  void Print(std::ostream &os)
  {
    os << year << "." << (int)month << "." << (int)day;

  }
  };

struct DS
{
  double Internal;
  // 16 bytes as integer would mean we can have 10^16 as max int
  // which only double can hold
};


struct FL
{
  float Internal;
};

struct FD
{
  double Internal;
};

struct IS
{
  int Internal;
  void Print(std::ostream &os)
  {
    os << Internal;
  }
};

struct LO
{
  char Internal[64];
  void Print(std::ostream &os) {
    os << Internal;
  }
};

struct LT
{
  std::string Internal;
  void Print(std::ostream &os) {
    os << Internal.size();
    os << std::endl;
    os << Internal;
  }
};


struct OB
{
  explicit OB(const char *array = 0, uint32_t length = 0):Internal(array),Length(length) {}
  void Print( std::ostream &os )
  {
    const char *p = Internal;
    const char *end = Internal+Length;
    while(p != end)
    {
      os << "\\" << (int)*p++;
    }
  }
private:
  const char *Internal;
  uint32_t Length;
};

struct OF;

struct OW;

struct PN
{
  char Component[5][64];
  void Print(std::ostream &os)
  {
    //os << "Family Name Complex: " << Component[0] << std::endl;
    //os << "Given  Name Complex: " << Component[1] << std::endl;
    //os << "Middle Name        : " << Component[2] << std::endl;
    //os << "Name Suffix        : " << Component[3] << std::endl;
    //os << "Name Prefix        : " << Component[4] << std::endl;
    os << Component[0] << "^";
    os << Component[1] << "^";
    os << Component[2] << "^";
    os << Component[3] << "^";
    os << Component[4];
  }
};

struct SH
{
  char Internal[16+1];
};

struct SL
{
  signed long Internal;
};

struct SQ;

struct SS
{
  signed short Internal;
};

struct ST
{
  std::string Internal;
};

struct TM
{
  unsigned short hours:5;
  unsigned short minutes:6;
  unsigned short seconds:6;
  unsigned int fracseconds:20;
  void Print(std::ostream &os) {
    os << std::setw(2) << std::setfill('0') << hours;
    os << std::setw(2) << std::setfill('0') << minutes;
    os << std::setw(2) << std::setfill('0') << seconds;
    os << ".";
    os << std::setw(6) << std::setfill('0') << fracseconds;
  }
};

struct Tri
{
  short Internal:2;
  //operator void *() const { return Internal; }
  operator short () const { return Internal; }
};

struct DT
{
  DA da;
  TM tm;
  Tri OffsetSign;
  unsigned short Hours:5;
  unsigned short Minutes:6;
// YYYYMMDDHHMMSS.FFFFFF&ZZZZ
  void Print ( std::ostream &os)
  {
    da.Print( os );
    tm.Print( os );
    if( OffsetSign )
    {
    if ( OffsetSign == -1 ) os << "-";
    else if ( OffsetSign == +1 ) os << "+";
    else return;
    os << std::setw(2) << std::setfill('0') << Hours;
    os << std::setw(2) << std::setfill('0') << Minutes;
    }
  }
};


struct UI
{
  char Internal[64+1];
  void Print(std::ostream &os) {
    os << strlen(Internal) << std::endl;
    os << Internal;
  }
};

struct UL
{
  unsigned long Internal;
};

struct UN
{
  std::vector<char> Internal;
};

struct US
{
  unsigned short Internal;
};

struct UT
{
  std::string Internal;
};

struct S
{
  union { char s[2]; } Internal;
  //unsigned short Internal;
  void Print( std::ostream &os )
  {
    os << strlen( Internal.s ) << std::endl;
    os << Internal.s;
  }
};

int main()
{
  AE ae = { "application enti" };
  ae.Print( std::cout );
  std::cout << std::endl;

  AS as = { 1, 'Y' };
  as.Print( std::cout );
  std::cout << std::endl;

  DA da = { 10978, 11, 32 };
  da.Print( std::cout );
  std::cout << std::endl;
  std::cout << "DA:" << sizeof (DA) << std::endl;

  double dd = 10*10*10*10*10*10*10*10;
  std::cout << dd << std::endl;
  DS ds = { 10*10*10*10*10*10*10*10 + 1  };
  std::cout << ds.Internal << std::endl;

  DT dt = { 1979, 7, 6 };
  dt.Print( std::cout << "DT: "  );
  std::cout << std::endl;

  DT dt1 = { 1979, 7, 6, 23, 59, 59, 999999 };
  dt1.Print( std::cout << "DT: "  );
  std::cout << std::endl;

  DT dt2 = { 1979, 7, 6, 23, 59, 59, 999999, -1, 12, 42 };
  dt2.Print( std::cout << "DT: "  );
  std::cout << std::endl;

  DT dt3 = { 1979, 7, 6, 23, 59, 59, 999999, +1, 12, 42 };
  dt3.Print( std::cout << "DT: "  );
  std::cout << std::endl;


  IS is = { 2 << 30 };
  is.Print( std::cout );
  std::cout << std::endl;

  //std::cout << 8*8 << std::endl;
  //std::cout << 8*8*8*8 << std::endl;
  date d = {31, 12, 1978 };
  d.Print( std::cout );
  std::cout << std::endl;
  std::cout << "date:" << sizeof (date) << std::endl;


  LO lo = { "coucou" };
  lo.Print( std::cout );
  std::cout << std::endl;

  LT lt = { "very long text\0hello mathieu" };
  lt.Print( std::cout );
  std::cout << std::endl;

  OB ob("\0\1", 2);
  ob.Print( std::cout );
  std::cout << std::endl;

  PN pn1 = { "abc123", "def", "ghi", "klm", "opq" };
  pn1.Print( std::cout );
  std::cout << std::endl;

  PN pn2 = { "malaterre", "mathieu olivier patrick"};
  pn2.Print( std::cout );
  std::cout << std::endl;

// Rev. John Robert Quincy Adams, B.A. M.Div. “Adams^John Robert Quincy^^Rev.^B.A. M.Div.” [One family name; three given names; no middle name; one prefix; two suffixes.]
        PN pn3 = { "Adams", "John Robert Quincy", "", "Rev.", "B.A. M.Div." };
  pn3.Print( std::cout );
  std::cout << std::endl;
// Susan Morrison-Jones, Ph.D., Chief Executive Officer “Morrison-Jones^Susan^^^Ph.D., Chief Executive Officer” [Two family names; one given name; no middle name; no prefix; two suffixes.]
        PN pn4 = { "Morrison-Jones", "Susan", "", "", "Ph.D., Chief Executive Officer" };
  pn4.Print( std::cout );
  std::cout << std::endl;

// John Doe “Doe^John” [One family name; one given name; no middle name, prefix, or suffix. Delimiters have been omitted for the three trailing null components.]
        PN pn5 = { "Doe", "John" };
  pn5.Print( std::cout );
  std::cout << std::endl;


// (for examples of the encoding of Person Names using multi-byte character sets see Annex H)
// Smith^Fluffy [A cat, rather than a
//human, whose responsible party family name is Smith, and whose own name is Fluffy]
        PN pn6 = { "Smith", "Fluffy" };
  pn6.Print( std::cout );
  std::cout << std::endl;
//ABC Farms^Running on Water [A horse whose responsible organization is named ABC Farms, and whose name is “Running On Water”]
        PN pn7 = { "ABC Farms", "Running on Water" };
  pn7.Print( std::cout );
  std::cout << std::endl;




  std::cout << "TM:" << sizeof (TM) << std::endl;
  TM tm1 = { 7, 12, 49, 999999 };
  tm1.Print( std::cout );
  std::cout << std::endl;

  TM tm2 = { 23, 59, 59, 999999 };
  tm2.Print( std::cout );
  std::cout << std::endl;

  TM tm3 = { 0, 0, 0, 0 };
  tm3.Print( std::cout );
  std::cout << std::endl;


  UI ui = {
  "1234567890.1234567890.1234567890.1234567890.1234567890.123456789" };
  ui.Print( std::cout );
  std::cout << std::endl;

  //S s = "10";
  S s = { '1' };
  s.Print(std::cout);
  std::cout << std::endl;

  std::cout << "Tri:" << std::endl;
  Tri t1 = { 0 };
  std::cout << t1.Internal << std::endl;
  Tri t2 = { 1 };
  std::cout << t2.Internal << std::endl;
  Tri t3 = { 2 };
  std::cout << t3.Internal << std::endl;
  Tri t4 = { 3 };
  std::cout << t4.Internal << std::endl;
  Tri t5 = { '+' };
  std::cout << t5.Internal << std::endl;
  Tri t6 = { '-' };
  std::cout << t6.Internal << std::endl;
  Tri t7 = { -1 };
  std::cout << t7.Internal << std::endl;

  //DateFormat df = 'Y';

  return 0;
}
