
// File: index.xml

// File: classgdcm_1_1AES.xml
%typemap("csclassmodifiers") gdcm::AES " /**C++ includes: gdcmAES.h */
public class";

%csmethodmodifiers  gdcm::AES::AES " /** gdcm::AES::AES()  */ public";

%csmethodmodifiers  gdcm::AES::CryptCbc " /** bool
gdcm::AES::CryptCbc(int mode, unsigned int length, char iv[16], const
char *input, char *output) const

AES-CBC buffer encryption/decryption.

Parameters:
-----------

mode:  ENCRYPT or DECRYPT

length:  length of the input data

iv:  initialization vector (updated after use)

input:  buffer holding the input data

output:  buffer holding the output data

false on error (invalid key, or length not multiple 16)

*/ public";

%csmethodmodifiers  gdcm::AES::CryptCfb128 " /** bool
gdcm::AES::CryptCfb128(int mode, unsigned int length, unsigned int
&iv_off, char iv[16], const char *input, char *output) const

AES-CFB128 buffer encryption/decryption.

Parameters:
-----------

mode:  ENCRYPT or DECRYPT

length:  length of the input data

iv_off:  offset in IV (updated after use)

iv:  initialization vector (updated after use)

input:  buffer holding the input data

output:  buffer holding the output data

false on error (invalid key)

*/ public";

%csmethodmodifiers  gdcm::AES::CryptEcb " /** bool
gdcm::AES::CryptEcb(int mode, const char input[16], char output[16])
const

AES-ECB block encryption/decryption.

Parameters:
-----------

mode:  ENCRYPT or DECRYPT

input:  16-byte input block

output:  16-byte output block

false on error (invalid key)

*/ public";

%csmethodmodifiers  gdcm::AES::SetkeyDec " /** bool
gdcm::AES::SetkeyDec(const char *key, unsigned int keysize)

AES key schedule (decryption).

Parameters:
-----------

key:  decryption key

keysize:  must be 16, 24 or 32

false on error (wrong keysize, key null)

*/ public";

%csmethodmodifiers  gdcm::AES::SetkeyEnc " /** bool
gdcm::AES::SetkeyEnc(const char *key, unsigned int keysize)

AES key schedule (encryption).

Parameters:
-----------

key:  encryption key

keysize:  must be 16, 24 or 32

false on error (wrong keysize, key null)

*/ public";

%csmethodmodifiers  gdcm::AES::~AES " /** gdcm::AES::~AES()  */
public";


// File: classstd_1_1allocator.xml
%typemap("csclassmodifiers") std::allocator " /** STL class.

*/ public class";


// File: classgdcm_1_1Anonymizer.xml
%typemap("csclassmodifiers") gdcm::Anonymizer " /**C++ includes:
gdcmAnonymizer.h */ public class";

%csmethodmodifiers  gdcm::Anonymizer::Anonymizer " /**
gdcm::Anonymizer::Anonymizer()  */ public";

%csmethodmodifiers
gdcm::Anonymizer::BasicApplicationLevelConfidentialityProfile " /**
bool
gdcm::Anonymizer::BasicApplicationLevelConfidentialityProfile(bool
deidentify=true)

PS 3.15 / E.1.1 De-Identifier An Application may claim conformance to
the Basic Application Level Confidentiality Profile as a deidentifier
if it protects all Attributes that might be used by unauthorized
entities to identify the patient. NOT THREAD SAFE

*/ public";

%csmethodmodifiers  gdcm::Anonymizer::Empty " /** bool
gdcm::Anonymizer::Empty(Tag const &t)

Make Tag t empty (if not found tag will be created) Warning: does not
handle SQ element

*/ public";

%csmethodmodifiers  gdcm::Anonymizer::GetFile " /** File&
gdcm::Anonymizer::GetFile()  */ public";

%csmethodmodifiers  gdcm::Anonymizer::GetX509 " /** const X509*
gdcm::Anonymizer::GetX509() const  */ public";

%csmethodmodifiers  gdcm::Anonymizer::Remove " /** bool
gdcm::Anonymizer::Remove(Tag const &t)

remove a tag (even a SQ can be removed)

*/ public";

%csmethodmodifiers  gdcm::Anonymizer::RemoveGroupLength " /** bool
gdcm::Anonymizer::RemoveGroupLength()

Main function that loop over all elements and remove group length.

*/ public";

%csmethodmodifiers  gdcm::Anonymizer::RemovePrivateTags " /** bool
gdcm::Anonymizer::RemovePrivateTags()

Main function that loop over all elements and remove private tags.

*/ public";

%csmethodmodifiers  gdcm::Anonymizer::RemoveRetired " /** bool
gdcm::Anonymizer::RemoveRetired()

Main function that loop over all elements and remove retired element.

*/ public";

%csmethodmodifiers  gdcm::Anonymizer::Replace " /** bool
gdcm::Anonymizer::Replace(Tag const &t, const char *value, VL const
&vl)

when the value contains , it is a good idea to specify the length.
This function is required when dealing with VRBINARY tag

*/ public";

%csmethodmodifiers  gdcm::Anonymizer::Replace " /** bool
gdcm::Anonymizer::Replace(Tag const &t, const char *value)

Replace tag with another value, if tag is not found it will be
created: WARNING: this function can only execute if tag is a VRASCII

*/ public";

%csmethodmodifiers  gdcm::Anonymizer::SetFile " /** void
gdcm::Anonymizer::SetFile(const File &f)

Set/Get File.

*/ public";

%csmethodmodifiers  gdcm::Anonymizer::SetX509 " /** void
gdcm::Anonymizer::SetX509(X509 *x509)

Set/Get AES key that will be used to encrypt the dataset within
BasicApplicationLevelConfidentialityProfile Warning: set is done by
copy (not reference)

*/ public";

%csmethodmodifiers  gdcm::Anonymizer::~Anonymizer " /**
gdcm::Anonymizer::~Anonymizer()  */ public";


// File: classgdcm_1_1ApplicationEntity.xml
%typemap("csclassmodifiers") gdcm::ApplicationEntity " /**
ApplicationEntity AE Application Entity

A string of characters that identifies an Application Entity with
leading and trailing spaces (20H) being non-significant. A value
consisting solely of spaces shall not be used.

Default Character Repertoire excluding character code 5CH (the
BACKSLASH \\\\ in ISO-IR 6), and control characters LF, FF, CR and
ESC.

16 bytes maximum.

C++ includes: gdcmApplicationEntity.h */ public class";

%csmethodmodifiers  gdcm::ApplicationEntity::IsValid " /** bool
gdcm::ApplicationEntity::IsValid() const  */ public";

%csmethodmodifiers  gdcm::ApplicationEntity::Print " /** void
gdcm::ApplicationEntity::Print(std::ostream &os) const  */ public";

%csmethodmodifiers  gdcm::ApplicationEntity::SetBlob " /** void
gdcm::ApplicationEntity::SetBlob(const std::vector< char > &v)  */
public";

%csmethodmodifiers  gdcm::ApplicationEntity::Squeeze " /** void
gdcm::ApplicationEntity::Squeeze()  */ public";


// File: classgdcm_1_1ASN1.xml
%typemap("csclassmodifiers") gdcm::ASN1 " /**C++ includes: gdcmASN1.h
*/ public class";

%csmethodmodifiers  gdcm::ASN1::ASN1 " /** gdcm::ASN1::ASN1()  */
public";

%csmethodmodifiers  gdcm::ASN1::TestPBKDF2 " /** int
gdcm::ASN1::TestPBKDF2()  */ public";

%csmethodmodifiers  gdcm::ASN1::~ASN1 " /** gdcm::ASN1::~ASN1()  */
public";


// File: classgdcm_1_1Attribute.xml
%typemap("csclassmodifiers") gdcm::Attribute " /**  Attribute class
This class use template metaprograming tricks to let the user know
when the template instanciation does not match the public dictionary.

Typical example that compile is: Attribute<0x0008,0x9007> a =
{\"ORIGINAL\",\"PRIMARY\",\"T1\",\"NONE\"};

Examples that will NOT compile are:

Attribute<0x0018,0x1182, VR::IS, VM::VM1> fd1 = {}; // not enough
parameters Attribute<0x0018,0x1182, VR::IS, VM::VM2> fd2 = {0,1,2}; //
too many initializers Attribute<0x0018,0x1182, VR::IS, VM::VM3> fd3 =
{0,1,2}; // VM3 is not valid Attribute<0x0018,0x1182, VR::UL, VM::VM2>
fd3 = {0,1}; // UL is not valid VR

C++ includes: gdcmAttribute.h */ public class";

%csmethodmodifiers  gdcm::Attribute::GDCM_STATIC_ASSERT " /**
gdcm::Attribute< Group, Element, TVR, TVM
>::GDCM_STATIC_ASSERT(((((VR::VRType) TVR &VR::VR_VM1)&&((VM::VMType)
TVM==VM::VM1))||!((VR::VRType) TVR &VR::VR_VM1)))  */ public";

%csmethodmodifiers  gdcm::Attribute::GDCM_STATIC_ASSERT " /**
gdcm::Attribute< Group, Element, TVR, TVM
>::GDCM_STATIC_ASSERT(((VM::VMType) TVM &(VM::VMType)(TagToType<
Group, Element >::VMType)))  */ public";

%csmethodmodifiers  gdcm::Attribute::GDCM_STATIC_ASSERT " /**
gdcm::Attribute< Group, Element, TVR, TVM
>::GDCM_STATIC_ASSERT(((VR::VRType) TVR &(VR::VRType)(TagToType<
Group, Element >::VRType)))  */ public";

%csmethodmodifiers  gdcm::Attribute::GetAsDataElement " /**
DataElement gdcm::Attribute< Group, Element, TVR, TVM
>::GetAsDataElement() const  */ public";

%csmethodmodifiers  gdcm::Attribute::GetNumberOfValues " /** unsigned
int gdcm::Attribute< Group, Element, TVR, TVM >::GetNumberOfValues()
const  */ public";

%csmethodmodifiers  gdcm::Attribute::GetValue " /** ArrayType const&
gdcm::Attribute< Group, Element, TVR, TVM >::GetValue(unsigned int
idx=0) const  */ public";

%csmethodmodifiers  gdcm::Attribute::GetValue " /** ArrayType&
gdcm::Attribute< Group, Element, TVR, TVM >::GetValue(unsigned int
idx=0)  */ public";

%csmethodmodifiers  gdcm::Attribute::GetValues " /** const ArrayType*
gdcm::Attribute< Group, Element, TVR, TVM >::GetValues() const  */
public";

%csmethodmodifiers  gdcm::Attribute::Print " /** void gdcm::Attribute<
Group, Element, TVR, TVM >::Print(std::ostream &os) const  */ public";

%csmethodmodifiers  gdcm::Attribute::Set " /** void gdcm::Attribute<
Group, Element, TVR, TVM >::Set(DataSet const &ds)  */ public";

%csmethodmodifiers  gdcm::Attribute::SetFromDataElement " /** void
gdcm::Attribute< Group, Element, TVR, TVM
>::SetFromDataElement(DataElement const &de)  */ public";

%csmethodmodifiers  gdcm::Attribute::SetValue " /** void
gdcm::Attribute< Group, Element, TVR, TVM >::SetValue(ArrayType v,
unsigned int idx=0)  */ public";

%csmethodmodifiers  gdcm::Attribute::SetValues " /** void
gdcm::Attribute< Group, Element, TVR, TVM >::SetValues(const ArrayType
*array, unsigned int numel=VMType)  */ public";


// File: classgdcm_1_1Attribute_3_01Group_00_01Element_00_01TVR_00_01VM_1_1VM1__n_01_4.xml
%typemap("csclassmodifiers") gdcm::Attribute< Group, Element, TVR,
VM::VM1_n > " /**C++ includes: gdcmAttribute.h */ public class";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::Attribute " /** gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::Attribute()  */ public";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::GDCM_STATIC_ASSERT " /** gdcm::Attribute< Group, Element, TVR,
VM::VM1_n >::GDCM_STATIC_ASSERT(((((VR::VRType) TVR
&VR::VR_VM1)&&((VM::VMType) TagToType< Group, Element
>::VMType==VM::VM1))||!((VR::VRType) TVR &VR::VR_VM1)))  */ public";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::GDCM_STATIC_ASSERT " /** gdcm::Attribute< Group, Element, TVR,
VM::VM1_n >::GDCM_STATIC_ASSERT((VM::VM1_n &(VM::VMType)(TagToType<
Group, Element >::VMType)))  */ public";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::GDCM_STATIC_ASSERT " /** gdcm::Attribute< Group, Element, TVR,
VM::VM1_n >::GDCM_STATIC_ASSERT(((VR::VRType) TVR
&(VR::VRType)(TagToType< Group, Element >::VRType)))  */ public";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::GetAsDataElement " /** DataElement gdcm::Attribute< Group, Element,
TVR, VM::VM1_n >::GetAsDataElement() const  */ public";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::GetNumberOfValues " /** unsigned int gdcm::Attribute< Group,
Element, TVR, VM::VM1_n >::GetNumberOfValues() const  */ public";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::GetValue " /** ArrayType const& gdcm::Attribute< Group, Element,
TVR, VM::VM1_n >::GetValue(unsigned int idx=0) const  */ public";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::GetValue " /** ArrayType& gdcm::Attribute< Group, Element, TVR,
VM::VM1_n >::GetValue(unsigned int idx=0)  */ public";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::GetValues " /** const ArrayType* gdcm::Attribute< Group, Element,
TVR, VM::VM1_n >::GetValues() const  */ public";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::Print " /** void gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::Print(std::ostream &os) const  */ public";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::SetFromDataElement " /** void gdcm::Attribute< Group, Element, TVR,
VM::VM1_n >::SetFromDataElement(DataElement const &de)  */ public";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::SetNumberOfValues " /** void gdcm::Attribute< Group, Element, TVR,
VM::VM1_n >::SetNumberOfValues(unsigned int numel)  */ public";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::SetValue " /** void gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::SetValue(ArrayType v)  */ public";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::SetValue " /** void gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::SetValue(unsigned int idx, ArrayType v)  */ public";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::SetValues " /** void gdcm::Attribute< Group, Element, TVR,
VM::VM1_n >::SetValues(const ArrayType *array, unsigned int numel,
bool own=false)  */ public";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::~Attribute " /** gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::~Attribute()  */ public";


// File: classgdcm_1_1Attribute_3_01Group_00_01Element_00_01TVR_00_01VM_1_1VM2__2n_01_4.xml
%typemap("csclassmodifiers") gdcm::Attribute< Group, Element, TVR,
VM::VM2_2n > " /**C++ includes: gdcmAttribute.h */ public class";


// File: classgdcm_1_1Attribute_3_01Group_00_01Element_00_01TVR_00_01VM_1_1VM2__n_01_4.xml
%typemap("csclassmodifiers") gdcm::Attribute< Group, Element, TVR,
VM::VM2_n > " /**C++ includes: gdcmAttribute.h */ public class";

%csmethodmodifiers  gdcm::Attribute< Group, Element, TVR, VM::VM2_n
>::GetVM " /** VM gdcm::Attribute< Group, Element, TVR, VM::VM2_n
>::GetVM() const  */ public";


// File: classgdcm_1_1Attribute_3_01Group_00_01Element_00_01TVR_00_01VM_1_1VM3__3n_01_4.xml
%typemap("csclassmodifiers") gdcm::Attribute< Group, Element, TVR,
VM::VM3_3n > " /**C++ includes: gdcmAttribute.h */ public class";


// File: classgdcm_1_1Attribute_3_01Group_00_01Element_00_01TVR_00_01VM_1_1VM3__n_01_4.xml
%typemap("csclassmodifiers") gdcm::Attribute< Group, Element, TVR,
VM::VM3_n > " /**C++ includes: gdcmAttribute.h */ public class";


// File: classgdcm_1_1AudioCodec.xml
%typemap("csclassmodifiers") gdcm::AudioCodec " /**  AudioCodec.

C++ includes: gdcmAudioCodec.h */ public class";

%csmethodmodifiers  gdcm::AudioCodec::AudioCodec " /**
gdcm::AudioCodec::AudioCodec()  */ public";

%csmethodmodifiers  gdcm::AudioCodec::CanCode " /** bool
gdcm::AudioCodec::CanCode(TransferSyntax const &) const

Return whether this coder support this transfer syntax (can code it).

*/ public";

%csmethodmodifiers  gdcm::AudioCodec::CanDecode " /** bool
gdcm::AudioCodec::CanDecode(TransferSyntax const &) const

Return whether this decoder support this transfer syntax (can decode
it).

*/ public";

%csmethodmodifiers  gdcm::AudioCodec::Decode " /** bool
gdcm::AudioCodec::Decode(DataElement const &is, DataElement &os)

Decode.

*/ public";

%csmethodmodifiers  gdcm::AudioCodec::~AudioCodec " /**
gdcm::AudioCodec::~AudioCodec()  */ public";


// File: classstd_1_1auto__ptr.xml
%typemap("csclassmodifiers") std::auto_ptr " /** STL class.

*/ public class";


// File: classstd_1_1bad__alloc.xml
%typemap("csclassmodifiers") std::bad_alloc " /** STL class.

*/ public class";


// File: classstd_1_1bad__cast.xml
%typemap("csclassmodifiers") std::bad_cast " /** STL class.

*/ public class";


// File: classstd_1_1bad__exception.xml
%typemap("csclassmodifiers") std::bad_exception " /** STL class.

*/ public class";


// File: classstd_1_1bad__typeid.xml
%typemap("csclassmodifiers") std::bad_typeid " /** STL class.

*/ public class";


// File: classgdcm_1_1Base64.xml
%typemap("csclassmodifiers") gdcm::Base64 " /**C++ includes:
gdcmBase64.h */ public class";

%csmethodmodifiers  gdcm::Base64::Base64 " /** gdcm::Base64::Base64()
*/ public";

%csmethodmodifiers  gdcm::Base64::~Base64 " /**
gdcm::Base64::~Base64()  */ public";


// File: classstd_1_1basic__fstream.xml
%typemap("csclassmodifiers") std::basic_fstream " /** STL class.

*/ public class";


// File: classstd_1_1basic__ifstream.xml
%typemap("csclassmodifiers") std::basic_ifstream " /** STL class.

*/ public class";


// File: classstd_1_1basic__ios.xml
%typemap("csclassmodifiers") std::basic_ios " /** STL class.

*/ public class";


// File: classstd_1_1basic__iostream.xml
%typemap("csclassmodifiers") std::basic_iostream " /** STL class.

*/ public class";


// File: classstd_1_1basic__istream.xml
%typemap("csclassmodifiers") std::basic_istream " /** STL class.

*/ public class";


// File: classstd_1_1basic__istringstream.xml
%typemap("csclassmodifiers") std::basic_istringstream " /** STL class.

*/ public class";


// File: classstd_1_1basic__ofstream.xml
%typemap("csclassmodifiers") std::basic_ofstream " /** STL class.

*/ public class";


// File: classstd_1_1basic__ostream.xml
%typemap("csclassmodifiers") std::basic_ostream " /** STL class.

*/ public class";


// File: classstd_1_1basic__ostringstream.xml
%typemap("csclassmodifiers") std::basic_ostringstream " /** STL class.

*/ public class";


// File: classstd_1_1basic__string.xml
%typemap("csclassmodifiers") std::basic_string " /** STL class.

*/ public class";


// File: classstd_1_1basic__string_1_1const__iterator.xml
%typemap("csclassmodifiers") std::basic_string::const_iterator " /**
STL iterator class.

*/ public class";


// File: classstd_1_1basic__string_1_1const__reverse__iterator.xml
%typemap("csclassmodifiers") std::basic_string::const_reverse_iterator
" /** STL iterator class.

*/ public class";


// File: classstd_1_1basic__string_1_1iterator.xml
%typemap("csclassmodifiers") std::basic_string::iterator " /** STL
iterator class.

*/ public class";


// File: classstd_1_1basic__string_1_1reverse__iterator.xml
%typemap("csclassmodifiers") std::basic_string::reverse_iterator " /**
STL iterator class.

*/ public class";


// File: classstd_1_1basic__stringstream.xml
%typemap("csclassmodifiers") std::basic_stringstream " /** STL class.

*/ public class";


// File: classzlib__stream_1_1basic__unzip__streambuf.xml
%typemap("csclassmodifiers") zlib_stream::basic_unzip_streambuf " /**
A stream decorator that takes compressed input and unzips it to a
istream.

The class wraps up the deflate method of the zlib library
1.1.4http://www.gzip.org/zlib/

C++ includes: zipstreamimpl.h */ public class";

%csmethodmodifiers
zlib_stream::basic_unzip_streambuf::basic_unzip_streambuf " /**
zlib_stream::basic_unzip_streambuf< charT, traits
>::basic_unzip_streambuf(istream_reference istream, int window_size,
size_t read_buffer_size, size_t input_buffer_size)

Construct a unzip stream More info on the following parameters can be
found in the zlib documentation.

*/ public";

%csmethodmodifiers  zlib_stream::basic_unzip_streambuf::get_crc " /**
unsigned long zlib_stream::basic_unzip_streambuf< charT, traits
>::get_crc(void) const  */ public";

%csmethodmodifiers  zlib_stream::basic_unzip_streambuf::get_in_size "
/** long zlib_stream::basic_unzip_streambuf< charT, traits
>::get_in_size(void) const  */ public";

%csmethodmodifiers  zlib_stream::basic_unzip_streambuf::get_istream "
/** istream_reference zlib_stream::basic_unzip_streambuf< charT,
traits >::get_istream(void)

returns the compressed input istream

*/ public";

%csmethodmodifiers  zlib_stream::basic_unzip_streambuf::get_out_size "
/** long zlib_stream::basic_unzip_streambuf< charT, traits
>::get_out_size(void) const  */ public";

%csmethodmodifiers  zlib_stream::basic_unzip_streambuf::get_zerr " /**
int zlib_stream::basic_unzip_streambuf< charT, traits
>::get_zerr(void) const  */ public";

%csmethodmodifiers  zlib_stream::basic_unzip_streambuf::get_zip_stream
" /** z_stream& zlib_stream::basic_unzip_streambuf< charT, traits
>::get_zip_stream(void)  */ public";

%csmethodmodifiers  zlib_stream::basic_unzip_streambuf::underflow "
/** int_type zlib_stream::basic_unzip_streambuf< charT, traits
>::underflow(void)  */ public";

%csmethodmodifiers
zlib_stream::basic_unzip_streambuf::~basic_unzip_streambuf " /**
zlib_stream::basic_unzip_streambuf< charT, traits
>::~basic_unzip_streambuf(void)  */ public";


// File: classzlib__stream_1_1basic__zip__istream.xml
%typemap("csclassmodifiers") zlib_stream::basic_zip_istream " /**C++
includes: zipstreamimpl.h */ public class";

%csmethodmodifiers  zlib_stream::basic_zip_istream::basic_zip_istream
" /** zlib_stream::basic_zip_istream< charT, traits
>::basic_zip_istream(istream_reference istream, int window_size=-15,
size_t read_buffer_size=zstream_default_buffer_size, size_t
input_buffer_size=zstream_default_buffer_size)  */ public";

%csmethodmodifiers  zlib_stream::basic_zip_istream::check_crc " /**
bool zlib_stream::basic_zip_istream< charT, traits >::check_crc(void)
*/ public";

%csmethodmodifiers  zlib_stream::basic_zip_istream::check_data_size "
/** bool zlib_stream::basic_zip_istream< charT, traits
>::check_data_size(void) const  */ public";

%csmethodmodifiers  zlib_stream::basic_zip_istream::get_gzip_crc " /**
long zlib_stream::basic_zip_istream< charT, traits
>::get_gzip_crc(void) const  */ public";

%csmethodmodifiers  zlib_stream::basic_zip_istream::get_gzip_data_size
" /** long zlib_stream::basic_zip_istream< charT, traits
>::get_gzip_data_size(void) const  */ public";

%csmethodmodifiers  zlib_stream::basic_zip_istream::is_gzip " /** bool
zlib_stream::basic_zip_istream< charT, traits >::is_gzip(void) const
*/ public";


// File: classzlib__stream_1_1basic__zip__ostream.xml
%typemap("csclassmodifiers") zlib_stream::basic_zip_ostream " /**C++
includes: zipstreamimpl.h */ public class";

%csmethodmodifiers  zlib_stream::basic_zip_ostream::basic_zip_ostream
" /** zlib_stream::basic_zip_ostream< charT, traits
>::basic_zip_ostream(ostream_reference ostream, bool is_gzip=false,
int level=Z_DEFAULT_COMPRESSION, EStrategy strategy=DefaultStrategy,
int window_size=-15, int memory_level=8, size_t
buffer_size=zstream_default_buffer_size)  */ public";

%csmethodmodifiers  zlib_stream::basic_zip_ostream::finished " /**
void zlib_stream::basic_zip_ostream< charT, traits >::finished(void)
*/ public";

%csmethodmodifiers  zlib_stream::basic_zip_ostream::is_gzip " /** bool
zlib_stream::basic_zip_ostream< charT, traits >::is_gzip(void) const
*/ public";

%csmethodmodifiers  zlib_stream::basic_zip_ostream::zflush " /**
basic_zip_ostream<charT, traits>& zlib_stream::basic_zip_ostream<
charT, traits >::zflush(void)  */ public";

%csmethodmodifiers  zlib_stream::basic_zip_ostream::~basic_zip_ostream
" /** zlib_stream::basic_zip_ostream< charT, traits
>::~basic_zip_ostream(void)  */ public";


// File: classzlib__stream_1_1basic__zip__streambuf.xml
%typemap("csclassmodifiers") zlib_stream::basic_zip_streambuf " /** A
stream decorator that takes raw input and zips it to a ostream.

The class wraps up the inflate method of the zlib library
1.1.4http://www.gzip.org/zlib/

C++ includes: zipstreamimpl.h */ public class";

%csmethodmodifiers
zlib_stream::basic_zip_streambuf::basic_zip_streambuf " /**
zlib_stream::basic_zip_streambuf< charT, traits
>::basic_zip_streambuf(ostream_reference ostream, int level, EStrategy
strategy, int window_size, int memory_level, size_t buffer_size)  */
public";

%csmethodmodifiers  zlib_stream::basic_zip_streambuf::flush " /**
std::streamsize zlib_stream::basic_zip_streambuf< charT, traits
>::flush(void)  */ public";

%csmethodmodifiers  zlib_stream::basic_zip_streambuf::get_crc " /**
unsigned long zlib_stream::basic_zip_streambuf< charT, traits
>::get_crc(void) const  */ public";

%csmethodmodifiers  zlib_stream::basic_zip_streambuf::get_in_size "
/** unsigned long zlib_stream::basic_zip_streambuf< charT, traits
>::get_in_size(void) const  */ public";

%csmethodmodifiers  zlib_stream::basic_zip_streambuf::get_ostream "
/** ostream_reference zlib_stream::basic_zip_streambuf< charT, traits
>::get_ostream(void) const  */ public";

%csmethodmodifiers  zlib_stream::basic_zip_streambuf::get_out_size "
/** long zlib_stream::basic_zip_streambuf< charT, traits
>::get_out_size(void) const  */ public";

%csmethodmodifiers  zlib_stream::basic_zip_streambuf::get_zerr " /**
int zlib_stream::basic_zip_streambuf< charT, traits >::get_zerr(void)
const  */ public";

%csmethodmodifiers  zlib_stream::basic_zip_streambuf::overflow " /**
int_type zlib_stream::basic_zip_streambuf< charT, traits
>::overflow(int_type c)  */ public";

%csmethodmodifiers  zlib_stream::basic_zip_streambuf::sync " /** int
zlib_stream::basic_zip_streambuf< charT, traits >::sync(void)  */
public";

%csmethodmodifiers
zlib_stream::basic_zip_streambuf::~basic_zip_streambuf " /**
zlib_stream::basic_zip_streambuf< charT, traits
>::~basic_zip_streambuf(void)  */ public";


// File: classgdcm_1_1BasicOffsetTable.xml
%typemap("csclassmodifiers") gdcm::BasicOffsetTable " /** Class to
represent a BasicOffsetTable.

C++ includes: gdcmBasicOffsetTable.h */ public class";

%csmethodmodifiers  gdcm::BasicOffsetTable::BasicOffsetTable " /**
gdcm::BasicOffsetTable::BasicOffsetTable()  */ public";

%csmethodmodifiers  gdcm::BasicOffsetTable::Read " /** std::istream&
gdcm::BasicOffsetTable::Read(std::istream &is)  */ public";


// File: classgdcm_1_1Bitmap.xml
%typemap("csclassmodifiers") gdcm::Bitmap " /**  Bitmap class A bitmap
based image. Used as parent for both IconImage and the main Pixel Data
Image It does not contains any World Space information (IPP, IOP).

C++ includes: gdcmBitmap.h */ public class";

%csmethodmodifiers  gdcm::Bitmap::AreOverlaysInPixelData " /** virtual
bool gdcm::Bitmap::AreOverlaysInPixelData() const  */ public";

%csmethodmodifiers  gdcm::Bitmap::Bitmap " /** gdcm::Bitmap::Bitmap()
*/ public";

%csmethodmodifiers  gdcm::Bitmap::Clear " /** void
gdcm::Bitmap::Clear()  */ public";

%csmethodmodifiers  gdcm::Bitmap::GetBuffer " /** bool
gdcm::Bitmap::GetBuffer(char *buffer) const

Acces the raw data.

*/ public";

%csmethodmodifiers  gdcm::Bitmap::GetBufferLength " /** unsigned long
gdcm::Bitmap::GetBufferLength() const

Return the length of the image after decompression WARNING for palette
color: It will NOT take into account the Palette Color thus you need
to multiply this length by 3 if computing the size of equivalent RGB
image

*/ public";

%csmethodmodifiers  gdcm::Bitmap::GetColumns " /** unsigned int
gdcm::Bitmap::GetColumns() const  */ public";

%csmethodmodifiers  gdcm::Bitmap::GetDataElement " /** DataElement&
gdcm::Bitmap::GetDataElement()  */ public";

%csmethodmodifiers  gdcm::Bitmap::GetDataElement " /** const
DataElement& gdcm::Bitmap::GetDataElement() const  */ public";

%csmethodmodifiers  gdcm::Bitmap::GetDimension " /** unsigned int
gdcm::Bitmap::GetDimension(unsigned int idx) const  */ public";

%csmethodmodifiers  gdcm::Bitmap::GetDimensions " /** const unsigned
int* gdcm::Bitmap::GetDimensions() const

Return the dimension of the pixel data, first dimension (x), then 2nd
(y), then 3rd (z)...

*/ public";

%csmethodmodifiers  gdcm::Bitmap::GetLUT " /** LookupTable&
gdcm::Bitmap::GetLUT()  */ public";

%csmethodmodifiers  gdcm::Bitmap::GetLUT " /** const LookupTable&
gdcm::Bitmap::GetLUT() const  */ public";

%csmethodmodifiers  gdcm::Bitmap::GetNeedByteSwap " /** bool
gdcm::Bitmap::GetNeedByteSwap() const  */ public";

%csmethodmodifiers  gdcm::Bitmap::GetNumberOfDimensions " /** unsigned
int gdcm::Bitmap::GetNumberOfDimensions() const

Return the number of dimension of the pixel data bytes; for example 2
for a 2D matrices of values.

*/ public";

%csmethodmodifiers  gdcm::Bitmap::GetPhotometricInterpretation " /**
const PhotometricInterpretation&
gdcm::Bitmap::GetPhotometricInterpretation() const

return the photometric interpretation

*/ public";

%csmethodmodifiers  gdcm::Bitmap::GetPixelFormat " /** PixelFormat&
gdcm::Bitmap::GetPixelFormat()  */ public";

%csmethodmodifiers  gdcm::Bitmap::GetPixelFormat " /** const
PixelFormat& gdcm::Bitmap::GetPixelFormat() const

Get/Set PixelFormat.

*/ public";

%csmethodmodifiers  gdcm::Bitmap::GetPlanarConfiguration " /**
unsigned int gdcm::Bitmap::GetPlanarConfiguration() const

return the planar configuration

*/ public";

%csmethodmodifiers  gdcm::Bitmap::GetRows " /** unsigned int
gdcm::Bitmap::GetRows() const  */ public";

%csmethodmodifiers  gdcm::Bitmap::GetTransferSyntax " /** const
TransferSyntax& gdcm::Bitmap::GetTransferSyntax() const  */ public";

%csmethodmodifiers  gdcm::Bitmap::IsEmpty " /** bool
gdcm::Bitmap::IsEmpty() const  */ public";

%csmethodmodifiers  gdcm::Bitmap::IsLossy " /** bool
gdcm::Bitmap::IsLossy() const

Return whether or not the image was compressed using a lossy
compressor or not Transfer Syntax alone is not sufficient to detect
that. Warning if the image contains an invalid stream, the return code
is also 'false' So this call return true only when the following
combination is true: 1. The image can succefully be read 2. The image
is indeed lossy

*/ public";

%csmethodmodifiers  gdcm::Bitmap::Print " /** void
gdcm::Bitmap::Print(std::ostream &) const  */ public";

%csmethodmodifiers  gdcm::Bitmap::SetColumns " /** void
gdcm::Bitmap::SetColumns(unsigned int col)  */ public";

%csmethodmodifiers  gdcm::Bitmap::SetDataElement " /** void
gdcm::Bitmap::SetDataElement(DataElement const &de)  */ public";

%csmethodmodifiers  gdcm::Bitmap::SetDimension " /** void
gdcm::Bitmap::SetDimension(unsigned int idx, unsigned int dim)  */
public";

%csmethodmodifiers  gdcm::Bitmap::SetDimensions " /** void
gdcm::Bitmap::SetDimensions(const unsigned int dims[3])  */ public";

%csmethodmodifiers  gdcm::Bitmap::SetLossyFlag " /** void
gdcm::Bitmap::SetLossyFlag(bool f)  */ public";

%csmethodmodifiers  gdcm::Bitmap::SetLUT " /** void
gdcm::Bitmap::SetLUT(LookupTable const &lut)

Set/Get LUT.

*/ public";

%csmethodmodifiers  gdcm::Bitmap::SetNeedByteSwap " /** void
gdcm::Bitmap::SetNeedByteSwap(bool b)  */ public";

%csmethodmodifiers  gdcm::Bitmap::SetNumberOfDimensions " /** void
gdcm::Bitmap::SetNumberOfDimensions(unsigned int dim)  */ public";

%csmethodmodifiers  gdcm::Bitmap::SetPhotometricInterpretation " /**
void
gdcm::Bitmap::SetPhotometricInterpretation(PhotometricInterpretation
const &pi)  */ public";

%csmethodmodifiers  gdcm::Bitmap::SetPixelFormat " /** void
gdcm::Bitmap::SetPixelFormat(PixelFormat const &pf)  */ public";

%csmethodmodifiers  gdcm::Bitmap::SetPlanarConfiguration " /** void
gdcm::Bitmap::SetPlanarConfiguration(unsigned int pc)  */ public";

%csmethodmodifiers  gdcm::Bitmap::SetRows " /** void
gdcm::Bitmap::SetRows(unsigned int rows)  */ public";

%csmethodmodifiers  gdcm::Bitmap::SetTransferSyntax " /** void
gdcm::Bitmap::SetTransferSyntax(TransferSyntax const &ts)

Transfer syntax.

*/ public";

%csmethodmodifiers  gdcm::Bitmap::~Bitmap " /**
gdcm::Bitmap::~Bitmap()  */ public";


// File: classstd_1_1bitset.xml
%typemap("csclassmodifiers") std::bitset " /** STL class.

*/ public class";


// File: classgdcm_1_1ByteBuffer.xml
%typemap("csclassmodifiers") gdcm::ByteBuffer " /**  ByteBuffer.

Detailled description here looks like a std::streambuf or std::filebuf
class with the get and peek pointer

C++ includes: gdcmByteBuffer.h */ public class";

%csmethodmodifiers  gdcm::ByteBuffer::ByteBuffer " /**
gdcm::ByteBuffer::ByteBuffer()  */ public";

%csmethodmodifiers  gdcm::ByteBuffer::Get " /** char*
gdcm::ByteBuffer::Get(int len)  */ public";

%csmethodmodifiers  gdcm::ByteBuffer::GetStart " /** const char*
gdcm::ByteBuffer::GetStart() const  */ public";

%csmethodmodifiers  gdcm::ByteBuffer::ShiftEnd " /** void
gdcm::ByteBuffer::ShiftEnd(int len)  */ public";

%csmethodmodifiers  gdcm::ByteBuffer::UpdatePosition " /** void
gdcm::ByteBuffer::UpdatePosition()  */ public";


// File: classgdcm_1_1ByteSwap.xml
%typemap("csclassmodifiers") gdcm::ByteSwap " /**  ByteSwap.

Perform machine dependent byte swaping (Little Endian, Big Endian, Bad
Little Endian, Bad Big Endian). TODO: bswap_32 / bswap_64 ...

C++ includes: gdcmByteSwap.h */ public class";


// File: classgdcm_1_1ByteSwapFilter.xml
%typemap("csclassmodifiers") gdcm::ByteSwapFilter " /**
ByteSwapFilter In place byte-swapping of a dataset FIXME: FL status ??

C++ includes: gdcmByteSwapFilter.h */ public class";

%csmethodmodifiers  gdcm::ByteSwapFilter::ByteSwap " /** bool
gdcm::ByteSwapFilter::ByteSwap()  */ public";

%csmethodmodifiers  gdcm::ByteSwapFilter::ByteSwapFilter " /**
gdcm::ByteSwapFilter::ByteSwapFilter(DataSet &ds)  */ public";

%csmethodmodifiers  gdcm::ByteSwapFilter::SetByteSwapTag " /** void
gdcm::ByteSwapFilter::SetByteSwapTag(bool b)  */ public";

%csmethodmodifiers  gdcm::ByteSwapFilter::~ByteSwapFilter " /**
gdcm::ByteSwapFilter::~ByteSwapFilter()  */ public";


// File: classgdcm_1_1ByteValue.xml
%typemap("csclassmodifiers") gdcm::ByteValue " /** Class to represent
binary value (array of bytes).

C++ includes: gdcmByteValue.h */ public class";

%csmethodmodifiers  gdcm::ByteValue::ByteValue " /**
gdcm::ByteValue::ByteValue(std::vector< char > &v)  */ public";

%csmethodmodifiers  gdcm::ByteValue::ByteValue " /**
gdcm::ByteValue::ByteValue(const char *array=0, VL const &vl=0)  */
public";

%csmethodmodifiers  gdcm::ByteValue::Clear " /** void
gdcm::ByteValue::Clear()  */ public";

%csmethodmodifiers  gdcm::ByteValue::Fill " /** void
gdcm::ByteValue::Fill(char c)  */ public";

%csmethodmodifiers  gdcm::ByteValue::GetBuffer " /** bool
gdcm::ByteValue::GetBuffer(char *buffer, unsigned long length) const
*/ public";

%csmethodmodifiers  gdcm::ByteValue::GetLength " /** VL
gdcm::ByteValue::GetLength() const  */ public";

%csmethodmodifiers  gdcm::ByteValue::GetPointer " /** const char*
gdcm::ByteValue::GetPointer() const  */ public";

%csmethodmodifiers  gdcm::ByteValue::IsEmpty " /** bool
gdcm::ByteValue::IsEmpty() const  */ public";

%csmethodmodifiers  gdcm::ByteValue::IsPrintable " /** bool
gdcm::ByteValue::IsPrintable(VL length) const

Checks whether a 'ByteValue' is printable or not (in order to avoid
corrupting the terminal of invocation when printing) I dont think this
function is working since it does not handle UNICODE or character
set...

*/ public";

%csmethodmodifiers  gdcm::ByteValue::PrintASCII " /** void
gdcm::ByteValue::PrintASCII(std::ostream &os, VL maxlength) const  */
public";

%csmethodmodifiers  gdcm::ByteValue::PrintGroupLength " /** void
gdcm::ByteValue::PrintGroupLength(std::ostream &os)  */ public";

%csmethodmodifiers  gdcm::ByteValue::PrintHex " /** void
gdcm::ByteValue::PrintHex(std::ostream &os, VL maxlength) const  */
public";

%csmethodmodifiers  gdcm::ByteValue::Read " /** std::istream&
gdcm::ByteValue::Read(std::istream &is)  */ public";

%csmethodmodifiers  gdcm::ByteValue::Read " /** std::istream&
gdcm::ByteValue::Read(std::istream &is)  */ public";

%csmethodmodifiers  gdcm::ByteValue::SetLength " /** void
gdcm::ByteValue::SetLength(VL vl)  */ public";

%csmethodmodifiers  gdcm::ByteValue::Write " /** std::ostream const&
gdcm::ByteValue::Write(std::ostream &os) const  */ public";

%csmethodmodifiers  gdcm::ByteValue::Write " /** std::ostream const&
gdcm::ByteValue::Write(std::ostream &os) const  */ public";

%csmethodmodifiers  gdcm::ByteValue::WriteBuffer " /** bool
gdcm::ByteValue::WriteBuffer(std::ostream &os) const  */ public";

%csmethodmodifiers  gdcm::ByteValue::~ByteValue " /**
gdcm::ByteValue::~ByteValue()  */ public";


// File: classgdcm_1_1Codec.xml
%typemap("csclassmodifiers") gdcm::Codec " /**  Codec class.

C++ includes: gdcmCodec.h */ public class";


// File: classgdcm_1_1Coder.xml
%typemap("csclassmodifiers") gdcm::Coder " /**  Coder.

C++ includes: gdcmCoder.h */ public class";

%csmethodmodifiers  gdcm::Coder::CanCode " /** virtual bool
gdcm::Coder::CanCode(TransferSyntax const &) const =0

Return whether this coder support this transfer syntax (can code it).

*/ public";

%csmethodmodifiers  gdcm::Coder::Code " /** virtual bool
gdcm::Coder::Code(DataElement const &in, DataElement &out)

Code.

*/ public";

%csmethodmodifiers  gdcm::Coder::~Coder " /** virtual
gdcm::Coder::~Coder()  */ public";


// File: classstd_1_1complex.xml
%typemap("csclassmodifiers") std::complex " /** STL class.

*/ public class";


// File: classgdcm_1_1ConstCharWrapper.xml
%typemap("csclassmodifiers") gdcm::ConstCharWrapper " /**C++ includes:
gdcmConstCharWrapper.h */ public class";

%csmethodmodifiers  gdcm::ConstCharWrapper::ConstCharWrapper " /**
gdcm::ConstCharWrapper::ConstCharWrapper(const char *i=0)  */ public";


// File: classgdcm_1_1CP246ExplicitDataElement.xml
%typemap("csclassmodifiers") gdcm::CP246ExplicitDataElement " /**
Class to read/write a DataElement as CP246Explicit Data Element.

Some system are producing SQ, declare them as UN, but encode the SQ as
'Explicit' instead of Implicit

C++ includes: gdcmCP246ExplicitDataElement.h */ public class";

%csmethodmodifiers  gdcm::CP246ExplicitDataElement::GetLength " /** VL
gdcm::CP246ExplicitDataElement::GetLength() const  */ public";

%csmethodmodifiers  gdcm::CP246ExplicitDataElement::Read " /**
std::istream& gdcm::CP246ExplicitDataElement::Read(std::istream &is)
*/ public";

%csmethodmodifiers  gdcm::CP246ExplicitDataElement::ReadWithLength "
/** std::istream&
gdcm::CP246ExplicitDataElement::ReadWithLength(std::istream &is, VL
&length)  */ public";


// File: classgdcm_1_1CSAElement.xml
%typemap("csclassmodifiers") gdcm::CSAElement " /** Class to represent
a CSA Element.

C++ includes: gdcmCSAElement.h */ public class";

%csmethodmodifiers  gdcm::CSAElement::CSAElement " /**
gdcm::CSAElement::CSAElement(const CSAElement &_val)  */ public";

%csmethodmodifiers  gdcm::CSAElement::CSAElement " /**
gdcm::CSAElement::CSAElement(unsigned int kf=0)  */ public";

%csmethodmodifiers  gdcm::CSAElement::GetByteValue " /** const
ByteValue* gdcm::CSAElement::GetByteValue() const  */ public";

%csmethodmodifiers  gdcm::CSAElement::GetKey " /** unsigned int
gdcm::CSAElement::GetKey() const

Set/Get Key.

*/ public";

%csmethodmodifiers  gdcm::CSAElement::GetName " /** const char*
gdcm::CSAElement::GetName() const

Set/Get Name.

*/ public";

%csmethodmodifiers  gdcm::CSAElement::GetNoOfItems " /** unsigned int
gdcm::CSAElement::GetNoOfItems() const

Set/Get NoOfItems.

*/ public";

%csmethodmodifiers  gdcm::CSAElement::GetSyngoDT " /** unsigned int
gdcm::CSAElement::GetSyngoDT() const

Set/Get SyngoDT.

*/ public";

%csmethodmodifiers  gdcm::CSAElement::GetValue " /** Value&
gdcm::CSAElement::GetValue()  */ public";

%csmethodmodifiers  gdcm::CSAElement::GetValue " /** Value const&
gdcm::CSAElement::GetValue() const

Set/Get Value (bytes array, SQ of items, SQ of fragments):.

*/ public";

%csmethodmodifiers  gdcm::CSAElement::GetVM " /** const VM&
gdcm::CSAElement::GetVM() const

Set/Get VM.

*/ public";

%csmethodmodifiers  gdcm::CSAElement::GetVR " /** VR const&
gdcm::CSAElement::GetVR() const

Set/Get VR.

*/ public";

%csmethodmodifiers  gdcm::CSAElement::IsEmpty " /** bool
gdcm::CSAElement::IsEmpty() const  */ public";

%csmethodmodifiers  gdcm::CSAElement::SetByteValue " /** void
gdcm::CSAElement::SetByteValue(const char *array, VL length)  */
public";

%csmethodmodifiers  gdcm::CSAElement::SetKey " /** void
gdcm::CSAElement::SetKey(unsigned int key)  */ public";

%csmethodmodifiers  gdcm::CSAElement::SetName " /** void
gdcm::CSAElement::SetName(const char *name)  */ public";

%csmethodmodifiers  gdcm::CSAElement::SetNoOfItems " /** void
gdcm::CSAElement::SetNoOfItems(unsigned int items)  */ public";

%csmethodmodifiers  gdcm::CSAElement::SetSyngoDT " /** void
gdcm::CSAElement::SetSyngoDT(unsigned int syngodt)  */ public";

%csmethodmodifiers  gdcm::CSAElement::SetValue " /** void
gdcm::CSAElement::SetValue(Value const &vl)  */ public";

%csmethodmodifiers  gdcm::CSAElement::SetVM " /** void
gdcm::CSAElement::SetVM(const VM &vm)  */ public";

%csmethodmodifiers  gdcm::CSAElement::SetVR " /** void
gdcm::CSAElement::SetVR(VR const &vr)  */ public";


// File: classgdcm_1_1CSAHeader.xml
%typemap("csclassmodifiers") gdcm::CSAHeader " /** Class for
CSAHeader.

SIEMENS store private information in tag (0x0029,0x10,\"SIEMENS CSA
HEADER\") this class is meant for user wishing to access values stored
within this private attribute. There are basically two main 'format'
for this attribute : SV10/NOMAGIC and DATASET_FORMAT SV10 and NOMAGIC
are from a user prospective identical, see CSAHeader.xml for possible
name / value stored in this format. DATASET_FORMAT is in fact simply
just another DICOM dataset (implicit) with -currently unknown- value.
This can be only be printer for now.

WARNING:  : Everything you do with this code is at your own risk,
since decoding process was not written from specification documents.
: the API of this class might change.

: MrEvaProtocol in 29,1020 contains ^M that would be nice to get rid
of on UNIX system...

also 5.1.3.2.4.1 MEDCOM History Information and 5.1.4.3 CSA Non-Image
Module inhttp://tamsinfo.toshiba.com/docrequest/pdf/E.Soft_v2.0.pdf

C++ includes: gdcmCSAHeader.h */ public class";

%csmethodmodifiers  gdcm::CSAHeader::CSAHeader " /**
gdcm::CSAHeader::CSAHeader()  */ public";

%csmethodmodifiers  gdcm::CSAHeader::FindCSAElementByName " /** bool
gdcm::CSAHeader::FindCSAElementByName(const char *name)

Return true if the CSA element matching 'name' is found or not
WARNING:  Case Sensitive

*/ public";

%csmethodmodifiers  gdcm::CSAHeader::GetCSAElementByName " /** const
CSAElement& gdcm::CSAHeader::GetCSAElementByName(const char *name)

Return the CSAElement corresponding to name 'name' WARNING:  Case
Sensitive

*/ public";

%csmethodmodifiers  gdcm::CSAHeader::GetDataSet " /** const DataSet&
gdcm::CSAHeader::GetDataSet() const

Return the DataSet output (use only if Format == DATASET_FORMAT ).

*/ public";

%csmethodmodifiers  gdcm::CSAHeader::GetFormat " /** CSAHeaderType
gdcm::CSAHeader::GetFormat() const

return the format of the CSAHeader SV10 and NOMAGIC are equivalent.

*/ public";

%csmethodmodifiers  gdcm::CSAHeader::GetInterfile " /** const char*
gdcm::CSAHeader::GetInterfile() const

Return the string output (use only if Format == Interfile).

*/ public";

%csmethodmodifiers  gdcm::CSAHeader::LoadFromDataElement " /** bool
gdcm::CSAHeader::LoadFromDataElement(DataElement const &de)

Decode the CSAHeader from element 'de'.

*/ public";

%csmethodmodifiers  gdcm::CSAHeader::Print " /** void
gdcm::CSAHeader::Print(std::ostream &os) const

Print the CSAHeader (use only if Format == SV10 or NOMAGIC).

*/ public";

%csmethodmodifiers  gdcm::CSAHeader::Read " /** std::istream&
gdcm::CSAHeader::Read(std::istream &is)  */ public";

%csmethodmodifiers  gdcm::CSAHeader::Write " /** const std::ostream&
gdcm::CSAHeader::Write(std::ostream &os) const  */ public";

%csmethodmodifiers  gdcm::CSAHeader::~CSAHeader " /**
gdcm::CSAHeader::~CSAHeader()  */ public";


// File: classgdcm_1_1CSAHeaderDict.xml
%typemap("csclassmodifiers") gdcm::CSAHeaderDict " /** Class to
represent a map of CSAHeaderDictEntry.

C++ includes: gdcmCSAHeaderDict.h */ public class";

%csmethodmodifiers  gdcm::CSAHeaderDict::AddCSAHeaderDictEntry " /**
void gdcm::CSAHeaderDict::AddCSAHeaderDictEntry(const
CSAHeaderDictEntry &de)  */ public";

%csmethodmodifiers  gdcm::CSAHeaderDict::Begin " /** ConstIterator
gdcm::CSAHeaderDict::Begin() const  */ public";

%csmethodmodifiers  gdcm::CSAHeaderDict::CSAHeaderDict " /**
gdcm::CSAHeaderDict::CSAHeaderDict()  */ public";

%csmethodmodifiers  gdcm::CSAHeaderDict::End " /** ConstIterator
gdcm::CSAHeaderDict::End() const  */ public";

%csmethodmodifiers  gdcm::CSAHeaderDict::GetCSAHeaderDictEntry " /**
const CSAHeaderDictEntry&
gdcm::CSAHeaderDict::GetCSAHeaderDictEntry(const char *name) const  */
public";

%csmethodmodifiers  gdcm::CSAHeaderDict::IsEmpty " /** bool
gdcm::CSAHeaderDict::IsEmpty() const  */ public";


// File: classgdcm_1_1CSAHeaderDictEntry.xml
%typemap("csclassmodifiers") gdcm::CSAHeaderDictEntry " /** Class to
represent an Entry in the Dict Does not really exist within the DICOM
definition, just a way to minimize storage and have a mapping from
gdcm::Tag to the needed information.

bla TODO FIXME: Need a PublicCSAHeaderDictEntry...indeed
CSAHeaderDictEntry has a notion of retired which does not exist in
PrivateCSAHeaderDictEntry...

See:   gdcm::Dict

C++ includes: gdcmCSAHeaderDictEntry.h */ public class";

%csmethodmodifiers  gdcm::CSAHeaderDictEntry::CSAHeaderDictEntry " /**
gdcm::CSAHeaderDictEntry::CSAHeaderDictEntry(const char *name=\"\", VR
const &vr=VR::INVALID, VM const &vm=VM::VM0, const char *desc=\"\")
*/ public";

%csmethodmodifiers  gdcm::CSAHeaderDictEntry::GetDescription " /**
const char* gdcm::CSAHeaderDictEntry::GetDescription() const

Set/Get Description.

*/ public";

%csmethodmodifiers  gdcm::CSAHeaderDictEntry::GetName " /** const
char* gdcm::CSAHeaderDictEntry::GetName() const

Set/Get Name.

*/ public";

%csmethodmodifiers  gdcm::CSAHeaderDictEntry::GetVM " /** const VM&
gdcm::CSAHeaderDictEntry::GetVM() const

Set/Get VM.

*/ public";

%csmethodmodifiers  gdcm::CSAHeaderDictEntry::GetVR " /** const VR&
gdcm::CSAHeaderDictEntry::GetVR() const

Set/Get VR.

*/ public";

%csmethodmodifiers  gdcm::CSAHeaderDictEntry::SetDescription " /**
void gdcm::CSAHeaderDictEntry::SetDescription(const char *desc)  */
public";

%csmethodmodifiers  gdcm::CSAHeaderDictEntry::SetName " /** void
gdcm::CSAHeaderDictEntry::SetName(const char *name)  */ public";

%csmethodmodifiers  gdcm::CSAHeaderDictEntry::SetVM " /** void
gdcm::CSAHeaderDictEntry::SetVM(VM const &vm)  */ public";

%csmethodmodifiers  gdcm::CSAHeaderDictEntry::SetVR " /** void
gdcm::CSAHeaderDictEntry::SetVR(const VR &vr)  */ public";


// File: classgdcm_1_1CSAHeaderDictException.xml
%typemap("csclassmodifiers") gdcm::CSAHeaderDictException " /**C++
includes: gdcmCSAHeaderDict.h */ public class";


// File: classgdcm_1_1Curve.xml
%typemap("csclassmodifiers") gdcm::Curve " /**  Curve class to handle
element 50xx,3000 Curve Data WARNING: This is deprecated and lastly
defined in PS 3.3 - 2004.

Examples: GE_DLX-8-MONO2-Multiframe-Jpeg_Lossless.dcm

GE_DLX-8-MONO2-Multiframe.dcm

gdcmSampleData/Philips_Medical_Images/integris_HV_5000/xa_integris.dcm

TOSHIBA-CurveData[1-3].dcm

C++ includes: gdcmCurve.h */ public class";

%csmethodmodifiers  gdcm::Curve::Curve " /** gdcm::Curve::Curve(Curve
const &ov)  */ public";

%csmethodmodifiers  gdcm::Curve::Curve " /** gdcm::Curve::Curve()  */
public";

%csmethodmodifiers  gdcm::Curve::Decode " /** void
gdcm::Curve::Decode(std::istream &is, std::ostream &os)  */ public";

%csmethodmodifiers  gdcm::Curve::GetAsPoints " /** void
gdcm::Curve::GetAsPoints(float *array) const  */ public";

%csmethodmodifiers  gdcm::Curve::GetDataValueRepresentation " /**
unsigned short gdcm::Curve::GetDataValueRepresentation() const  */
public";

%csmethodmodifiers  gdcm::Curve::GetDimensions " /** unsigned short
gdcm::Curve::GetDimensions() const  */ public";

%csmethodmodifiers  gdcm::Curve::GetGroup " /** unsigned short
gdcm::Curve::GetGroup() const  */ public";

%csmethodmodifiers  gdcm::Curve::GetNumberOfPoints " /** unsigned
short gdcm::Curve::GetNumberOfPoints() const  */ public";

%csmethodmodifiers  gdcm::Curve::GetTypeOfData " /** const char*
gdcm::Curve::GetTypeOfData() const  */ public";

%csmethodmodifiers  gdcm::Curve::GetTypeOfDataDescription " /** const
char* gdcm::Curve::GetTypeOfDataDescription() const  */ public";

%csmethodmodifiers  gdcm::Curve::IsEmpty " /** bool
gdcm::Curve::IsEmpty() const  */ public";

%csmethodmodifiers  gdcm::Curve::Print " /** void
gdcm::Curve::Print(std::ostream &) const  */ public";

%csmethodmodifiers  gdcm::Curve::SetCurve " /** void
gdcm::Curve::SetCurve(const char *array, unsigned int length)  */
public";

%csmethodmodifiers  gdcm::Curve::SetCurveDescription " /** void
gdcm::Curve::SetCurveDescription(const char *curvedescription)  */
public";

%csmethodmodifiers  gdcm::Curve::SetDataValueRepresentation " /** void
gdcm::Curve::SetDataValueRepresentation(unsigned short
datavaluerepresentation)  */ public";

%csmethodmodifiers  gdcm::Curve::SetDimensions " /** void
gdcm::Curve::SetDimensions(unsigned short dimensions)  */ public";

%csmethodmodifiers  gdcm::Curve::SetGroup " /** void
gdcm::Curve::SetGroup(unsigned short group)  */ public";

%csmethodmodifiers  gdcm::Curve::SetNumberOfPoints " /** void
gdcm::Curve::SetNumberOfPoints(unsigned short numberofpoints)  */
public";

%csmethodmodifiers  gdcm::Curve::SetTypeOfData " /** void
gdcm::Curve::SetTypeOfData(const char *typeofdata)  */ public";

%csmethodmodifiers  gdcm::Curve::Update " /** void
gdcm::Curve::Update(const DataElement &de)  */ public";

%csmethodmodifiers  gdcm::Curve::~Curve " /** gdcm::Curve::~Curve()
*/ public";


// File: classgdcm_1_1DataElement.xml
%typemap("csclassmodifiers") gdcm::DataElement " /** Class to
represent a Data Element either Implicit or Explicit.

DATA ELEMENT: A unit of information as defined by a single entry in
the data dictionary. An encoded Information Object Definition ( IOD)
Attribute that is composed of, at a minimum, three fields: a Data
Element Tag, a Value Length, and a Value Field. For some specific
Transfer Syntaxes, a Data Element also contains a VR Field where the
Value Representation of that Data Element is specified explicitly.

C++ includes: gdcmDataElement.h */ public class";

%csmethodmodifiers  gdcm::DataElement::Clear " /** void
gdcm::DataElement::Clear()  */ public";

%csmethodmodifiers  gdcm::DataElement::DataElement " /**
gdcm::DataElement::DataElement(const DataElement &_val)  */ public";

%csmethodmodifiers  gdcm::DataElement::DataElement " /**
gdcm::DataElement::DataElement(const Tag &t=Tag(0), const VL &vl=0,
const VR &vr=VR::INVALID)  */ public";

%csmethodmodifiers  gdcm::DataElement::Empty " /** void
gdcm::DataElement::Empty()  */ public";

%csmethodmodifiers  gdcm::DataElement::GetByteValue " /** ByteValue*
gdcm::DataElement::GetByteValue()  */ public";

%csmethodmodifiers  gdcm::DataElement::GetByteValue " /** const
ByteValue* gdcm::DataElement::GetByteValue() const

Return the Value of DataElement as a ByteValue (if possible) WARNING:
: You need to check for NULL return value

*/ public";

%csmethodmodifiers  gdcm::DataElement::GetLength " /** VL
gdcm::DataElement::GetLength() const  */ public";

%csmethodmodifiers  gdcm::DataElement::GetSequenceOfFragments " /**
const SequenceOfFragments* gdcm::DataElement::GetSequenceOfFragments()
const

Return the Value of DataElement as a Sequence Of Fragments (if
possible) WARNING:  : You need to check for NULL return value

*/ public";

%csmethodmodifiers  gdcm::DataElement::GetSequenceOfItems " /**
SequenceOfItems* gdcm::DataElement::GetSequenceOfItems()  */ public";

%csmethodmodifiers  gdcm::DataElement::GetSequenceOfItems " /** const
SequenceOfItems* gdcm::DataElement::GetSequenceOfItems() const

Return the Value of DataElement as a Sequence Of Items (if possible)
WARNING:  : You need to check for NULL return value

: In some case a Value could not have been recognized as a
SequenceOfItems in those case the return of the function will be NULL,
while the Value would be a valid SequenceOfItems, in those case prefer
GetValueAsSQ. In which case the code internally trigger an assert to
warn developper.

*/ public";

%csmethodmodifiers  gdcm::DataElement::GetTag " /** Tag&
gdcm::DataElement::GetTag()  */ public";

%csmethodmodifiers  gdcm::DataElement::GetTag " /** const Tag&
gdcm::DataElement::GetTag() const

Get Tag.

*/ public";

%csmethodmodifiers  gdcm::DataElement::GetValue " /** Value&
gdcm::DataElement::GetValue()  */ public";

%csmethodmodifiers  gdcm::DataElement::GetValue " /** Value const&
gdcm::DataElement::GetValue() const

Set/Get Value (bytes array, SQ of items, SQ of fragments):.

*/ public";

%csmethodmodifiers  gdcm::DataElement::GetValueAsSQ " /**
SmartPointer<SequenceOfItems> gdcm::DataElement::GetValueAsSQ() const

Interpret the Value stored in the DataElement. This is more robust
(but also more expensive) to call this function rather than the
simpliest form: GetSequenceOfItems() It also return NULL when the
Value is NOT of type SequenceOfItems WARNING:  in case
GetSequenceOfItems() succeed the function return this value, otherwise
it creates a new SequenceOfItems, you should handle that in your case,
for instance: SmartPointer<SequenceOfItems> sqi = de.GetValueAsSQ();

*/ public";

%csmethodmodifiers  gdcm::DataElement::GetVL " /** const VL&
gdcm::DataElement::GetVL() const

Get VL.

*/ public";

%csmethodmodifiers  gdcm::DataElement::GetVR " /** VR const&
gdcm::DataElement::GetVR() const

Get VR do not set VR::SQ on bytevalue data element

*/ public";

%csmethodmodifiers  gdcm::DataElement::IsEmpty " /** bool
gdcm::DataElement::IsEmpty() const

Check if Data Element is empty.

*/ public";

%csmethodmodifiers  gdcm::DataElement::IsUndefinedLength " /** bool
gdcm::DataElement::IsUndefinedLength() const

return if Value Length if of undefined length

*/ public";

%csmethodmodifiers  gdcm::DataElement::Read " /** std::istream&
gdcm::DataElement::Read(std::istream &is)  */ public";

%csmethodmodifiers  gdcm::DataElement::ReadOrSkip " /** std::istream&
gdcm::DataElement::ReadOrSkip(std::istream &is, std::set< Tag > const
&skiptags)  */ public";

%csmethodmodifiers  gdcm::DataElement::ReadWithLength " /**
std::istream& gdcm::DataElement::ReadWithLength(std::istream &is, VL
&length)  */ public";

%csmethodmodifiers  gdcm::DataElement::SetByteValue " /** void
gdcm::DataElement::SetByteValue(const char *array, VL length)

Set the byte value WARNING:  user need to read DICOM standard for an
understanding of: even padding vs space padding By default even
padding is achieved using  regardless of the of VR

*/ public";

%csmethodmodifiers  gdcm::DataElement::SetTag " /** void
gdcm::DataElement::SetTag(const Tag &t)

Set Tag Use with cautious (need to match Part 6)

*/ public";

%csmethodmodifiers  gdcm::DataElement::SetValue " /** void
gdcm::DataElement::SetValue(Value const &vl)

WARNING:  you need to set the ValueLengthField explicitely

*/ public";

%csmethodmodifiers  gdcm::DataElement::SetVL " /** void
gdcm::DataElement::SetVL(const VL &vl)

Set VL Use with cautious (need to match Part 6), advanced user only
See:   SetByteValue

*/ public";

%csmethodmodifiers  gdcm::DataElement::SetVLToUndefined " /** void
gdcm::DataElement::SetVLToUndefined()  */ public";

%csmethodmodifiers  gdcm::DataElement::SetVR " /** void
gdcm::DataElement::SetVR(VR const &vr)

Set VR Use with cautious (need to match Part 6), advanced user only vr
is a VR::VRALL (not a dual one such as OB_OW)

*/ public";

%csmethodmodifiers  gdcm::DataElement::Write " /** const std::ostream&
gdcm::DataElement::Write(std::ostream &os) const  */ public";


// File: classgdcm_1_1DataElementException.xml
%typemap("csclassmodifiers") gdcm::DataElementException " /**C++
includes: gdcmDataSet.h */ public class";


// File: classgdcm_1_1DataSet.xml
%typemap("csclassmodifiers") gdcm::DataSet " /** Class to represent a
Data Set (which contains Data Elements) A Data Set represents an
instance of a real world Information Object.

DATA SET: Exchanged information consisting of a structured set of
Attribute values directly or indirectly related to Information
Objects. The value of each Attribute in a Data Set is expressed as a
Data Element. A collection of Data Elements ordered by increasing Data
Element Tag number that is an encoding of the values of Attributes of
a real world object.

Implementation note. If one do: DataSet ds; ds.SetLength(0);
ds.Read(is); setting length to 0 actually means try to read is as if
it was a root DataSet. Other value are undefined (nested dataset with
undefined length) or defined length (different from 0) means nested
dataset with defined length.  TODO: a DataSet DOES NOT have a TS
type... a file does !

C++ includes: gdcmDataSet.h */ public class";

%csmethodmodifiers  gdcm::DataSet::Begin " /** Iterator
gdcm::DataSet::Begin()  */ public";

%csmethodmodifiers  gdcm::DataSet::Begin " /** ConstIterator
gdcm::DataSet::Begin() const  */ public";

%csmethodmodifiers  gdcm::DataSet::Clear " /** void
gdcm::DataSet::Clear()  */ public";

%csmethodmodifiers  gdcm::DataSet::ComputeGroupLength " /** unsigned
int gdcm::DataSet::ComputeGroupLength(Tag const &tag) const  */
public";

%csmethodmodifiers  gdcm::DataSet::End " /** Iterator
gdcm::DataSet::End()  */ public";

%csmethodmodifiers  gdcm::DataSet::End " /** ConstIterator
gdcm::DataSet::End() const  */ public";

%csmethodmodifiers  gdcm::DataSet::FindDataElement " /** bool
gdcm::DataSet::FindDataElement(const Tag &t) const  */ public";

%csmethodmodifiers  gdcm::DataSet::FindDataElement " /** bool
gdcm::DataSet::FindDataElement(const PrivateTag &t) const

Look up if private tag 't' is present in the dataset:.

*/ public";

%csmethodmodifiers  gdcm::DataSet::FindNextDataElement " /** const
DataElement& gdcm::DataSet::FindNextDataElement(const Tag &t) const
*/ public";

%csmethodmodifiers  gdcm::DataSet::GetDataElement " /** const
DataElement& gdcm::DataSet::GetDataElement(const PrivateTag &t) const

Return the dataelement.

*/ public";

%csmethodmodifiers  gdcm::DataSet::GetDataElement " /** const
DataElement& gdcm::DataSet::GetDataElement(const Tag &t) const

Return the DataElement with Tag 't' WARNING:  : This only search at
the 'root level' of the DataSet

*/ public";

%csmethodmodifiers  gdcm::DataSet::GetDES " /** DataElementSet&
gdcm::DataSet::GetDES()  */ public";

%csmethodmodifiers  gdcm::DataSet::GetDES " /** const DataElementSet&
gdcm::DataSet::GetDES() const  */ public";

%csmethodmodifiers  gdcm::DataSet::GetLength " /** VL
gdcm::DataSet::GetLength() const  */ public";

%csmethodmodifiers  gdcm::DataSet::GetPrivateCreator " /** std::string
gdcm::DataSet::GetPrivateCreator(const Tag &t) const

Return the private creator of the private tag 't':.

*/ public";

%csmethodmodifiers  gdcm::DataSet::Insert " /** void
gdcm::DataSet::Insert(const DataElement &de)

Insert a DataElement in the DataSet. WARNING:  : Tag need to be >= 0x8
to be considered valid data element

*/ public";

%csmethodmodifiers  gdcm::DataSet::IsEmpty " /** bool
gdcm::DataSet::IsEmpty() const

Returns if the dataset is empty.

*/ public";

%csmethodmodifiers  gdcm::DataSet::Print " /** void
gdcm::DataSet::Print(std::ostream &os, std::string const &indent=\"\")
const  */ public";

%csmethodmodifiers  gdcm::DataSet::Read " /** std::istream&
gdcm::DataSet::Read(std::istream &is)  */ public";

%csmethodmodifiers  gdcm::DataSet::ReadNested " /** std::istream&
gdcm::DataSet::ReadNested(std::istream &is)  */ public";

%csmethodmodifiers  gdcm::DataSet::ReadUpToTag " /** std::istream&
gdcm::DataSet::ReadUpToTag(std::istream &is, const Tag &t, std::set<
Tag > const &skiptags)  */ public";

%csmethodmodifiers  gdcm::DataSet::ReadUpToTagWithLength " /**
std::istream& gdcm::DataSet::ReadUpToTagWithLength(std::istream &is,
const Tag &t, VL &length)  */ public";

%csmethodmodifiers  gdcm::DataSet::ReadWithLength " /** std::istream&
gdcm::DataSet::ReadWithLength(std::istream &is, VL &length)  */
public";

%csmethodmodifiers  gdcm::DataSet::Remove " /** SizeType
gdcm::DataSet::Remove(const Tag &tag)

Completely remove a dataelement from the dataset.

*/ public";

%csmethodmodifiers  gdcm::DataSet::Replace " /** void
gdcm::DataSet::Replace(const DataElement &de)

Replace a dataelement with another one.

*/ public";

%csmethodmodifiers  gdcm::DataSet::Size " /** unsigned int
gdcm::DataSet::Size() const  */ public";

%csmethodmodifiers  gdcm::DataSet::Write " /** std::ostream const&
gdcm::DataSet::Write(std::ostream &os) const  */ public";


// File: classgdcm_1_1DataSetHelper.xml
%typemap("csclassmodifiers") gdcm::DataSetHelper " /**  DataSetHelper
(internal class, not intended for user level).

C++ includes: gdcmDataSetHelper.h */ public class";


// File: classgdcm_1_1Decoder.xml
%typemap("csclassmodifiers") gdcm::Decoder " /**  Decoder.

C++ includes: gdcmDecoder.h */ public class";

%csmethodmodifiers  gdcm::Decoder::CanDecode " /** virtual bool
gdcm::Decoder::CanDecode(TransferSyntax const &) const =0

Return whether this decoder support this transfer syntax (can decode
it).

*/ public";

%csmethodmodifiers  gdcm::Decoder::Decode " /** virtual bool
gdcm::Decoder::Decode(DataElement const &is, DataElement &os)

Decode.

*/ public";

%csmethodmodifiers  gdcm::Decoder::~Decoder " /** virtual
gdcm::Decoder::~Decoder()  */ public";


// File: classgdcm_1_1DefinedTerms.xml
%typemap("csclassmodifiers") gdcm::DefinedTerms " /** Defined Terms
are used when the specified explicit Values may be extended by
implementors to include additional new Values. These new Values shall
be specified in the Conformance Statement (see PS 3.2) and shall not
have the same meaning as currently defined Values in this standard. A
Data Element with Defined Terms that does not contain a Value
equivalent to one of the Values currently specified in this standard
shall not be considered to have an invalid value. Note: Interpretation
Type ID (4008,0210) is an example of a Data Element having Defined
Terms. It is defined to have a Value that may be one of the set of
standard Values; REPORT or AMENDMENT (see PS 3.3). Because this Data
Element has Defined Terms other Interpretation Type IDs may be defined
by the implementor.

C++ includes: gdcmDefinedTerms.h */ public class";

%csmethodmodifiers  gdcm::DefinedTerms::DefinedTerms " /**
gdcm::DefinedTerms::DefinedTerms()  */ public";


// File: classgdcm_1_1Defs.xml
%typemap("csclassmodifiers") gdcm::Defs " /** FIXME I do not like the
name 'Defs'.

bla

C++ includes: gdcmDefs.h */ public class";

%csmethodmodifiers  gdcm::Defs::Defs " /** gdcm::Defs::Defs()  */
public";

%csmethodmodifiers  gdcm::Defs::GetIODs " /** IODs&
gdcm::Defs::GetIODs()  */ public";

%csmethodmodifiers  gdcm::Defs::GetIODs " /** const IODs&
gdcm::Defs::GetIODs() const  */ public";

%csmethodmodifiers  gdcm::Defs::GetMacros " /** Macros&
gdcm::Defs::GetMacros()  */ public";

%csmethodmodifiers  gdcm::Defs::GetMacros " /** const Macros&
gdcm::Defs::GetMacros() const  */ public";

%csmethodmodifiers  gdcm::Defs::GetModules " /** Modules&
gdcm::Defs::GetModules()  */ public";

%csmethodmodifiers  gdcm::Defs::GetModules " /** const Modules&
gdcm::Defs::GetModules() const  */ public";

%csmethodmodifiers  gdcm::Defs::GetTypeFromTag " /** Type
gdcm::Defs::GetTypeFromTag(const File &file, const Tag &tag) const  */
public";

%csmethodmodifiers  gdcm::Defs::IsEmpty " /** bool
gdcm::Defs::IsEmpty() const  */ public";

%csmethodmodifiers  gdcm::Defs::Verify " /** bool
gdcm::Defs::Verify(const DataSet &ds) const  */ public";

%csmethodmodifiers  gdcm::Defs::Verify " /** bool
gdcm::Defs::Verify(const File &file) const  */ public";

%csmethodmodifiers  gdcm::Defs::~Defs " /** gdcm::Defs::~Defs()  */
public";


// File: classgdcm_1_1DeltaEncodingCodec.xml
%typemap("csclassmodifiers") gdcm::DeltaEncodingCodec " /**
DeltaEncodingCodec compression used by some private vendor.

C++ includes: gdcmDeltaEncodingCodec.h */ public class";

%csmethodmodifiers  gdcm::DeltaEncodingCodec::CanDecode " /** bool
gdcm::DeltaEncodingCodec::CanDecode(TransferSyntax const &ts)  */
public";

%csmethodmodifiers  gdcm::DeltaEncodingCodec::Decode " /** bool
gdcm::DeltaEncodingCodec::Decode(DataElement const &is, DataElement
&os)

Decode.

*/ public";

%csmethodmodifiers  gdcm::DeltaEncodingCodec::DeltaEncodingCodec " /**
gdcm::DeltaEncodingCodec::DeltaEncodingCodec()  */ public";

%csmethodmodifiers  gdcm::DeltaEncodingCodec::~DeltaEncodingCodec "
/** gdcm::DeltaEncodingCodec::~DeltaEncodingCodec()  */ public";


// File: classstd_1_1deque.xml
%typemap("csclassmodifiers") std::deque " /** STL class.

*/ public class";


// File: classstd_1_1deque_1_1const__iterator.xml
%typemap("csclassmodifiers") std::deque::const_iterator " /** STL
iterator class.

*/ public class";


// File: classstd_1_1deque_1_1const__reverse__iterator.xml
%typemap("csclassmodifiers") std::deque::const_reverse_iterator " /**
STL iterator class.

*/ public class";


// File: classstd_1_1deque_1_1iterator.xml
%typemap("csclassmodifiers") std::deque::iterator " /** STL iterator
class.

*/ public class";


// File: classstd_1_1deque_1_1reverse__iterator.xml
%typemap("csclassmodifiers") std::deque::reverse_iterator " /** STL
iterator class.

*/ public class";


// File: classgdcm_1_1DICOMDIR.xml
%typemap("csclassmodifiers") gdcm::DICOMDIR " /**  DICOMDIR.

C++ includes: gdcmDICOMDIR.h */ public class";

%csmethodmodifiers  gdcm::DICOMDIR::DICOMDIR " /**
gdcm::DICOMDIR::DICOMDIR(const FileSet &fs)  */ public";

%csmethodmodifiers  gdcm::DICOMDIR::DICOMDIR " /**
gdcm::DICOMDIR::DICOMDIR()  */ public";


// File: classgdcm_1_1Dict.xml
%typemap("csclassmodifiers") gdcm::Dict " /** Class to represent a map
of DictEntry.

bla TODO FIXME: For Element == 0x0 need to return Name = Group Length
ValueRepresentation = UL ValueMultiplicity = 1

C++ includes: gdcmDict.h */ public class";

%csmethodmodifiers  gdcm::Dict::AddDictEntry " /** void
gdcm::Dict::AddDictEntry(const Tag &tag, const DictEntry &de)  */
public";

%csmethodmodifiers  gdcm::Dict::Begin " /** ConstIterator
gdcm::Dict::Begin() const  */ public";

%csmethodmodifiers  gdcm::Dict::Dict " /** gdcm::Dict::Dict()  */
public";

%csmethodmodifiers  gdcm::Dict::End " /** ConstIterator
gdcm::Dict::End() const  */ public";

%csmethodmodifiers  gdcm::Dict::GetDictEntry " /** const DictEntry&
gdcm::Dict::GetDictEntry(const Tag &tag) const  */ public";

%csmethodmodifiers  gdcm::Dict::GetDictEntryByName " /** const
DictEntry& gdcm::Dict::GetDictEntryByName(const char *name, Tag &tag)
const

Inefficient way of looking up tag by name. Technically DICOM does not
garantee uniqueness (and Curve / Overlay are there to prove it). But
most of the time name is in fact uniq and can be uniquely link to a
tag

*/ public";

%csmethodmodifiers  gdcm::Dict::IsEmpty " /** bool
gdcm::Dict::IsEmpty() const  */ public";


// File: classgdcm_1_1DictConverter.xml
%typemap("csclassmodifiers") gdcm::DictConverter " /** Class to
convert a .dic file into something else: CXX code : embeded dict into
shared lib (DICT_DEFAULT)

Debug mode (DICT_DEBUG)

XML dict (DICT_XML).

C++ includes: gdcmDictConverter.h */ public class";

%csmethodmodifiers  gdcm::DictConverter::Convert " /** void
gdcm::DictConverter::Convert()  */ public";

%csmethodmodifiers  gdcm::DictConverter::DictConverter " /**
gdcm::DictConverter::DictConverter()  */ public";

%csmethodmodifiers  gdcm::DictConverter::GetDictName " /** const
std::string& gdcm::DictConverter::GetDictName() const  */ public";

%csmethodmodifiers  gdcm::DictConverter::GetInputFilename " /** const
std::string& gdcm::DictConverter::GetInputFilename() const  */
public";

%csmethodmodifiers  gdcm::DictConverter::GetOutputFilename " /** const
std::string& gdcm::DictConverter::GetOutputFilename() const  */
public";

%csmethodmodifiers  gdcm::DictConverter::GetOutputType " /** int
gdcm::DictConverter::GetOutputType() const  */ public";

%csmethodmodifiers  gdcm::DictConverter::SetDictName " /** void
gdcm::DictConverter::SetDictName(const char *name)  */ public";

%csmethodmodifiers  gdcm::DictConverter::SetInputFileName " /** void
gdcm::DictConverter::SetInputFileName(const char *filename)  */
public";

%csmethodmodifiers  gdcm::DictConverter::SetOutputFileName " /** void
gdcm::DictConverter::SetOutputFileName(const char *filename)  */
public";

%csmethodmodifiers  gdcm::DictConverter::SetOutputType " /** void
gdcm::DictConverter::SetOutputType(int type)  */ public";

%csmethodmodifiers  gdcm::DictConverter::~DictConverter " /**
gdcm::DictConverter::~DictConverter()  */ public";


// File: classgdcm_1_1DictEntry.xml
%typemap("csclassmodifiers") gdcm::DictEntry " /** Class to represent
an Entry in the Dict Does not really exist within the DICOM
definition, just a way to minimize storage and have a mapping from
gdcm::Tag to the needed information.

bla TODO FIXME: Need a PublicDictEntry...indeed DictEntry has a notion
of retired which does not exist in PrivateDictEntry...

See:   gdcm::Dict

C++ includes: gdcmDictEntry.h */ public class";

%csmethodmodifiers  gdcm::DictEntry::DictEntry " /**
gdcm::DictEntry::DictEntry(const char *name=\"\", VR const
&vr=VR::INVALID, VM const &vm=VM::VM0, bool ret=false)  */ public";

%csmethodmodifiers  gdcm::DictEntry::GetKeyword " /** const char*
gdcm::DictEntry::GetKeyword() const

same as GetName but without spaces

*/ public";

%csmethodmodifiers  gdcm::DictEntry::GetName " /** const char*
gdcm::DictEntry::GetName() const

Set/Get Name.

*/ public";

%csmethodmodifiers  gdcm::DictEntry::GetRetired " /** bool
gdcm::DictEntry::GetRetired() const

Set/Get Retired flag.

*/ public";

%csmethodmodifiers  gdcm::DictEntry::GetVM " /** const VM&
gdcm::DictEntry::GetVM() const

Set/Get VM.

*/ public";

%csmethodmodifiers  gdcm::DictEntry::GetVR " /** const VR&
gdcm::DictEntry::GetVR() const

Set/Get VR.

*/ public";

%csmethodmodifiers  gdcm::DictEntry::IsUnique " /** bool
gdcm::DictEntry::IsUnique() const

Return whether the name of the DataElement can be considered to be
unique. As of 2008 all elements name were unique (except the
expclitely 'XX' ones)

*/ public";

%csmethodmodifiers  gdcm::DictEntry::SetElementXX " /** void
gdcm::DictEntry::SetElementXX(bool v)

Set whether element is shared in multiple elements (Source Image IDs
typically).

*/ public";

%csmethodmodifiers  gdcm::DictEntry::SetGroupXX " /** void
gdcm::DictEntry::SetGroupXX(bool v)

Set whether element is shared in multiple groups (Curve/Overlay
typically).

*/ public";

%csmethodmodifiers  gdcm::DictEntry::SetName " /** void
gdcm::DictEntry::SetName(const char *name)  */ public";

%csmethodmodifiers  gdcm::DictEntry::SetRetired " /** void
gdcm::DictEntry::SetRetired(bool retired)  */ public";

%csmethodmodifiers  gdcm::DictEntry::SetVM " /** void
gdcm::DictEntry::SetVM(VM const &vm)  */ public";

%csmethodmodifiers  gdcm::DictEntry::SetVR " /** void
gdcm::DictEntry::SetVR(const VR &vr)  */ public";


// File: classgdcm_1_1DictPrinter.xml
%typemap("csclassmodifiers") gdcm::DictPrinter " /**  DictPrinter
class.

C++ includes: gdcmDictPrinter.h */ public class";

%csmethodmodifiers  gdcm::DictPrinter::DictPrinter " /**
gdcm::DictPrinter::DictPrinter()  */ public";

%csmethodmodifiers  gdcm::DictPrinter::Print " /** void
gdcm::DictPrinter::Print(std::ostream &os)  */ public";

%csmethodmodifiers  gdcm::DictPrinter::~DictPrinter " /**
gdcm::DictPrinter::~DictPrinter()  */ public";


// File: classgdcm_1_1Dicts.xml
%typemap("csclassmodifiers") gdcm::Dicts " /** Class to manipulate the
sum of knowledge (all the dict user load).

bla

C++ includes: gdcmDicts.h */ public class";

%csmethodmodifiers  gdcm::Dicts::Dicts " /** gdcm::Dicts::Dicts()  */
public";

%csmethodmodifiers  gdcm::Dicts::GetCSAHeaderDict " /** const
CSAHeaderDict& gdcm::Dicts::GetCSAHeaderDict() const  */ public";

%csmethodmodifiers  gdcm::Dicts::GetDictEntry " /** const DictEntry&
gdcm::Dicts::GetDictEntry(const Tag &tag, const char *owner=NULL)
const  */ public";

%csmethodmodifiers  gdcm::Dicts::GetPrivateDict " /** const
PrivateDict& gdcm::Dicts::GetPrivateDict() const  */ public";

%csmethodmodifiers  gdcm::Dicts::GetPublicDict " /** const Dict&
gdcm::Dicts::GetPublicDict() const  */ public";

%csmethodmodifiers  gdcm::Dicts::IsEmpty " /** bool
gdcm::Dicts::IsEmpty() const  */ public";

%csmethodmodifiers  gdcm::Dicts::~Dicts " /** gdcm::Dicts::~Dicts()
*/ public";


// File: classgdcm_1_1DirectionCosines.xml
%typemap("csclassmodifiers") gdcm::DirectionCosines " /** class to
handle DirectionCosines

C++ includes: gdcmDirectionCosines.h */ public class";

%csmethodmodifiers  gdcm::DirectionCosines::ComputeDistAlongNormal "
/** double gdcm::DirectionCosines::ComputeDistAlongNormal(const double
ipp[3]) const  */ public";

%csmethodmodifiers  gdcm::DirectionCosines::Cross " /** void
gdcm::DirectionCosines::Cross(double z[3]) const

Compute Cross product.

*/ public";

%csmethodmodifiers  gdcm::DirectionCosines::CrossDot " /** double
gdcm::DirectionCosines::CrossDot(DirectionCosines const &dc) const  */
public";

%csmethodmodifiers  gdcm::DirectionCosines::DirectionCosines " /**
gdcm::DirectionCosines::DirectionCosines(const double dircos[6])  */
public";

%csmethodmodifiers  gdcm::DirectionCosines::DirectionCosines " /**
gdcm::DirectionCosines::DirectionCosines()  */ public";

%csmethodmodifiers  gdcm::DirectionCosines::Dot " /** double
gdcm::DirectionCosines::Dot() const

Compute Dot.

*/ public";

%csmethodmodifiers  gdcm::DirectionCosines::IsValid " /** bool
gdcm::DirectionCosines::IsValid() const

Return whether or not this is a valid direction cosines.

*/ public";

%csmethodmodifiers  gdcm::DirectionCosines::Normalize " /** void
gdcm::DirectionCosines::Normalize()

Normalize in-place.

*/ public";

%csmethodmodifiers  gdcm::DirectionCosines::Print " /** void
gdcm::DirectionCosines::Print(std::ostream &) const

Print.

*/ public";

%csmethodmodifiers  gdcm::DirectionCosines::SetFromString " /** bool
gdcm::DirectionCosines::SetFromString(const char *str)  */ public";

%csmethodmodifiers  gdcm::DirectionCosines::~DirectionCosines " /**
gdcm::DirectionCosines::~DirectionCosines()  */ public";


// File: classgdcm_1_1Directory.xml
%typemap("csclassmodifiers") gdcm::Directory " /** Class for
manipulation directories.

This implementation provide a cross platform implementation for
manipulating directores: basically traversing directories and
harvesting files

will not take into account unix type hidden file recursive option will
not look into UNIX type hidden directory (those starting with a '.')

Since python or C# provide there own equivalent implementation, in
which case gdcm::Directory does not make much sense.

C++ includes: gdcmDirectory.h */ public class";

%csmethodmodifiers  gdcm::Directory::Directory " /**
gdcm::Directory::Directory()  */ public";

%csmethodmodifiers  gdcm::Directory::GetDirectories " /**
FilenamesType const& gdcm::Directory::GetDirectories() const

Return the Directories traversed.

*/ public";

%csmethodmodifiers  gdcm::Directory::GetFilenames " /** FilenamesType
const& gdcm::Directory::GetFilenames() const

Set/Get the file names within the directory.

*/ public";

%csmethodmodifiers  gdcm::Directory::GetToplevel " /** FilenameType
const& gdcm::Directory::GetToplevel() const

Get the name of the toplevel directory.

*/ public";

%csmethodmodifiers  gdcm::Directory::Load " /** unsigned int
gdcm::Directory::Load(FilenameType const &name, bool recursive=false)

construct a list of filenames and subdirectory beneath directory: name
WARNING:  : hidden file and hidden directory are not loaded.

*/ public";

%csmethodmodifiers  gdcm::Directory::Print " /** void
gdcm::Directory::Print(std::ostream &os=std::cout)

Print.

*/ public";

%csmethodmodifiers  gdcm::Directory::~Directory " /**
gdcm::Directory::~Directory()  */ public";


// File: classstd_1_1domain__error.xml
%typemap("csclassmodifiers") std::domain_error " /** STL class.

*/ public class";


// File: classgdcm_1_1DummyValueGenerator.xml
%typemap("csclassmodifiers") gdcm::DummyValueGenerator " /** Class for
generating dummy value.

bla

C++ includes: gdcmDummyValueGenerator.h */ public class";


// File: classgdcm_1_1Dumper.xml
%typemap("csclassmodifiers") gdcm::Dumper " /**  Codec class.

Use it to simply dump value read from the file. No interpretation is
done. But it is real fast ! Almost no overhead

C++ includes: gdcmDumper.h */ public class";

%csmethodmodifiers  gdcm::Dumper::Dumper " /** gdcm::Dumper::Dumper()
*/ public";

%csmethodmodifiers  gdcm::Dumper::~Dumper " /**
gdcm::Dumper::~Dumper()  */ public";


// File: classgdcm_1_1Element.xml
%typemap("csclassmodifiers") gdcm::Element " /**  Element class.

TODO

C++ includes: gdcmElement.h */ public class";

%csmethodmodifiers  gdcm::Element::GetLength " /** unsigned long
gdcm::Element< TVR, TVM >::GetLength() const  */ public";

%csmethodmodifiers  gdcm::Element::GetValue " /** VRToType<TVR>::Type&
gdcm::Element< TVR, TVM >::GetValue(unsigned int idx=0)  */ public";

%csmethodmodifiers  gdcm::Element::GetValue " /** const
VRToType<TVR>::Type& gdcm::Element< TVR, TVM >::GetValue(unsigned int
idx=0) const  */ public";

%csmethodmodifiers  gdcm::Element::GetValues " /** const
VRToType<TVR>::Type* gdcm::Element< TVR, TVM >::GetValues() const  */
public";

%csmethodmodifiers  gdcm::Element::Print " /** void gdcm::Element<
TVR, TVM >::Print(std::ostream &_os) const  */ public";

%csmethodmodifiers  gdcm::Element::Read " /** void gdcm::Element< TVR,
TVM >::Read(std::istream &_is)  */ public";

%csmethodmodifiers  gdcm::Element::Set " /** void gdcm::Element< TVR,
TVM >::Set(Value const &v)  */ public";

%csmethodmodifiers  gdcm::Element::SetFromDataElement " /** void
gdcm::Element< TVR, TVM >::SetFromDataElement(DataElement const &de)
*/ public";

%csmethodmodifiers  gdcm::Element::SetValue " /** void gdcm::Element<
TVR, TVM >::SetValue(typename VRToType< TVR >::Type v, unsigned int
idx=0)  */ public";

%csmethodmodifiers  gdcm::Element::Write " /** void gdcm::Element<
TVR, TVM >::Write(std::ostream &_os) const  */ public";


// File: classgdcm_1_1Element_3_01TVR_00_01VM_1_1VM1__n_01_4.xml
%typemap("csclassmodifiers") gdcm::Element< TVR, VM::VM1_n > " /**C++
includes: gdcmElement.h */ public class";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM1_n >::Element " /**
gdcm::Element< TVR, VM::VM1_n >::Element(const Element &_val)  */
public";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM1_n >::Element " /**
gdcm::Element< TVR, VM::VM1_n >::Element()  */ public";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM1_n >::GetAsDataElement
" /** DataElement gdcm::Element< TVR, VM::VM1_n >::GetAsDataElement()
const  */ public";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM1_n >::GetLength " /**
unsigned long gdcm::Element< TVR, VM::VM1_n >::GetLength() const  */
public";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM1_n >::GetValue " /**
VRToType<TVR>::Type& gdcm::Element< TVR, VM::VM1_n
>::GetValue(unsigned int idx=0)  */ public";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM1_n >::GetValue " /**
const VRToType<TVR>::Type& gdcm::Element< TVR, VM::VM1_n
>::GetValue(unsigned int idx=0) const  */ public";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM1_n >::Print " /** void
gdcm::Element< TVR, VM::VM1_n >::Print(std::ostream &_os) const  */
public";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM1_n >::Read " /** void
gdcm::Element< TVR, VM::VM1_n >::Read(std::istream &_is)  */ public";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM1_n >::Set " /** void
gdcm::Element< TVR, VM::VM1_n >::Set(Value const &v)  */ public";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM1_n >::SetArray " /**
void gdcm::Element< TVR, VM::VM1_n >::SetArray(const Type *array,
unsigned long len, bool save=false)  */ public";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM1_n >::SetLength " /**
void gdcm::Element< TVR, VM::VM1_n >::SetLength(unsigned long len)  */
public";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM1_n >::SetValue " /**
void gdcm::Element< TVR, VM::VM1_n >::SetValue(typename VRToType< TVR
>::Type v, unsigned int idx=0)  */ public";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM1_n >::Write " /** void
gdcm::Element< TVR, VM::VM1_n >::Write(std::ostream &_os) const  */
public";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM1_n >::WriteASCII " /**
void gdcm::Element< TVR, VM::VM1_n >::WriteASCII(std::ostream &os)
const  */ public";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM1_n >::~Element " /**
gdcm::Element< TVR, VM::VM1_n >::~Element()  */ public";


// File: classgdcm_1_1Element_3_01TVR_00_01VM_1_1VM2__2n_01_4.xml
%typemap("csclassmodifiers") gdcm::Element< TVR, VM::VM2_2n > " /**C++
includes: gdcmElement.h */ public class";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM2_2n >::SetLength " /**
void gdcm::Element< TVR, VM::VM2_2n >::SetLength(int len)  */ public";


// File: classgdcm_1_1Element_3_01TVR_00_01VM_1_1VM2__n_01_4.xml
%typemap("csclassmodifiers") gdcm::Element< TVR, VM::VM2_n > " /**C++
includes: gdcmElement.h */ public class";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM2_n >::SetLength " /**
void gdcm::Element< TVR, VM::VM2_n >::SetLength(int len)  */ public";


// File: classgdcm_1_1Element_3_01TVR_00_01VM_1_1VM3__3n_01_4.xml
%typemap("csclassmodifiers") gdcm::Element< TVR, VM::VM3_3n > " /**C++
includes: gdcmElement.h */ public class";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM3_3n >::SetLength " /**
void gdcm::Element< TVR, VM::VM3_3n >::SetLength(int len)  */ public";


// File: classgdcm_1_1Element_3_01TVR_00_01VM_1_1VM3__n_01_4.xml
%typemap("csclassmodifiers") gdcm::Element< TVR, VM::VM3_n > " /**C++
includes: gdcmElement.h */ public class";

%csmethodmodifiers  gdcm::Element< TVR, VM::VM3_n >::SetLength " /**
void gdcm::Element< TVR, VM::VM3_n >::SetLength(int len)  */ public";


// File: classgdcm_1_1Element_3_01VR_1_1AS_00_01VM_1_1VM5_01_4.xml
%typemap("csclassmodifiers") gdcm::Element< VR::AS, VM::VM5 > " /**C++
includes: gdcmElement.h */ public class";

%csmethodmodifiers  gdcm::Element< VR::AS, VM::VM5 >::GetLength " /**
unsigned long gdcm::Element< VR::AS, VM::VM5 >::GetLength() const  */
public";

%csmethodmodifiers  gdcm::Element< VR::AS, VM::VM5 >::Print " /** void
gdcm::Element< VR::AS, VM::VM5 >::Print(std::ostream &_os) const  */
public";


// File: classgdcm_1_1Element_3_01VR_1_1OB_00_01VM_1_1VM1_01_4.xml
%typemap("csclassmodifiers") gdcm::Element< VR::OB, VM::VM1 > " /**C++
includes: gdcmElement.h */ public class";


// File: classgdcm_1_1Element_3_01VR_1_1OW_00_01VM_1_1VM1_01_4.xml
%typemap("csclassmodifiers") gdcm::Element< VR::OW, VM::VM1 > " /**C++
includes: gdcmElement.h */ public class";


// File: classgdcm_1_1EncapsulatedDocument.xml
%typemap("csclassmodifiers") gdcm::EncapsulatedDocument " /**
EncapsulatedDocument.

C++ includes: gdcmEncapsulatedDocument.h */ public class";

%csmethodmodifiers  gdcm::EncapsulatedDocument::EncapsulatedDocument "
/** gdcm::EncapsulatedDocument::EncapsulatedDocument()  */ public";


// File: classgdcm_1_1EncodingImplementation_3_01VR_1_1VRASCII_01_4.xml
%typemap("csclassmodifiers") gdcm::EncodingImplementation< VR::VRASCII
> " /**C++ includes: gdcmElement.h */ public class";

%csmethodmodifiers  gdcm::EncodingImplementation< VR::VRASCII >::Write
" /** void gdcm::EncodingImplementation< VR::VRASCII >::Write(const
double *data, unsigned long length, std::ostream &_os)  */ public";

%csmethodmodifiers  gdcm::EncodingImplementation< VR::VRASCII >::Write
" /** void gdcm::EncodingImplementation< VR::VRASCII >::Write(const
float *data, unsigned long length, std::ostream &_os)  */ public";


// File: classgdcm_1_1EncodingImplementation_3_01VR_1_1VRBINARY_01_4.xml
%typemap("csclassmodifiers") gdcm::EncodingImplementation<
VR::VRBINARY > " /**C++ includes: gdcmElement.h */ public class";


// File: classgdcm_1_1EnumeratedValues.xml
%typemap("csclassmodifiers") gdcm::EnumeratedValues " /** Enumerated
Values are used when the specified explicit Values are the only Values
allowed for a Data Element. A Data Element with Enumerated Values that
does not have a Value equivalent to one of the Values specified in
this standard has an invalid value within the scope of a specific
Information Object/SOP Class definition. Note: 1. Patient Sex (0010,
0040) is an example of a Data Element having Enumerated Values. It is
defined to have a Value that is either \"M, \"F, or \"O (see PS 3.3).
No other Value shall be given to this Data Element. 2. Future
modifications of this standard may add to the set of allowed values
for Data Elements with Enumerated Values. Such additions by themselves
may or may not require a change in SOP Class UIDs, depending on the
semantics of the Data Element.

C++ includes: gdcmEnumeratedValues.h */ public class";

%csmethodmodifiers  gdcm::EnumeratedValues::EnumeratedValues " /**
gdcm::EnumeratedValues::EnumeratedValues()  */ public";


// File: classstd_1_1exception.xml
%typemap("csclassmodifiers") std::exception " /** STL class.

*/ public class";


// File: classgdcm_1_1Exception.xml
%typemap("csclassmodifiers") gdcm::Exception " /**  Exception.

Standard exception handling object.

C++ includes: gdcmException.h */ public class";

%csmethodmodifiers  gdcm::Exception::Exception " /**
gdcm::Exception::Exception(const char *desc=\"None\", const char
*file=__FILE__, unsigned int lineNumber=__LINE__, const char
*loc=\"\")  */ public";

%csmethodmodifiers  gdcm::Exception::GetDescription " /** const char*
gdcm::Exception::GetDescription() const

Return the Description.

*/ public";

%csmethodmodifiers  gdcm::Exception::what " /** const char*
gdcm::Exception::what() const  throw () what implementation

*/ public";

%csmethodmodifiers  gdcm::Exception::~Exception " /** virtual
gdcm::Exception::~Exception()  throw () */ public";


// File: classgdcm_1_1ExplicitDataElement.xml
%typemap("csclassmodifiers") gdcm::ExplicitDataElement " /** Class to
read/write a DataElement as Explicit Data Element.

bla

C++ includes: gdcmExplicitDataElement.h */ public class";

%csmethodmodifiers  gdcm::ExplicitDataElement::GetLength " /** VL
gdcm::ExplicitDataElement::GetLength() const  */ public";

%csmethodmodifiers  gdcm::ExplicitDataElement::Read " /**
std::istream& gdcm::ExplicitDataElement::Read(std::istream &is)  */
public";

%csmethodmodifiers  gdcm::ExplicitDataElement::ReadWithLength " /**
std::istream& gdcm::ExplicitDataElement::ReadWithLength(std::istream
&is, VL &length)  */ public";

%csmethodmodifiers  gdcm::ExplicitDataElement::Write " /** const
std::ostream& gdcm::ExplicitDataElement::Write(std::ostream &os) const
*/ public";


// File: classgdcm_1_1ExplicitImplicitDataElement.xml
%typemap("csclassmodifiers") gdcm::ExplicitImplicitDataElement " /**
Class to read/write a DataElement as ExplicitImplicit Data Element.

This only happen for some Philips images Should I derive from
ExplicitDataElement instead ? This is the class that is the closest
the GDCM1.x parser. At each element we try first to read it as
explicit, if this fails, then we try again as an implicit element.

C++ includes: gdcmExplicitImplicitDataElement.h */ public class";

%csmethodmodifiers  gdcm::ExplicitImplicitDataElement::GetLength " /**
VL gdcm::ExplicitImplicitDataElement::GetLength() const  */ public";

%csmethodmodifiers  gdcm::ExplicitImplicitDataElement::Read " /**
std::istream& gdcm::ExplicitImplicitDataElement::Read(std::istream
&is)  */ public";

%csmethodmodifiers  gdcm::ExplicitImplicitDataElement::ReadWithLength
" /** std::istream&
gdcm::ExplicitImplicitDataElement::ReadWithLength(std::istream &is, VL
&length)  */ public";


// File: classgdcm_1_1Fiducials.xml
%typemap("csclassmodifiers") gdcm::Fiducials " /**  Fiducials.

C++ includes: gdcmFiducials.h */ public class";

%csmethodmodifiers  gdcm::Fiducials::Fiducials " /**
gdcm::Fiducials::Fiducials()  */ public";


// File: classgdcm_1_1File.xml
%typemap("csclassmodifiers") gdcm::File " /** a DICOM File See PS 3.10
File: A File is an ordered string of zero or more bytes, where the
first byte is at the beginning of the file and the last byte at the
end of the File. Files are identified by a unique File ID and may by
written, read and/or deleted.

C++ includes: gdcmFile.h */ public class";

%csmethodmodifiers  gdcm::File::File " /** gdcm::File::File()  */
public";

%csmethodmodifiers  gdcm::File::GetDataSet " /** DataSet&
gdcm::File::GetDataSet()  */ public";

%csmethodmodifiers  gdcm::File::GetDataSet " /** const DataSet&
gdcm::File::GetDataSet() const  */ public";

%csmethodmodifiers  gdcm::File::GetHeader " /** FileMetaInformation&
gdcm::File::GetHeader()  */ public";

%csmethodmodifiers  gdcm::File::GetHeader " /** const
FileMetaInformation& gdcm::File::GetHeader() const  */ public";

%csmethodmodifiers  gdcm::File::Read " /** std::istream&
gdcm::File::Read(std::istream &is)  */ public";

%csmethodmodifiers  gdcm::File::SetDataSet " /** void
gdcm::File::SetDataSet(const DataSet &ds)  */ public";

%csmethodmodifiers  gdcm::File::SetHeader " /** void
gdcm::File::SetHeader(const FileMetaInformation &fmi)  */ public";

%csmethodmodifiers  gdcm::File::Write " /** std::ostream const&
gdcm::File::Write(std::ostream &os) const  */ public";

%csmethodmodifiers  gdcm::File::~File " /** gdcm::File::~File()  */
public";


// File: classgdcm_1_1FileExplicitFilter.xml
%typemap("csclassmodifiers") gdcm::FileExplicitFilter " /**C++
includes: gdcmFileExplicitFilter.h */ public class";

%csmethodmodifiers  gdcm::FileExplicitFilter::Change " /** bool
gdcm::FileExplicitFilter::Change()

Set FMI Transfer Syntax.

Change

*/ public";

%csmethodmodifiers  gdcm::FileExplicitFilter::FileExplicitFilter " /**
gdcm::FileExplicitFilter::FileExplicitFilter()  */ public";

%csmethodmodifiers  gdcm::FileExplicitFilter::GetFile " /** File&
gdcm::FileExplicitFilter::GetFile()  */ public";

%csmethodmodifiers  gdcm::FileExplicitFilter::SetChangePrivateTags "
/** void gdcm::FileExplicitFilter::SetChangePrivateTags(bool b)

Decide whether or not to VR'ify private tags.

*/ public";

%csmethodmodifiers  gdcm::FileExplicitFilter::SetFile " /** void
gdcm::FileExplicitFilter::SetFile(const File &f)

Set/Get File.

*/ public";

%csmethodmodifiers  gdcm::FileExplicitFilter::SetRecomputeItemLength "
/** void gdcm::FileExplicitFilter::SetRecomputeItemLength(bool b)

By default set Sequence & Item length to Undefined to avoid
recomputing length:.

*/ public";

%csmethodmodifiers
gdcm::FileExplicitFilter::SetRecomputeSequenceLength " /** void
gdcm::FileExplicitFilter::SetRecomputeSequenceLength(bool b)  */
public";

%csmethodmodifiers  gdcm::FileExplicitFilter::SetUseVRUN " /** void
gdcm::FileExplicitFilter::SetUseVRUN(bool b)

When VR=16bits in explicit but Implicit has a 32bits length, use
VR=UN.

*/ public";

%csmethodmodifiers  gdcm::FileExplicitFilter::~FileExplicitFilter "
/** gdcm::FileExplicitFilter::~FileExplicitFilter()  */ public";


// File: classgdcm_1_1FileMetaInformation.xml
%typemap("csclassmodifiers") gdcm::FileMetaInformation " /** Class to
represent a File Meta Information.

FileMetaInformation is a Explicit Structured Set. Whenever the file
contains an ImplicitDataElement DataSet, a conversion will take place.
Todo If user adds an element with group != 0x0002 it will be
written... Definition: The File Meta Information includes identifying
information on the encapsulated Data Set. This header consists of a
128 byte File Preamble, followed by a 4 byte DICOM prefix, followed by
the File Meta Elements shown in Table 7.1-1. This header shall be
present in every DICOM file.

C++ includes: gdcmFileMetaInformation.h */ public class";

%csmethodmodifiers  gdcm::FileMetaInformation::FileMetaInformation "
/** gdcm::FileMetaInformation::FileMetaInformation(FileMetaInformation
const &fmi)  */ public";

%csmethodmodifiers  gdcm::FileMetaInformation::FileMetaInformation "
/** gdcm::FileMetaInformation::FileMetaInformation()  */ public";

%csmethodmodifiers  gdcm::FileMetaInformation::FillFromDataSet " /**
void gdcm::FileMetaInformation::FillFromDataSet(DataSet const &ds)

Construct a FileMetaInformation from an already existing DataSet:.

*/ public";

%csmethodmodifiers
gdcm::FileMetaInformation::GetDataSetTransferSyntax " /** const
TransferSyntax& gdcm::FileMetaInformation::GetDataSetTransferSyntax()
const  */ public";

%csmethodmodifiers  gdcm::FileMetaInformation::GetMediaStorage " /**
MediaStorage gdcm::FileMetaInformation::GetMediaStorage() const  */
public";

%csmethodmodifiers  gdcm::FileMetaInformation::GetMetaInformationTS "
/** TransferSyntax::NegociatedType
gdcm::FileMetaInformation::GetMetaInformationTS() const  */ public";

%csmethodmodifiers  gdcm::FileMetaInformation::GetPreamble " /**
Preamble& gdcm::FileMetaInformation::GetPreamble()  */ public";

%csmethodmodifiers  gdcm::FileMetaInformation::GetPreamble " /** const
Preamble& gdcm::FileMetaInformation::GetPreamble() const

Get Preamble.

*/ public";

%csmethodmodifiers  gdcm::FileMetaInformation::Insert " /** void
gdcm::FileMetaInformation::Insert(const DataElement &de)

Insert a DataElement in the DataSet. WARNING:  : Tag need to be >= 0x8
to be considered valid data element

*/ public";

%csmethodmodifiers  gdcm::FileMetaInformation::IsValid " /** bool
gdcm::FileMetaInformation::IsValid() const  */ public";

%csmethodmodifiers  gdcm::FileMetaInformation::Read " /**
std::istream& gdcm::FileMetaInformation::Read(std::istream &is)

Read.

*/ public";

%csmethodmodifiers  gdcm::FileMetaInformation::ReadCompat " /**
std::istream& gdcm::FileMetaInformation::ReadCompat(std::istream &is)
*/ public";

%csmethodmodifiers  gdcm::FileMetaInformation::Replace " /** void
gdcm::FileMetaInformation::Replace(const DataElement &de)

Replace a dataelement with another one.

*/ public";

%csmethodmodifiers
gdcm::FileMetaInformation::SetDataSetTransferSyntax " /** void
gdcm::FileMetaInformation::SetDataSetTransferSyntax(const
TransferSyntax &ts)  */ public";

%csmethodmodifiers  gdcm::FileMetaInformation::SetPreamble " /** void
gdcm::FileMetaInformation::SetPreamble(const Preamble &p)  */ public";

%csmethodmodifiers  gdcm::FileMetaInformation::Write " /**
std::ostream& gdcm::FileMetaInformation::Write(std::ostream &os) const

Write.

*/ public";

%csmethodmodifiers  gdcm::FileMetaInformation::~FileMetaInformation "
/** gdcm::FileMetaInformation::~FileMetaInformation()  */ public";


// File: classgdcm_1_1Filename.xml
%typemap("csclassmodifiers") gdcm::Filename " /** Class to manipulate
file name's.

OS independant representation of a filename (to query path, name and
extension from a filename)

C++ includes: gdcmFilename.h */ public class";

%csmethodmodifiers  gdcm::Filename::Filename " /**
gdcm::Filename::Filename(const char *filename=\"\")  */ public";

%csmethodmodifiers  gdcm::Filename::GetExtension " /** const char*
gdcm::Filename::GetExtension()

return only the extension part of a filename

*/ public";

%csmethodmodifiers  gdcm::Filename::GetFileName " /** const char*
gdcm::Filename::GetFileName() const

Return the full filename.

*/ public";

%csmethodmodifiers  gdcm::Filename::GetName " /** const char*
gdcm::Filename::GetName()

return only the name part of a filename

*/ public";

%csmethodmodifiers  gdcm::Filename::GetPath " /** const char*
gdcm::Filename::GetPath()

Return only the path component of a filename.

*/ public";

%csmethodmodifiers  gdcm::Filename::IsEmpty " /** bool
gdcm::Filename::IsEmpty() const

return whether the filename is empty

*/ public";

%csmethodmodifiers  gdcm::Filename::IsIdentical " /** bool
gdcm::Filename::IsIdentical(Filename const &fn) const  */ public";

%csmethodmodifiers  gdcm::Filename::ToUnixSlashes " /** const char*
gdcm::Filename::ToUnixSlashes()

Convert backslash (windows style) to UNIX style slash.

*/ public";


// File: classgdcm_1_1FilenameGenerator.xml
%typemap("csclassmodifiers") gdcm::FilenameGenerator " /**
FilenameGenerator.

class to generate filenames based on a pattern (C-style)

Output will be:

for i = 0, number of filenames: outfilename[i] = prefix + (pattern %
i)

where pattern % i means C-style snprintf of Pattern using value 'i'

C++ includes: gdcmFilenameGenerator.h */ public class";

%csmethodmodifiers  gdcm::FilenameGenerator::FilenameGenerator " /**
gdcm::FilenameGenerator::FilenameGenerator()  */ public";

%csmethodmodifiers  gdcm::FilenameGenerator::Generate " /** bool
gdcm::FilenameGenerator::Generate()

Generate (return success).

*/ public";

%csmethodmodifiers  gdcm::FilenameGenerator::GetFilename " /** const
char* gdcm::FilenameGenerator::GetFilename(unsigned int n) const

Get a particular filename (call after Generate).

*/ public";

%csmethodmodifiers  gdcm::FilenameGenerator::GetFilenames " /**
FilenamesType const& gdcm::FilenameGenerator::GetFilenames() const

Return all filenames.

*/ public";

%csmethodmodifiers  gdcm::FilenameGenerator::GetNumberOfFilenames "
/** unsigned int gdcm::FilenameGenerator::GetNumberOfFilenames() const
*/ public";

%csmethodmodifiers  gdcm::FilenameGenerator::GetPattern " /** const
char* gdcm::FilenameGenerator::GetPattern() const  */ public";

%csmethodmodifiers  gdcm::FilenameGenerator::GetPrefix " /** const
char* gdcm::FilenameGenerator::GetPrefix() const  */ public";

%csmethodmodifiers  gdcm::FilenameGenerator::SetNumberOfFilenames "
/** void gdcm::FilenameGenerator::SetNumberOfFilenames(unsigned int
nfiles)

Set/Get the number of filenames to generate.

*/ public";

%csmethodmodifiers  gdcm::FilenameGenerator::SetPattern " /** void
gdcm::FilenameGenerator::SetPattern(const char *pattern)

Set/Get pattern.

*/ public";

%csmethodmodifiers  gdcm::FilenameGenerator::SetPrefix " /** void
gdcm::FilenameGenerator::SetPrefix(const char *prefix)

Set/Get prefix.

*/ public";

%csmethodmodifiers  gdcm::FilenameGenerator::~FilenameGenerator " /**
gdcm::FilenameGenerator::~FilenameGenerator()  */ public";


// File: classgdcm_1_1FileSet.xml
%typemap("csclassmodifiers") gdcm::FileSet " /** File-set: A File-set
is a collection of DICOM Files (and possibly non- DICOM Files) that
share a common naming space within which File IDs are unique.

C++ includes: gdcmFileSet.h */ public class";

%csmethodmodifiers  gdcm::FileSet::AddFile " /** bool
gdcm::FileSet::AddFile(const char *filename)

Add a file 'filename' to the list of files. Return true on success,
false in case filename could not be found on system.

*/ public";

%csmethodmodifiers  gdcm::FileSet::AddFile " /** void
gdcm::FileSet::AddFile(File const &)

Deprecated . Does nothing

*/ public";

%csmethodmodifiers  gdcm::FileSet::FileSet " /**
gdcm::FileSet::FileSet()  */ public";

%csmethodmodifiers  gdcm::FileSet::GetFiles " /** FilesType const&
gdcm::FileSet::GetFiles() const  */ public";

%csmethodmodifiers  gdcm::FileSet::SetFiles " /** void
gdcm::FileSet::SetFiles(FilesType const &files)  */ public";


// File: classgdcm_1_1FileWithName.xml
%typemap("csclassmodifiers") gdcm::FileWithName " /**  SerieHelper.

Backward only class do not use in newer code

C++ includes: gdcmSerieHelper.h */ public class";

%csmethodmodifiers  gdcm::FileWithName::FileWithName " /**
gdcm::FileWithName::FileWithName(File &f)  */ public";


// File: classgdcm_1_1Fragment.xml
%typemap("csclassmodifiers") gdcm::Fragment " /** Class to represent a
Fragment.

C++ includes: gdcmFragment.h */ public class";

%csmethodmodifiers  gdcm::Fragment::Fragment " /**
gdcm::Fragment::Fragment()  */ public";

%csmethodmodifiers  gdcm::Fragment::GetLength " /** VL
gdcm::Fragment::GetLength() const  */ public";

%csmethodmodifiers  gdcm::Fragment::Read " /** std::istream&
gdcm::Fragment::Read(std::istream &is)  */ public";

%csmethodmodifiers  gdcm::Fragment::Write " /** std::ostream&
gdcm::Fragment::Write(std::ostream &os) const  */ public";


// File: classstd_1_1fstream.xml
%typemap("csclassmodifiers") std::fstream " /** STL class.

*/ public class";


// File: classitk_1_1GDCMImageIO2.xml
%typemap("csclassmodifiers") itk::GDCMImageIO2 " /** ImageIO class for
reading and writing DICOM V3.0 and ACR/NEMA (V1.0 & V2.0) images This
class is only an adaptor to the gdcm library (currently gdcm 2.0 is
used):.

http://gdcm.sourceforge.net

WARNING:  this class is deprecated, as gdcm 2.x has been integrated in
ITK starting ITK 3.12

C++ includes: itkGDCMImageIO2.h */ public class";

%csmethodmodifiers  itk::GDCMImageIO2::CanReadFile " /** virtual bool
itk::GDCMImageIO2::CanReadFile(const char *)

Determine the file type. Returns true if this ImageIO can read the
file specified.

*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::CanWriteFile " /** virtual bool
itk::GDCMImageIO2::CanWriteFile(const char *)

Determine the file type. Returns true if this ImageIO can write the
file specified. GDCM triggers on \".dcm\" and \".dicom\".

*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::GetBodyPart " /** void
itk::GDCMImageIO2::GetBodyPart(char *part)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::GetInstitution " /** void
itk::GDCMImageIO2::GetInstitution(char *ins)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::GetManufacturer " /** void
itk::GDCMImageIO2::GetManufacturer(char *manu)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::GetModality " /** void
itk::GDCMImageIO2::GetModality(char *modality)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::GetModel " /** void
itk::GDCMImageIO2::GetModel(char *model)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::GetNumberOfSeriesInStudy " /**
void itk::GDCMImageIO2::GetNumberOfSeriesInStudy(char *series)  */
public";

%csmethodmodifiers  itk::GDCMImageIO2::GetNumberOfStudyRelatedSeries "
/** void itk::GDCMImageIO2::GetNumberOfStudyRelatedSeries(char
*series)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::GetPatientAge " /** void
itk::GDCMImageIO2::GetPatientAge(char *age)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::GetPatientDOB " /** void
itk::GDCMImageIO2::GetPatientDOB(char *dob)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::GetPatientID " /** void
itk::GDCMImageIO2::GetPatientID(char *id)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::GetPatientName " /** void
itk::GDCMImageIO2::GetPatientName(char *name)

Convenience methods to query patient information and scanner
information. These methods are here for compatibility with the
DICOMImageIO2 class.

*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::GetPatientSex " /** void
itk::GDCMImageIO2::GetPatientSex(char *sex)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::GetScanOptions " /** void
itk::GDCMImageIO2::GetScanOptions(char *options)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::GetStudyDate " /** void
itk::GDCMImageIO2::GetStudyDate(char *date)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::GetStudyDescription " /** void
itk::GDCMImageIO2::GetStudyDescription(char *desc)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::GetStudyID " /** void
itk::GDCMImageIO2::GetStudyID(char *id)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::GetValueFromTag " /** bool
itk::GDCMImageIO2::GetValueFromTag(const std::string &tag, std::string
&value)

More general method to retrieve an arbitrary DICOM value based on a
DICOM Tag (eg \"0123|4567\"). WARNING: You need to use the lower case
for hex 0x[a-f], for instance: \"0020|000d\" instead of \"0020|000D\"
(the latter won't work)

*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkBooleanMacro " /**
itk::GDCMImageIO2::itkBooleanMacro(LoadPrivateTags)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkBooleanMacro " /**
itk::GDCMImageIO2::itkBooleanMacro(LoadSequences)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkBooleanMacro " /**
itk::GDCMImageIO2::itkBooleanMacro(KeepOriginalUID)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkGetEnumMacro " /**
itk::GDCMImageIO2::itkGetEnumMacro(CompressionType, TCompressionType)
*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkGetMacro " /**
itk::GDCMImageIO2::itkGetMacro(LoadPrivateTags, bool)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkGetMacro " /**
itk::GDCMImageIO2::itkGetMacro(LoadSequences, bool)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkGetMacro " /**
itk::GDCMImageIO2::itkGetMacro(KeepOriginalUID, bool)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkGetMacro " /**
itk::GDCMImageIO2::itkGetMacro(RescaleIntercept, double)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkGetMacro " /**
itk::GDCMImageIO2::itkGetMacro(RescaleSlope, double)

Macro to access Rescale Slope and Rescale Intercept. Which are needed
to rescale properly image when needed. User then need to Always check
those value when access value from the DICOM header

*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkGetStringMacro " /**
itk::GDCMImageIO2::itkGetStringMacro(FrameOfReferenceInstanceUID)  */
public";

%csmethodmodifiers  itk::GDCMImageIO2::itkGetStringMacro " /**
itk::GDCMImageIO2::itkGetStringMacro(SeriesInstanceUID)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkGetStringMacro " /**
itk::GDCMImageIO2::itkGetStringMacro(StudyInstanceUID)

Access the generated DICOM UID's.

*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkGetStringMacro " /**
itk::GDCMImageIO2::itkGetStringMacro(UIDPrefix)

Macro to access the DICOM UID prefix. By default this is the ITK root
id. This default can be overriden if the exam is for example part of
an existing study.

*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkNewMacro " /**
itk::GDCMImageIO2::itkNewMacro(Self)

Method for creation through the object factory.

*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkSetEnumMacro " /**
itk::GDCMImageIO2::itkSetEnumMacro(CompressionType, TCompressionType)
*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkSetMacro " /**
itk::GDCMImageIO2::itkSetMacro(LoadPrivateTags, bool)

Parse any private tags in the DICOM file. Defaults to the value of
LoadPrivateTagsDefault. Loading DICOM files is faster when private
tags are not needed.

*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkSetMacro " /**
itk::GDCMImageIO2::itkSetMacro(LoadSequences, bool)

Parse any sequences in the DICOM file. Defaults to the value of
LoadSequencesDefault. Loading DICOM files is faster when sequences are
not needed.

*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkSetMacro " /**
itk::GDCMImageIO2::itkSetMacro(MaxSizeLoadEntry, long)

A DICOM file can contains multiple binary stream that can be very long
For example an Overlay on the image. Most of the time user do not want
to load this binary structure in memory since it can consume lot of
memory. Therefore any field that is bigger than the default value
0xfff is discarded and just seek'd This method allow advanced user to
force the reading of such field

*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkSetMacro " /**
itk::GDCMImageIO2::itkSetMacro(KeepOriginalUID, bool)

Preserve the original DICOM UID of the input files

*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkSetStringMacro " /**
itk::GDCMImageIO2::itkSetStringMacro(UIDPrefix)  */ public";

%csmethodmodifiers  itk::GDCMImageIO2::itkTypeMacro " /**
itk::GDCMImageIO2::itkTypeMacro(GDCMImageIO2, Superclass)

Run-time type information (and related methods).

*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::Read " /** virtual void
itk::GDCMImageIO2::Read(void *buffer)

Reads the data from disk into the memory buffer provided.

*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::ReadImageInformation " /**
virtual void itk::GDCMImageIO2::ReadImageInformation()

Set the spacing and dimesion information for the current filename.

*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::Write " /** virtual void
itk::GDCMImageIO2::Write(const void *buffer)

Writes the data to disk from the memory buffer provided. Make sure
that the IORegion has been set properly.

*/ public";

%csmethodmodifiers  itk::GDCMImageIO2::WriteImageInformation " /**
virtual void itk::GDCMImageIO2::WriteImageInformation()

Writes the spacing and dimentions of the image. Assumes SetFileName
has been called with a valid file name.

*/ public";


// File: classgdcm_1_1Global.xml
%typemap("csclassmodifiers") gdcm::Global " /**  Global.

Global should be included in any translation unit that will use Dict
or that implements the singleton pattern. It makes sure that the Dict
singleton is created before and destroyed after all other singletons
in GDCM.

C++ includes: gdcmGlobal.h */ public class";

%csmethodmodifiers  gdcm::Global::Append " /** bool
gdcm::Global::Append(const char *path)

Append path at the end of the path list WARNING:  not thread safe !

*/ public";

%csmethodmodifiers  gdcm::Global::GetDefs " /** Defs const&
gdcm::Global::GetDefs() const

retrieve the default/internal (Part 3) You need to explicitely call
LoadResourcesFiles before

*/ public";

%csmethodmodifiers  gdcm::Global::GetDicts " /** Dicts const&
gdcm::Global::GetDicts() const

retrieve the default/internal dicts (Part 6) This dict is filled up at
load time

*/ public";

%csmethodmodifiers  gdcm::Global::Global " /** gdcm::Global::Global()
*/ public";

%csmethodmodifiers  gdcm::Global::LoadResourcesFiles " /** bool
gdcm::Global::LoadResourcesFiles()

Load all internal XML files, ressource path need to have been set
before calling this member function (see Append/Prepend members func)
WARNING:  not thread safe !

*/ public";

%csmethodmodifiers  gdcm::Global::Prepend " /** bool
gdcm::Global::Prepend(const char *path)

Prepend path at the begining of the path list WARNING:  not thread
safe !

*/ public";

%csmethodmodifiers  gdcm::Global::~Global " /**
gdcm::Global::~Global()  */ public";


// File: classgdcm_1_1GroupDict.xml
%typemap("csclassmodifiers") gdcm::GroupDict " /** Class to represent
the mapping from group number to its abbreviation and name.

Should I rewrite this class to use a std::map instead of std::vector
for problem of memory consumption ?

C++ includes: gdcmGroupDict.h */ public class";

%csmethodmodifiers  gdcm::GroupDict::GetAbbreviation " /** std::string
const& gdcm::GroupDict::GetAbbreviation(uint16_t num) const  */
public";

%csmethodmodifiers  gdcm::GroupDict::GetName " /** std::string const&
gdcm::GroupDict::GetName(uint16_t num) const  */ public";

%csmethodmodifiers  gdcm::GroupDict::GroupDict " /**
gdcm::GroupDict::GroupDict()  */ public";

%csmethodmodifiers  gdcm::GroupDict::Size " /** unsigned long
gdcm::GroupDict::Size() const  */ public";

%csmethodmodifiers  gdcm::GroupDict::~GroupDict " /**
gdcm::GroupDict::~GroupDict()  */ public";


// File: classgdcm_1_1HAVEGE.xml
%typemap("csclassmodifiers") gdcm::HAVEGE " /**C++ includes:
gdcmHAVEGE.h */ public class";

%csmethodmodifiers  gdcm::HAVEGE::HAVEGE " /** gdcm::HAVEGE::HAVEGE()
*/ public";

%csmethodmodifiers  gdcm::HAVEGE::Rand " /** int gdcm::HAVEGE::Rand()

HAVEGE rand function.

A random int

*/ public";

%csmethodmodifiers  gdcm::HAVEGE::~HAVEGE " /**
gdcm::HAVEGE::~HAVEGE()  */ public";


// File: classstd_1_1ifstream.xml
%typemap("csclassmodifiers") std::ifstream " /** STL class.

*/ public class";


// File: classgdcm_1_1Image.xml
%typemap("csclassmodifiers") gdcm::Image " /**  Image.

This is the container for an Image in the general sense. From this
container you should be able to request information like: Origin

Dimension

PixelFormat ... But also to retrieve the image as a raw buffer (char
*) Since we have to deal with both RAW data and JPEG stream (which
internally encode all the above information) this API might seems
redundant. One way to solve that would be to subclass gdcm::Image with
gdcm::JPEGImage which would from the stream extract the header info
and fill it to please gdcm::Image...well except origin for instance

Basically you can see it as a storage for the PixelData element.
However it was also used for MRSpectroscopy object (as proof of
concept)

C++ includes: gdcmImage.h */ public class";

%csmethodmodifiers  gdcm::Image::GetDirectionCosines " /** double
gdcm::Image::GetDirectionCosines(unsigned int idx) const  */ public";

%csmethodmodifiers  gdcm::Image::GetDirectionCosines " /** const
double* gdcm::Image::GetDirectionCosines() const

Return a 6-tuples specifying the direction cosines A default value of
(1,0,0,0,1,0) will be return when the direction cosines was not
specified.

*/ public";

%csmethodmodifiers  gdcm::Image::GetIntercept " /** double
gdcm::Image::GetIntercept() const  */ public";

%csmethodmodifiers  gdcm::Image::GetOrigin " /** double
gdcm::Image::GetOrigin(unsigned int idx) const  */ public";

%csmethodmodifiers  gdcm::Image::GetOrigin " /** const double*
gdcm::Image::GetOrigin() const

Return a 3-tuples specifying the origin Will return (0,0,0) if the
origin was not specified.

*/ public";

%csmethodmodifiers  gdcm::Image::GetSlope " /** double
gdcm::Image::GetSlope() const  */ public";

%csmethodmodifiers  gdcm::Image::GetSpacing " /** double
gdcm::Image::GetSpacing(unsigned int idx) const  */ public";

%csmethodmodifiers  gdcm::Image::GetSpacing " /** const double*
gdcm::Image::GetSpacing() const

Return a 3-tuples specifying the spacing NOTE: 3rd value can be an
aribtrary 1 value when the spacing was not specified (ex. 2D image).
WARNING: when the spacing is not specifier, a default value of 1 will
be returned

*/ public";

%csmethodmodifiers  gdcm::Image::GetSwapCode " /** SwapCode
gdcm::Image::GetSwapCode() const

DEPRECATED DO NOT USE.

*/ public";

%csmethodmodifiers  gdcm::Image::Image " /** gdcm::Image::Image()  */
public";

%csmethodmodifiers  gdcm::Image::Print " /** void
gdcm::Image::Print(std::ostream &os) const

print

*/ public";

%csmethodmodifiers  gdcm::Image::SetDirectionCosines " /** void
gdcm::Image::SetDirectionCosines(unsigned int idx, double dircos)  */
public";

%csmethodmodifiers  gdcm::Image::SetDirectionCosines " /** void
gdcm::Image::SetDirectionCosines(const double *dircos)  */ public";

%csmethodmodifiers  gdcm::Image::SetDirectionCosines " /** void
gdcm::Image::SetDirectionCosines(const float *dircos)  */ public";

%csmethodmodifiers  gdcm::Image::SetIntercept " /** void
gdcm::Image::SetIntercept(double intercept)

intercept

*/ public";

%csmethodmodifiers  gdcm::Image::SetOrigin " /** void
gdcm::Image::SetOrigin(unsigned int idx, double ori)  */ public";

%csmethodmodifiers  gdcm::Image::SetOrigin " /** void
gdcm::Image::SetOrigin(const double *ori)  */ public";

%csmethodmodifiers  gdcm::Image::SetOrigin " /** void
gdcm::Image::SetOrigin(const float *ori)  */ public";

%csmethodmodifiers  gdcm::Image::SetSlope " /** void
gdcm::Image::SetSlope(double slope)

slope

*/ public";

%csmethodmodifiers  gdcm::Image::SetSpacing " /** void
gdcm::Image::SetSpacing(unsigned int idx, double spacing)  */ public";

%csmethodmodifiers  gdcm::Image::SetSpacing " /** void
gdcm::Image::SetSpacing(const double *spacing)  */ public";

%csmethodmodifiers  gdcm::Image::SetSwapCode " /** void
gdcm::Image::SetSwapCode(SwapCode sc)  */ public";

%csmethodmodifiers  gdcm::Image::~Image " /** gdcm::Image::~Image()
*/ public";


// File: classgdcm_1_1ImageApplyLookupTable.xml
%typemap("csclassmodifiers") gdcm::ImageApplyLookupTable " /**
ImageApplyLookupTable class It applies the LUT the PixelData (only
PALETTE_COLOR images) Output will be a PhotometricInterpretation=RGB
image.

C++ includes: gdcmImageApplyLookupTable.h */ public class";

%csmethodmodifiers  gdcm::ImageApplyLookupTable::Apply " /** bool
gdcm::ImageApplyLookupTable::Apply()

Apply.

*/ public";

%csmethodmodifiers  gdcm::ImageApplyLookupTable::ImageApplyLookupTable
" /** gdcm::ImageApplyLookupTable::ImageApplyLookupTable()  */
public";

%csmethodmodifiers
gdcm::ImageApplyLookupTable::~ImageApplyLookupTable " /**
gdcm::ImageApplyLookupTable::~ImageApplyLookupTable()  */ public";


// File: classgdcm_1_1ImageChangePhotometricInterpretation.xml
%typemap("csclassmodifiers")
gdcm::ImageChangePhotometricInterpretation " /**
ImageChangePhotometricInterpretation class Class to change the
Photometric Interpetation of an input DICOM.

C++ includes: gdcmImageChangePhotometricInterpretation.h */ public
class";

%csmethodmodifiers  gdcm::ImageChangePhotometricInterpretation::Change
" /** bool gdcm::ImageChangePhotometricInterpretation::Change()

Change.

*/ public";

%csmethodmodifiers
gdcm::ImageChangePhotometricInterpretation::GetPhotometricInterpretation
" /** const PhotometricInterpretation&
gdcm::ImageChangePhotometricInterpretation::GetPhotometricInterpretation()
const  */ public";

%csmethodmodifiers
gdcm::ImageChangePhotometricInterpretation::ImageChangePhotometricInterpretation
" /**
gdcm::ImageChangePhotometricInterpretation::ImageChangePhotometricInterpretation()
*/ public";

%csmethodmodifiers
gdcm::ImageChangePhotometricInterpretation::SetPhotometricInterpretation
" /** void
gdcm::ImageChangePhotometricInterpretation::SetPhotometricInterpretation(PhotometricInterpretation
const &pi)

Set/Get requested PhotometricInterpretation.

*/ public";

%csmethodmodifiers
gdcm::ImageChangePhotometricInterpretation::~ImageChangePhotometricInterpretation
" /**
gdcm::ImageChangePhotometricInterpretation::~ImageChangePhotometricInterpretation()
*/ public";


// File: classgdcm_1_1ImageChangePlanarConfiguration.xml
%typemap("csclassmodifiers") gdcm::ImageChangePlanarConfiguration "
/**  ImageChangePlanarConfiguration class Class to change the Planar
configuration of an input DICOM By default it will change into the
more usual reprensentation: PlanarConfiguration = 0.

C++ includes: gdcmImageChangePlanarConfiguration.h */ public class";

%csmethodmodifiers  gdcm::ImageChangePlanarConfiguration::Change " /**
bool gdcm::ImageChangePlanarConfiguration::Change()

Change.

*/ public";

%csmethodmodifiers
gdcm::ImageChangePlanarConfiguration::GetPlanarConfiguration " /**
unsigned int
gdcm::ImageChangePlanarConfiguration::GetPlanarConfiguration() const
*/ public";

%csmethodmodifiers
gdcm::ImageChangePlanarConfiguration::ImageChangePlanarConfiguration "
/**
gdcm::ImageChangePlanarConfiguration::ImageChangePlanarConfiguration()
*/ public";

%csmethodmodifiers
gdcm::ImageChangePlanarConfiguration::SetPlanarConfiguration " /**
void
gdcm::ImageChangePlanarConfiguration::SetPlanarConfiguration(unsigned
int pc)

Set/Get requested PlanarConfigation.

*/ public";

%csmethodmodifiers
gdcm::ImageChangePlanarConfiguration::~ImageChangePlanarConfiguration
" /**
gdcm::ImageChangePlanarConfiguration::~ImageChangePlanarConfiguration()
*/ public";


// File: classgdcm_1_1ImageChangeTransferSyntax.xml
%typemap("csclassmodifiers") gdcm::ImageChangeTransferSyntax " /**
ImageChangeTransferSyntax class Class to change the transfer syntax of
an input DICOM.

If only Force param is set but no input TransferSyntax is set, it is
assumed that user only wants to inspect encapsulated stream (advanced
dev. option).

C++ includes: gdcmImageChangeTransferSyntax.h */ public class";

%csmethodmodifiers  gdcm::ImageChangeTransferSyntax::Change " /** bool
gdcm::ImageChangeTransferSyntax::Change()

Change.

*/ public";

%csmethodmodifiers  gdcm::ImageChangeTransferSyntax::GetTransferSyntax
" /** const TransferSyntax&
gdcm::ImageChangeTransferSyntax::GetTransferSyntax() const

Get Transfer Syntax.

*/ public";

%csmethodmodifiers
gdcm::ImageChangeTransferSyntax::ImageChangeTransferSyntax " /**
gdcm::ImageChangeTransferSyntax::ImageChangeTransferSyntax()  */
public";

%csmethodmodifiers
gdcm::ImageChangeTransferSyntax::SetCompressIconImage " /** void
gdcm::ImageChangeTransferSyntax::SetCompressIconImage(bool b)

Decide whether or not to also compress the Icon Image using the same
Transfer Syntax Default is to simply decompress icon image

*/ public";

%csmethodmodifiers  gdcm::ImageChangeTransferSyntax::SetForce " /**
void gdcm::ImageChangeTransferSyntax::SetForce(bool f)

When target Transfer Syntax is identical to input target syntax, no
operation is actually done This is an issue when someone wants to
recompress using GDCM internal implementation a JPEG (for example)
image

*/ public";

%csmethodmodifiers  gdcm::ImageChangeTransferSyntax::SetTransferSyntax
" /** void gdcm::ImageChangeTransferSyntax::SetTransferSyntax(const
TransferSyntax &ts)

Set target Transfer Syntax.

*/ public";

%csmethodmodifiers  gdcm::ImageChangeTransferSyntax::SetUserCodec "
/** void gdcm::ImageChangeTransferSyntax::SetUserCodec(ImageCodec *ic)
*/ public";

%csmethodmodifiers
gdcm::ImageChangeTransferSyntax::~ImageChangeTransferSyntax " /**
gdcm::ImageChangeTransferSyntax::~ImageChangeTransferSyntax()  */
public";


// File: classgdcm_1_1ImageCodec.xml
%typemap("csclassmodifiers") gdcm::ImageCodec " /**  ImageCodec.

Main codec, this is a central place for all implementation

C++ includes: gdcmImageCodec.h */ public class";

%csmethodmodifiers  gdcm::ImageCodec::CanDecode " /** bool
gdcm::ImageCodec::CanDecode(TransferSyntax const &) const

Return whether this decoder support this transfer syntax (can decode
it).

*/ public";

%csmethodmodifiers  gdcm::ImageCodec::Decode " /** bool
gdcm::ImageCodec::Decode(DataElement const &is, DataElement &os)

Decode.

*/ public";

%csmethodmodifiers  gdcm::ImageCodec::GetDimensions " /** const
unsigned int* gdcm::ImageCodec::GetDimensions() const  */ public";

%csmethodmodifiers  gdcm::ImageCodec::GetHeaderInfo " /** virtual bool
gdcm::ImageCodec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
*/ public";

%csmethodmodifiers  gdcm::ImageCodec::GetLUT " /** const LookupTable&
gdcm::ImageCodec::GetLUT() const  */ public";

%csmethodmodifiers  gdcm::ImageCodec::GetNeedByteSwap " /** bool
gdcm::ImageCodec::GetNeedByteSwap() const  */ public";

%csmethodmodifiers  gdcm::ImageCodec::GetPhotometricInterpretation "
/** const PhotometricInterpretation&
gdcm::ImageCodec::GetPhotometricInterpretation() const  */ public";

%csmethodmodifiers  gdcm::ImageCodec::GetPixelFormat " /** const
PixelFormat& gdcm::ImageCodec::GetPixelFormat() const  */ public";

%csmethodmodifiers  gdcm::ImageCodec::GetPlanarConfiguration " /**
unsigned int gdcm::ImageCodec::GetPlanarConfiguration() const  */
public";

%csmethodmodifiers  gdcm::ImageCodec::ImageCodec " /**
gdcm::ImageCodec::ImageCodec()  */ public";

%csmethodmodifiers  gdcm::ImageCodec::IsLossy " /** bool
gdcm::ImageCodec::IsLossy() const  */ public";

%csmethodmodifiers  gdcm::ImageCodec::SetDimensions " /** void
gdcm::ImageCodec::SetDimensions(const unsigned int *d)  */ public";

%csmethodmodifiers  gdcm::ImageCodec::SetLUT " /** void
gdcm::ImageCodec::SetLUT(LookupTable const &lut)  */ public";

%csmethodmodifiers  gdcm::ImageCodec::SetNeedByteSwap " /** void
gdcm::ImageCodec::SetNeedByteSwap(bool b)  */ public";

%csmethodmodifiers  gdcm::ImageCodec::SetNeedOverlayCleanup " /** void
gdcm::ImageCodec::SetNeedOverlayCleanup(bool b)  */ public";

%csmethodmodifiers  gdcm::ImageCodec::SetNumberOfDimensions " /** void
gdcm::ImageCodec::SetNumberOfDimensions(unsigned int dim)  */ public";

%csmethodmodifiers  gdcm::ImageCodec::SetPhotometricInterpretation "
/** void
gdcm::ImageCodec::SetPhotometricInterpretation(PhotometricInterpretation
const &pi)  */ public";

%csmethodmodifiers  gdcm::ImageCodec::SetPixelFormat " /** virtual
void gdcm::ImageCodec::SetPixelFormat(PixelFormat const &pf)  */
public";

%csmethodmodifiers  gdcm::ImageCodec::SetPlanarConfiguration " /**
void gdcm::ImageCodec::SetPlanarConfiguration(unsigned int pc)  */
public";

%csmethodmodifiers  gdcm::ImageCodec::~ImageCodec " /**
gdcm::ImageCodec::~ImageCodec()  */ public";


// File: classgdcm_1_1ImageConverter.xml
%typemap("csclassmodifiers") gdcm::ImageConverter " /**  Image
Converter.

This is the class used to convert from on gdcm::Image to another This
is typically used to convert let say YBR JPEG compressed gdcm::Image
to a RAW RGB gdcm::Image. So that the buffer can be directly pass to
third party application. This filter is application level and not
integrated directly in GDCM

C++ includes: gdcmImageConverter.h */ public class";

%csmethodmodifiers  gdcm::ImageConverter::Convert " /** void
gdcm::ImageConverter::Convert()  */ public";

%csmethodmodifiers  gdcm::ImageConverter::GetOuput " /** const Image&
gdcm::ImageConverter::GetOuput() const  */ public";

%csmethodmodifiers  gdcm::ImageConverter::ImageConverter " /**
gdcm::ImageConverter::ImageConverter()  */ public";

%csmethodmodifiers  gdcm::ImageConverter::SetInput " /** void
gdcm::ImageConverter::SetInput(Image const &input)  */ public";

%csmethodmodifiers  gdcm::ImageConverter::~ImageConverter " /**
gdcm::ImageConverter::~ImageConverter()  */ public";


// File: classgdcm_1_1ImageFragmentSplitter.xml
%typemap("csclassmodifiers") gdcm::ImageFragmentSplitter " /**
ImageFragmentSplitter class For single frame image, DICOM standard
allow splitting the frame into multiple fragments.

C++ includes: gdcmImageFragmentSplitter.h */ public class";

%csmethodmodifiers  gdcm::ImageFragmentSplitter::GetFragmentSizeMax "
/** unsigned int gdcm::ImageFragmentSplitter::GetFragmentSizeMax()
const  */ public";

%csmethodmodifiers  gdcm::ImageFragmentSplitter::ImageFragmentSplitter
" /** gdcm::ImageFragmentSplitter::ImageFragmentSplitter()  */
public";

%csmethodmodifiers  gdcm::ImageFragmentSplitter::SetForce " /** void
gdcm::ImageFragmentSplitter::SetForce(bool f)

When file already has all it's segment < FragmentSizeMax there is not
need to run the filter. Unless the user explicitly say 'force'
recomputation !

*/ public";

%csmethodmodifiers  gdcm::ImageFragmentSplitter::SetFragmentSizeMax "
/** void gdcm::ImageFragmentSplitter::SetFragmentSizeMax(unsigned int
fragsize)

FragmentSizeMax needs to be an even number.

*/ public";

%csmethodmodifiers  gdcm::ImageFragmentSplitter::Split " /** bool
gdcm::ImageFragmentSplitter::Split()

Split.

*/ public";

%csmethodmodifiers
gdcm::ImageFragmentSplitter::~ImageFragmentSplitter " /**
gdcm::ImageFragmentSplitter::~ImageFragmentSplitter()  */ public";


// File: classgdcm_1_1ImageHelper.xml
%typemap("csclassmodifiers") gdcm::ImageHelper " /**  ImageHelper
(internal class, not intended for user level).

Helper for writing World images in DICOM. DICOM has a 'template'
approach to image where MR Image Storage are distinct object from
Enhanced MR Image Storage. For example the Pixel Spacing in one object
is not at the same position (ie Tag) as in the other this class is the
central (read: fragile) place where all the dispatching is done from a
unified view of a world image (typically VTK or ITK point of view)
down to the low level DICOM point of view.

WARNING:  : do not expect the API of this class to be maintained at
any point, since as Modalities are added the API might have to be
augmented or behavior changed to cope with new modalities.

C++ includes: gdcmImageHelper.h */ public class";


// File: classgdcm_1_1ImageReader.xml
%typemap("csclassmodifiers") gdcm::ImageReader " /**  ImageReader.

its role is to convert the DICOM DataSet into a gdcm::Image
representation By default it is also loading the lookup table and
overlay when found as they impact the rendering or the image  See PS
3.3-2008, Table C.7-11b IMAGE PIXEL MACRO ATTRIBUTES for the list of
attribute that belong to what gdcm calls a 'Image'

C++ includes: gdcmImageReader.h */ public class";

%csmethodmodifiers  gdcm::ImageReader::GetImage " /** Image&
gdcm::ImageReader::GetImage()  */ public";

%csmethodmodifiers  gdcm::ImageReader::GetImage " /** const Image&
gdcm::ImageReader::GetImage() const

Return the read image.

*/ public";

%csmethodmodifiers  gdcm::ImageReader::ImageReader " /**
gdcm::ImageReader::ImageReader()  */ public";

%csmethodmodifiers  gdcm::ImageReader::Read " /** bool
gdcm::ImageReader::Read()

Read the DICOM image. There are two reason for failure: 1. The input
filename is not DICOM 2. The input DICOM file does not contains an
Image.

*/ public";

%csmethodmodifiers  gdcm::ImageReader::~ImageReader " /**
gdcm::ImageReader::~ImageReader()  */ public";


// File: classgdcm_1_1ImageToImageFilter.xml
%typemap("csclassmodifiers") gdcm::ImageToImageFilter " /**
ImageToImageFilter class Super class for all filter taking an image
and producing an output image.

C++ includes: gdcmImageToImageFilter.h */ public class";

%csmethodmodifiers  gdcm::ImageToImageFilter::GetOutput " /** const
Image& gdcm::ImageToImageFilter::GetOutput() const

Get Output image.

*/ public";

%csmethodmodifiers  gdcm::ImageToImageFilter::ImageToImageFilter " /**
gdcm::ImageToImageFilter::ImageToImageFilter()  */ public";

%csmethodmodifiers  gdcm::ImageToImageFilter::~ImageToImageFilter "
/** gdcm::ImageToImageFilter::~ImageToImageFilter()  */ public";


// File: classgdcm_1_1ImageWriter.xml
%typemap("csclassmodifiers") gdcm::ImageWriter " /**  ImageWriter.

C++ includes: gdcmImageWriter.h */ public class";

%csmethodmodifiers  gdcm::ImageWriter::GetImage " /** Image&
gdcm::ImageWriter::GetImage()  */ public";

%csmethodmodifiers  gdcm::ImageWriter::GetImage " /** const Image&
gdcm::ImageWriter::GetImage() const

Set/Get Image to be written It will overwrite anything Image infos
found in DataSet (see parent class to see how to pass dataset)

*/ public";

%csmethodmodifiers  gdcm::ImageWriter::ImageWriter " /**
gdcm::ImageWriter::ImageWriter()  */ public";

%csmethodmodifiers  gdcm::ImageWriter::Write " /** bool
gdcm::ImageWriter::Write()

Write.

*/ public";

%csmethodmodifiers  gdcm::ImageWriter::~ImageWriter " /**
gdcm::ImageWriter::~ImageWriter()  */ public";


// File: classgdcm_1_1ImplicitDataElement.xml
%typemap("csclassmodifiers") gdcm::ImplicitDataElement " /** Class to
represent an *Implicit VR* Data Element.

bla

C++ includes: gdcmImplicitDataElement.h */ public class";

%csmethodmodifiers  gdcm::ImplicitDataElement::GetLength " /** VL
gdcm::ImplicitDataElement::GetLength() const  */ public";

%csmethodmodifiers  gdcm::ImplicitDataElement::Read " /**
std::istream& gdcm::ImplicitDataElement::Read(std::istream &is)  */
public";

%csmethodmodifiers  gdcm::ImplicitDataElement::ReadWithLength " /**
std::istream& gdcm::ImplicitDataElement::ReadWithLength(std::istream
&is, VL &length)  */ public";

%csmethodmodifiers  gdcm::ImplicitDataElement::Write " /** const
std::ostream& gdcm::ImplicitDataElement::Write(std::ostream &os) const
*/ public";


// File: classstd_1_1invalid__argument.xml
%typemap("csclassmodifiers") std::invalid_argument " /** STL class.

*/ public class";


// File: classgdcm_1_1IOD.xml
%typemap("csclassmodifiers") gdcm::IOD " /**C++ includes: gdcmIOD.h */
public class";

%csmethodmodifiers  gdcm::IOD::AddIODEntry " /** void
gdcm::IOD::AddIODEntry(const IODEntry &iode)  */ public";

%csmethodmodifiers  gdcm::IOD::Clear " /** void gdcm::IOD::Clear()  */
public";

%csmethodmodifiers  gdcm::IOD::GetIODEntry " /** const IODEntry&
gdcm::IOD::GetIODEntry(unsigned int idx) const  */ public";

%csmethodmodifiers  gdcm::IOD::GetNumberOfIODs " /** unsigned int
gdcm::IOD::GetNumberOfIODs() const  */ public";

%csmethodmodifiers  gdcm::IOD::IOD " /** gdcm::IOD::IOD()  */ public";


// File: classgdcm_1_1IODEntry.xml
%typemap("csclassmodifiers") gdcm::IODEntry " /** Class for
representing a IODEntry.

A.1.3 IOD Module Table and Functional Group Macro Table This Section
of each IOD defines in a tabular form the Modules comprising the IOD.
The following information must be specified for each Module in the
table: The name of the Module or Functional Group

A reference to the Section in Annex C which defines the Module or
Functional Group

The usage of the Module or Functional Group; whether it is:

Mandatory (see A.1.3.1) , abbreviated M

Conditional (see A.1.3.2) , abbreviated C

User Option (see A.1.3.3) , abbreviated U The Modules referenced are
defined in Annex C. A.1.3.1 MANDATORY MODULES For each IOD, Mandatory
Modules shall be supported per the definitions, semantics and
requirements defined in Annex C. PS 3.3 - 2008 Page 96

Standard - A.1.3.2 CONDITIONAL MODULES Conditional Modules are
Mandatory Modules if specific conditions are met. If the specified
conditions are not met, this Module shall not be supported; that is,
no information defined in that Module shall be sent. A.1.3.3 USER
OPTION MODULES User Option Modules may or may not be supported. If an
optional Module is supported, the Attribute Types specified in the
Modules in Annex C shall be supported.

See:   DictEntry

C++ includes: gdcmIODEntry.h */ public class";

%csmethodmodifiers  gdcm::IODEntry::GetIE " /** const char*
gdcm::IODEntry::GetIE() const  */ public";

%csmethodmodifiers  gdcm::IODEntry::GetName " /** const char*
gdcm::IODEntry::GetName() const  */ public";

%csmethodmodifiers  gdcm::IODEntry::GetRef " /** const char*
gdcm::IODEntry::GetRef() const  */ public";

%csmethodmodifiers  gdcm::IODEntry::GetUsage " /** const char*
gdcm::IODEntry::GetUsage() const  */ public";

%csmethodmodifiers  gdcm::IODEntry::GetUsageType " /**
Usage::UsageType gdcm::IODEntry::GetUsageType() const  */ public";

%csmethodmodifiers  gdcm::IODEntry::IODEntry " /**
gdcm::IODEntry::IODEntry(const char *name=\"\", const char *ref=\"\",
const char *usag=\"\")  */ public";

%csmethodmodifiers  gdcm::IODEntry::SetIE " /** void
gdcm::IODEntry::SetIE(const char *ie)  */ public";

%csmethodmodifiers  gdcm::IODEntry::SetName " /** void
gdcm::IODEntry::SetName(const char *name)  */ public";

%csmethodmodifiers  gdcm::IODEntry::SetRef " /** void
gdcm::IODEntry::SetRef(const char *ref)  */ public";

%csmethodmodifiers  gdcm::IODEntry::SetUsage " /** void
gdcm::IODEntry::SetUsage(const char *usag)  */ public";


// File: classgdcm_1_1IODs.xml
%typemap("csclassmodifiers") gdcm::IODs " /** Class for representing a
IODs.

bla

See:   IOD

C++ includes: gdcmIODs.h */ public class";

%csmethodmodifiers  gdcm::IODs::AddIOD " /** void
gdcm::IODs::AddIOD(const char *name, const IOD &module)  */ public";

%csmethodmodifiers  gdcm::IODs::Clear " /** void gdcm::IODs::Clear()
*/ public";

%csmethodmodifiers  gdcm::IODs::GetIOD " /** const IOD&
gdcm::IODs::GetIOD(const char *name) const  */ public";

%csmethodmodifiers  gdcm::IODs::IODs " /** gdcm::IODs::IODs()  */
public";


// File: classstd_1_1ios.xml
%typemap("csclassmodifiers") std::ios " /** STL class.

*/ public class";


// File: classstd_1_1ios__base.xml
%typemap("csclassmodifiers") std::ios_base " /** STL class.

*/ public class";


// File: classstd_1_1ios__base_1_1failure.xml
%typemap("csclassmodifiers") std::ios_base::failure " /** STL class.

*/ public class";


// File: classgdcm_1_1IPPSorter.xml
%typemap("csclassmodifiers") gdcm::IPPSorter " /**  IPPSorter
Implement a simple Image Position ( Patient) sorter, along the Image
Orientation ( Patient) direction. This algorithm does NOT support
duplicate and will FAIL in case of duplicate IPP.

WARNING:  See special note for SetZSpacingTolerance when computing the
ZSpacing from the IPP of each DICOM files (default tolerance for
consistant spacing is: 1e-6mm)

C++ includes: gdcmIPPSorter.h */ public class";

%csmethodmodifiers  gdcm::IPPSorter::GetZSpacing " /** double
gdcm::IPPSorter::GetZSpacing() const

Read-only function to provide access to the computed value for the
Z-Spacing The ComputeZSpacing must have been set to true before
execution of sort algorithm. Call this function *after* calling
Sort(); Z-Spacing will be 0 on 2 occasions: Sorting simply failed,
potentially duplicate IPP => ZSpacing = 0

ZSpacing could not be computed (Z-Spacing is not constant, or
ZTolerance is too low)

*/ public";

%csmethodmodifiers  gdcm::IPPSorter::GetZSpacingTolerance " /** double
gdcm::IPPSorter::GetZSpacingTolerance() const  */ public";

%csmethodmodifiers  gdcm::IPPSorter::IPPSorter " /**
gdcm::IPPSorter::IPPSorter()  */ public";

%csmethodmodifiers  gdcm::IPPSorter::SetComputeZSpacing " /** void
gdcm::IPPSorter::SetComputeZSpacing(bool b)

Functions related to Z-Spacing computation Set to true when sort
algorithm should also perform a regular Z-Spacing computation using
the Image Position ( Patient) Potential reason for failure: 1. ALL
slices are taken into account, if one slice if missing then ZSpacing
will be set to 0 since the spacing will not be found to be regular
along the Series

*/ public";

%csmethodmodifiers  gdcm::IPPSorter::SetZSpacingTolerance " /** void
gdcm::IPPSorter::SetZSpacingTolerance(double tol)

2. Another reason for failure is that that Z-Spacing is only slightly
changing (eg 1e-3) along the serie, a human can determine that this is
ok and change the tolerance from its default value: 1e-6

*/ public";

%csmethodmodifiers  gdcm::IPPSorter::Sort " /** virtual bool
gdcm::IPPSorter::Sort(std::vector< std::string > const &filenames)

Main entry point to the sorter. It will execute the filter, option
should be set before running this function (SetZSpacingTolerance, ...)
Return value indicate if sorting could be achived. Warning this does
*NOT* imply that spacing is consistant, it only means the file are
sorted according to IPP You should check if ZSpacing is 0 or not to
deduce if file are actually a 3D volume

*/ public";

%csmethodmodifiers  gdcm::IPPSorter::~IPPSorter " /**
gdcm::IPPSorter::~IPPSorter()  */ public";


// File: classstd_1_1istream.xml
%typemap("csclassmodifiers") std::istream " /** STL class.

*/ public class";


// File: classstd_1_1istringstream.xml
%typemap("csclassmodifiers") std::istringstream " /** STL class.

*/ public class";


// File: classgdcm_1_1Item.xml
%typemap("csclassmodifiers") gdcm::Item " /** Class to represent an
Item A component of the value of a Data Element that is of Value
Representation Sequence of Items. An Item contains a Data Set . See PS
3.5 7.5.1 Item Encoding Rules Each Item of a Data Element of VR SQ
shall be encoded as a DICOM Standart Data Element with a specific Data
Element Tag of Value (FFFE,E000). The Item Tag is followed by a 4 byte
Item Length field encoded in one of the following two ways Explicit/
Implicit.

ITEM: A component of the Value of a Data Element that is of Value
Representation Sequence of Items. An Item contains a Data Set.

C++ includes: gdcmItem.h */ public class";

%csmethodmodifiers  gdcm::Item::Clear " /** void gdcm::Item::Clear()
*/ public";

%csmethodmodifiers  gdcm::Item::FindDataElement " /** bool
gdcm::Item::FindDataElement(const Tag &t) const  */ public";

%csmethodmodifiers  gdcm::Item::GetDataElement " /** const
DataElement& gdcm::Item::GetDataElement(const Tag &t) const  */
public";

%csmethodmodifiers  gdcm::Item::GetLength " /** VL
gdcm::Item::GetLength() const  */ public";

%csmethodmodifiers  gdcm::Item::GetNestedDataSet " /** DataSet&
gdcm::Item::GetNestedDataSet()  */ public";

%csmethodmodifiers  gdcm::Item::GetNestedDataSet " /** const DataSet&
gdcm::Item::GetNestedDataSet() const  */ public";

%csmethodmodifiers  gdcm::Item::InsertDataElement " /** void
gdcm::Item::InsertDataElement(const DataElement &de)  */ public";

%csmethodmodifiers  gdcm::Item::Item " /** gdcm::Item::Item(Item const
&val)  */ public";

%csmethodmodifiers  gdcm::Item::Item " /** gdcm::Item::Item()  */
public";

%csmethodmodifiers  gdcm::Item::Read " /** std::istream&
gdcm::Item::Read(std::istream &is)  */ public";

%csmethodmodifiers  gdcm::Item::SetNestedDataSet " /** void
gdcm::Item::SetNestedDataSet(const DataSet &nested)  */ public";

%csmethodmodifiers  gdcm::Item::Write " /** const std::ostream&
gdcm::Item::Write(std::ostream &os) const  */ public";


// File: classgdcm_1_1JPEG12Codec.xml
%typemap("csclassmodifiers") gdcm::JPEG12Codec " /** Class to do JPEG
12bits (lossy & lossless).

internal class

C++ includes: gdcmJPEG12Codec.h */ public class";

%csmethodmodifiers  gdcm::JPEG12Codec::Decode " /** bool
gdcm::JPEG12Codec::Decode(std::istream &is, std::ostream &os)  */
public";

%csmethodmodifiers  gdcm::JPEG12Codec::GetHeaderInfo " /** bool
gdcm::JPEG12Codec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
*/ public";

%csmethodmodifiers  gdcm::JPEG12Codec::InternalCode " /** bool
gdcm::JPEG12Codec::InternalCode(const char *input, unsigned long len,
std::ostream &os)  */ public";

%csmethodmodifiers  gdcm::JPEG12Codec::JPEG12Codec " /**
gdcm::JPEG12Codec::JPEG12Codec()  */ public";

%csmethodmodifiers  gdcm::JPEG12Codec::~JPEG12Codec " /**
gdcm::JPEG12Codec::~JPEG12Codec()  */ public";


// File: classgdcm_1_1JPEG16Codec.xml
%typemap("csclassmodifiers") gdcm::JPEG16Codec " /** Class to do JPEG
16bits (lossless).

internal class

C++ includes: gdcmJPEG16Codec.h */ public class";

%csmethodmodifiers  gdcm::JPEG16Codec::Decode " /** bool
gdcm::JPEG16Codec::Decode(std::istream &is, std::ostream &os)  */
public";

%csmethodmodifiers  gdcm::JPEG16Codec::GetHeaderInfo " /** bool
gdcm::JPEG16Codec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
*/ public";

%csmethodmodifiers  gdcm::JPEG16Codec::InternalCode " /** bool
gdcm::JPEG16Codec::InternalCode(const char *input, unsigned long len,
std::ostream &os)  */ public";

%csmethodmodifiers  gdcm::JPEG16Codec::JPEG16Codec " /**
gdcm::JPEG16Codec::JPEG16Codec()  */ public";

%csmethodmodifiers  gdcm::JPEG16Codec::~JPEG16Codec " /**
gdcm::JPEG16Codec::~JPEG16Codec()  */ public";


// File: classgdcm_1_1JPEG2000Codec.xml
%typemap("csclassmodifiers") gdcm::JPEG2000Codec " /** Class to do
JPEG 2000.

the class will produce JPC (JPEG 2000 codestream), since some private
implementor are using full jp2 file the decoder tolerate jp2 input
this is an implementation of an ImageCodec

C++ includes: gdcmJPEG2000Codec.h */ public class";

%csmethodmodifiers  gdcm::JPEG2000Codec::CanCode " /** bool
gdcm::JPEG2000Codec::CanCode(TransferSyntax const &ts) const

Return whether this coder support this transfer syntax (can code it).

*/ public";

%csmethodmodifiers  gdcm::JPEG2000Codec::CanDecode " /** bool
gdcm::JPEG2000Codec::CanDecode(TransferSyntax const &ts) const

Return whether this decoder support this transfer syntax (can decode
it).

*/ public";

%csmethodmodifiers  gdcm::JPEG2000Codec::Code " /** bool
gdcm::JPEG2000Codec::Code(DataElement const &in, DataElement &out)

Code.

*/ public";

%csmethodmodifiers  gdcm::JPEG2000Codec::Decode " /** bool
gdcm::JPEG2000Codec::Decode(DataElement const &is, DataElement &os)

Decode.

*/ public";

%csmethodmodifiers  gdcm::JPEG2000Codec::GetHeaderInfo " /** virtual
bool gdcm::JPEG2000Codec::GetHeaderInfo(std::istream &is,
TransferSyntax &ts)  */ public";

%csmethodmodifiers  gdcm::JPEG2000Codec::GetQuality " /** double
gdcm::JPEG2000Codec::GetQuality(unsigned int idx=0) const  */ public";

%csmethodmodifiers  gdcm::JPEG2000Codec::GetRate " /** double
gdcm::JPEG2000Codec::GetRate(unsigned int idx=0) const  */ public";

%csmethodmodifiers  gdcm::JPEG2000Codec::JPEG2000Codec " /**
gdcm::JPEG2000Codec::JPEG2000Codec()  */ public";

%csmethodmodifiers  gdcm::JPEG2000Codec::SetNumberOfResolutions " /**
void gdcm::JPEG2000Codec::SetNumberOfResolutions(unsigned int nres)
*/ public";

%csmethodmodifiers  gdcm::JPEG2000Codec::SetQuality " /** void
gdcm::JPEG2000Codec::SetQuality(unsigned int idx, double q)  */
public";

%csmethodmodifiers  gdcm::JPEG2000Codec::SetRate " /** void
gdcm::JPEG2000Codec::SetRate(unsigned int idx, double rate)  */
public";

%csmethodmodifiers  gdcm::JPEG2000Codec::SetReversible " /** void
gdcm::JPEG2000Codec::SetReversible(bool res)  */ public";

%csmethodmodifiers  gdcm::JPEG2000Codec::SetTileSize " /** void
gdcm::JPEG2000Codec::SetTileSize(unsigned int tx, unsigned int ty)  */
public";

%csmethodmodifiers  gdcm::JPEG2000Codec::~JPEG2000Codec " /**
gdcm::JPEG2000Codec::~JPEG2000Codec()  */ public";


// File: classgdcm_1_1JPEG8Codec.xml
%typemap("csclassmodifiers") gdcm::JPEG8Codec " /** Class to do JPEG
8bits (lossy & lossless).

internal class

C++ includes: gdcmJPEG8Codec.h */ public class";

%csmethodmodifiers  gdcm::JPEG8Codec::Decode " /** bool
gdcm::JPEG8Codec::Decode(std::istream &is, std::ostream &os)  */
public";

%csmethodmodifiers  gdcm::JPEG8Codec::GetHeaderInfo " /** bool
gdcm::JPEG8Codec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
*/ public";

%csmethodmodifiers  gdcm::JPEG8Codec::InternalCode " /** bool
gdcm::JPEG8Codec::InternalCode(const char *input, unsigned long len,
std::ostream &os)  */ public";

%csmethodmodifiers  gdcm::JPEG8Codec::JPEG8Codec " /**
gdcm::JPEG8Codec::JPEG8Codec()  */ public";

%csmethodmodifiers  gdcm::JPEG8Codec::~JPEG8Codec " /**
gdcm::JPEG8Codec::~JPEG8Codec()  */ public";


// File: classgdcm_1_1JPEGCodec.xml
%typemap("csclassmodifiers") gdcm::JPEGCodec " /** JPEG codec Class to
do JPEG (8bits, 12bits, 16bits lossy & lossless). It redispatch in
between the different codec implementation: gdcm::JPEG8Codec,
gdcm::JPEG12Codec & gdcm::JPEG16Codec It also support inconsistency in
between DICOM header and JPEG compressed stream ImageCodec
implementation for the JPEG case.

Things you should know if you ever want to dive into DICOM/JPEG world
(among other):

http://groups.google.com/group/comp.protocols.dicom/browse_thread/thread/625e46919f2080e1

http://groups.google.com/group/comp.protocols.dicom/browse_thread/thread/75fdfccc65a6243

http://groups.google.com/group/comp.protocols.dicom/browse_thread/thread/2d525ef6a2f093ed

http://groups.google.com/group/comp.protocols.dicom/browse_thread/thread/6b93af410f8c921f

C++ includes: gdcmJPEGCodec.h */ public class";

%csmethodmodifiers  gdcm::JPEGCodec::CanCode " /** bool
gdcm::JPEGCodec::CanCode(TransferSyntax const &ts) const

Return whether this coder support this transfer syntax (can code it).

*/ public";

%csmethodmodifiers  gdcm::JPEGCodec::CanDecode " /** bool
gdcm::JPEGCodec::CanDecode(TransferSyntax const &ts) const

Return whether this decoder support this transfer syntax (can decode
it).

*/ public";

%csmethodmodifiers  gdcm::JPEGCodec::Code " /** bool
gdcm::JPEGCodec::Code(DataElement const &in, DataElement &out)

Compress into JPEG.

*/ public";

%csmethodmodifiers  gdcm::JPEGCodec::ComputeOffsetTable " /** void
gdcm::JPEGCodec::ComputeOffsetTable(bool b)

Compute the offset table:.

*/ public";

%csmethodmodifiers  gdcm::JPEGCodec::Decode " /** bool
gdcm::JPEGCodec::Decode(DataElement const &is, DataElement &os)

Decode.

*/ public";

%csmethodmodifiers  gdcm::JPEGCodec::GetHeaderInfo " /** virtual bool
gdcm::JPEGCodec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
*/ public";

%csmethodmodifiers  gdcm::JPEGCodec::GetLossless " /** bool
gdcm::JPEGCodec::GetLossless() const  */ public";

%csmethodmodifiers  gdcm::JPEGCodec::GetQuality " /** double
gdcm::JPEGCodec::GetQuality() const  */ public";

%csmethodmodifiers  gdcm::JPEGCodec::JPEGCodec " /**
gdcm::JPEGCodec::JPEGCodec()  */ public";

%csmethodmodifiers  gdcm::JPEGCodec::SetLossless " /** void
gdcm::JPEGCodec::SetLossless(bool l)  */ public";

%csmethodmodifiers  gdcm::JPEGCodec::SetPixelFormat " /** void
gdcm::JPEGCodec::SetPixelFormat(PixelFormat const &pf)  */ public";

%csmethodmodifiers  gdcm::JPEGCodec::SetQuality " /** void
gdcm::JPEGCodec::SetQuality(double q)  */ public";

%csmethodmodifiers  gdcm::JPEGCodec::~JPEGCodec " /**
gdcm::JPEGCodec::~JPEGCodec()  */ public";


// File: classgdcm_1_1JPEGLSCodec.xml
%typemap("csclassmodifiers") gdcm::JPEGLSCodec " /** JPEG-LS.

codec that implement the JPEG-LS compression this is an implementation
of ImageCodec for JPEG-LS  It uses the CharLS JPEG-LS implementation

C++ includes: gdcmJPEGLSCodec.h */ public class";

%csmethodmodifiers  gdcm::JPEGLSCodec::CanCode " /** bool
gdcm::JPEGLSCodec::CanCode(TransferSyntax const &ts) const

Return whether this coder support this transfer syntax (can code it).

*/ public";

%csmethodmodifiers  gdcm::JPEGLSCodec::CanDecode " /** bool
gdcm::JPEGLSCodec::CanDecode(TransferSyntax const &ts) const

Return whether this decoder support this transfer syntax (can decode
it).

*/ public";

%csmethodmodifiers  gdcm::JPEGLSCodec::Code " /** bool
gdcm::JPEGLSCodec::Code(DataElement const &in, DataElement &out)

Code.

*/ public";

%csmethodmodifiers  gdcm::JPEGLSCodec::Decode " /** bool
gdcm::JPEGLSCodec::Decode(DataElement const &is, DataElement &os)

Decode.

*/ public";

%csmethodmodifiers  gdcm::JPEGLSCodec::GetBufferLength " /** unsigned
long gdcm::JPEGLSCodec::GetBufferLength() const  */ public";

%csmethodmodifiers  gdcm::JPEGLSCodec::GetHeaderInfo " /** bool
gdcm::JPEGLSCodec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
*/ public";

%csmethodmodifiers  gdcm::JPEGLSCodec::JPEGLSCodec " /**
gdcm::JPEGLSCodec::JPEGLSCodec()  */ public";

%csmethodmodifiers  gdcm::JPEGLSCodec::SetBufferLength " /** void
gdcm::JPEGLSCodec::SetBufferLength(unsigned long l)  */ public";

%csmethodmodifiers  gdcm::JPEGLSCodec::~JPEGLSCodec " /**
gdcm::JPEGLSCodec::~JPEGLSCodec()  */ public";


// File: classstd_1_1length__error.xml
%typemap("csclassmodifiers") std::length_error " /** STL class.

*/ public class";


// File: classstd_1_1list.xml
%typemap("csclassmodifiers") std::list " /** STL class.

*/ public class";


// File: classstd_1_1list_1_1const__iterator.xml
%typemap("csclassmodifiers") std::list::const_iterator " /** STL
iterator class.

*/ public class";


// File: classstd_1_1list_1_1const__reverse__iterator.xml
%typemap("csclassmodifiers") std::list::const_reverse_iterator " /**
STL iterator class.

*/ public class";


// File: classstd_1_1list_1_1iterator.xml
%typemap("csclassmodifiers") std::list::iterator " /** STL iterator
class.

*/ public class";


// File: classstd_1_1list_1_1reverse__iterator.xml
%typemap("csclassmodifiers") std::list::reverse_iterator " /** STL
iterator class.

*/ public class";


// File: classgdcm_1_1LO.xml
%typemap("csclassmodifiers") gdcm::LO " /**  LO.

TODO

C++ includes: gdcmLO.h */ public class";

%csmethodmodifiers  gdcm::LO::IsValid " /** bool gdcm::LO::IsValid()
const  */ public";

%csmethodmodifiers  gdcm::LO::LO " /** gdcm::LO::LO(const Superclass
&s, size_type pos=0, size_type n=npos)  */ public";

%csmethodmodifiers  gdcm::LO::LO " /** gdcm::LO::LO(const value_type
*s, size_type n)  */ public";

%csmethodmodifiers  gdcm::LO::LO " /** gdcm::LO::LO(const value_type
*s)  */ public";

%csmethodmodifiers  gdcm::LO::LO " /** gdcm::LO::LO()  */ public";


// File: classstd_1_1logic__error.xml
%typemap("csclassmodifiers") std::logic_error " /** STL class.

*/ public class";


// File: classgdcm_1_1LookupTable.xml
%typemap("csclassmodifiers") gdcm::LookupTable " /**  LookupTable
class.

C++ includes: gdcmLookupTable.h */ public class";

%csmethodmodifiers  gdcm::LookupTable::Allocate " /** void
gdcm::LookupTable::Allocate(unsigned short bitsample=8)

Allocate the LUT.

*/ public";

%csmethodmodifiers  gdcm::LookupTable::Clear " /** void
gdcm::LookupTable::Clear()

Clear the LUT.

*/ public";

%csmethodmodifiers  gdcm::LookupTable::Decode " /** void
gdcm::LookupTable::Decode(std::istream &is, std::ostream &os) const

Decode the LUT.

*/ public";

%csmethodmodifiers  gdcm::LookupTable::GetBitSample " /** unsigned
short gdcm::LookupTable::GetBitSample() const

return the bit sample

*/ public";

%csmethodmodifiers  gdcm::LookupTable::GetBufferAsRGBA " /** bool
gdcm::LookupTable::GetBufferAsRGBA(unsigned char *rgba) const

return the LUT as RGBA buffer

*/ public";

%csmethodmodifiers  gdcm::LookupTable::GetLUT " /** void
gdcm::LookupTable::GetLUT(LookupTableType type, unsigned char *array,
unsigned int &length) const  */ public";

%csmethodmodifiers  gdcm::LookupTable::GetLUTDescriptor " /** void
gdcm::LookupTable::GetLUTDescriptor(LookupTableType type, unsigned
short &length, unsigned short &subscript, unsigned short &bitsize)
const  */ public";

%csmethodmodifiers  gdcm::LookupTable::GetLUTLength " /** unsigned int
gdcm::LookupTable::GetLUTLength(LookupTableType type) const  */
public";

%csmethodmodifiers  gdcm::LookupTable::GetPointer " /** const unsigned
char* gdcm::LookupTable::GetPointer() const

return a raw pointer to the LUT

*/ public";

%csmethodmodifiers  gdcm::LookupTable::InitializeBlueLUT " /** void
gdcm::LookupTable::InitializeBlueLUT(unsigned short length, unsigned
short subscript, unsigned short bitsize)  */ public";

%csmethodmodifiers  gdcm::LookupTable::InitializeGreenLUT " /** void
gdcm::LookupTable::InitializeGreenLUT(unsigned short length, unsigned
short subscript, unsigned short bitsize)  */ public";

%csmethodmodifiers  gdcm::LookupTable::InitializeLUT " /** void
gdcm::LookupTable::InitializeLUT(LookupTableType type, unsigned short
length, unsigned short subscript, unsigned short bitsize)

Generic interface:.

*/ public";

%csmethodmodifiers  gdcm::LookupTable::InitializeRedLUT " /** void
gdcm::LookupTable::InitializeRedLUT(unsigned short length, unsigned
short subscript, unsigned short bitsize)

RED / GREEN / BLUE specific:.

*/ public";

%csmethodmodifiers  gdcm::LookupTable::LookupTable " /**
gdcm::LookupTable::LookupTable(LookupTable const &lut)  */ public";

%csmethodmodifiers  gdcm::LookupTable::LookupTable " /**
gdcm::LookupTable::LookupTable()  */ public";

%csmethodmodifiers  gdcm::LookupTable::Print " /** void
gdcm::LookupTable::Print(std::ostream &) const  */ public";

%csmethodmodifiers  gdcm::LookupTable::SetBlueLUT " /** void
gdcm::LookupTable::SetBlueLUT(const unsigned char *blue, unsigned int
length)  */ public";

%csmethodmodifiers  gdcm::LookupTable::SetGreenLUT " /** void
gdcm::LookupTable::SetGreenLUT(const unsigned char *green, unsigned
int length)  */ public";

%csmethodmodifiers  gdcm::LookupTable::SetLUT " /** virtual void
gdcm::LookupTable::SetLUT(LookupTableType type, const unsigned char
*array, unsigned int length)  */ public";

%csmethodmodifiers  gdcm::LookupTable::SetRedLUT " /** void
gdcm::LookupTable::SetRedLUT(const unsigned char *red, unsigned int
length)  */ public";

%csmethodmodifiers  gdcm::LookupTable::WriteBufferAsRGBA " /** bool
gdcm::LookupTable::WriteBufferAsRGBA(const unsigned char *rgba)

Write the LUT as RGBA.

*/ public";

%csmethodmodifiers  gdcm::LookupTable::~LookupTable " /**
gdcm::LookupTable::~LookupTable()  */ public";


// File: classstd_1_1map.xml
%typemap("csclassmodifiers") std::map " /** STL class.

*/ public class";


// File: classstd_1_1map_1_1const__iterator.xml
%typemap("csclassmodifiers") std::map::const_iterator " /** STL
iterator class.

*/ public class";


// File: classstd_1_1map_1_1const__reverse__iterator.xml
%typemap("csclassmodifiers") std::map::const_reverse_iterator " /**
STL iterator class.

*/ public class";


// File: classstd_1_1map_1_1iterator.xml
%typemap("csclassmodifiers") std::map::iterator " /** STL iterator
class.

*/ public class";


// File: classstd_1_1map_1_1reverse__iterator.xml
%typemap("csclassmodifiers") std::map::reverse_iterator " /** STL
iterator class.

*/ public class";


// File: classgdcm_1_1MD5.xml
%typemap("csclassmodifiers") gdcm::MD5 " /**C++ includes: gdcmMD5.h */
public class";

%csmethodmodifiers  gdcm::MD5::MD5 " /** gdcm::MD5::MD5()  */ public";

%csmethodmodifiers  gdcm::MD5::~MD5 " /** gdcm::MD5::~MD5()  */
public";


// File: classgdcm_1_1MediaStorage.xml
%typemap("csclassmodifiers") gdcm::MediaStorage " /**  MediaStorage.

FIXME There should not be any notion of Image and/or PDF at that point
Only the codec can answer yes I support this Media Storage or not...
For instance an ImageCodec will answer yes to most of them while a
PDFCodec will answer only for the Encapsulated PDF

C++ includes: gdcmMediaStorage.h */ public class";

%csmethodmodifiers  gdcm::MediaStorage::GetModality " /** const char*
gdcm::MediaStorage::GetModality() const  */ public";

%csmethodmodifiers  gdcm::MediaStorage::GetString " /** const char*
gdcm::MediaStorage::GetString() const  */ public";

%csmethodmodifiers  gdcm::MediaStorage::GuessFromModality " /** void
gdcm::MediaStorage::GuessFromModality(const char *modality, unsigned
int dimension=2)  */ public";

%csmethodmodifiers  gdcm::MediaStorage::IsUndefined " /** bool
gdcm::MediaStorage::IsUndefined() const  */ public";

%csmethodmodifiers  gdcm::MediaStorage::MediaStorage " /**
gdcm::MediaStorage::MediaStorage(MSType type=MS_END)  */ public";

%csmethodmodifiers  gdcm::MediaStorage::SetFromDataSet " /** bool
gdcm::MediaStorage::SetFromDataSet(DataSet const &ds)

Advanced user only (functions should be protected level...) Those
function are lower level than SetFromFile

*/ public";

%csmethodmodifiers  gdcm::MediaStorage::SetFromFile " /** bool
gdcm::MediaStorage::SetFromFile(File const &file)

Attempt to set the MediaStorage from a file: WARNING: When no
MediaStorage & Modality are found BUT a PixelData element is found
then MediaStorage is set to the default SecondaryCaptureImageStorage
(return value is false in this case)

*/ public";

%csmethodmodifiers  gdcm::MediaStorage::SetFromHeader " /** bool
gdcm::MediaStorage::SetFromHeader(FileMetaInformation const &fmi)  */
public";

%csmethodmodifiers  gdcm::MediaStorage::SetFromModality " /** bool
gdcm::MediaStorage::SetFromModality(DataSet const &ds)  */ public";


// File: classgdcm_1_1Module.xml
%typemap("csclassmodifiers") gdcm::Module " /** Class for representing
a Module.

bla Module: A set of Attributes within an Information Entity or
Normalized IOD which are logically related to each other.

See:   Dict

C++ includes: gdcmModule.h */ public class";

%csmethodmodifiers  gdcm::Module::AddModuleEntry " /** void
gdcm::Module::AddModuleEntry(const Tag &tag, const ModuleEntry
&module)  */ public";

%csmethodmodifiers  gdcm::Module::Begin " /** Iterator
gdcm::Module::Begin()  */ public";

%csmethodmodifiers  gdcm::Module::Begin " /** ConstIterator
gdcm::Module::Begin() const  */ public";

%csmethodmodifiers  gdcm::Module::Clear " /** void
gdcm::Module::Clear()  */ public";

%csmethodmodifiers  gdcm::Module::End " /** Iterator
gdcm::Module::End()  */ public";

%csmethodmodifiers  gdcm::Module::End " /** ConstIterator
gdcm::Module::End() const  */ public";

%csmethodmodifiers  gdcm::Module::FindModuleEntry " /** bool
gdcm::Module::FindModuleEntry(const Tag &tag) const  */ public";

%csmethodmodifiers  gdcm::Module::GetModuleEntry " /** const
ModuleEntry& gdcm::Module::GetModuleEntry(const Tag &tag) const  */
public";

%csmethodmodifiers  gdcm::Module::GetName " /** const char*
gdcm::Module::GetName() const  */ public";

%csmethodmodifiers  gdcm::Module::Module " /** gdcm::Module::Module()
*/ public";

%csmethodmodifiers  gdcm::Module::SetName " /** void
gdcm::Module::SetName(const char *name)  */ public";

%csmethodmodifiers  gdcm::Module::Verify " /** bool
gdcm::Module::Verify(const DataSet &ds, Usage const &usage) const  */
public";


// File: classgdcm_1_1ModuleEntry.xml
%typemap("csclassmodifiers") gdcm::ModuleEntry " /** Class for
representing a ModuleEntry.

bla

See:   DictEntry

C++ includes: gdcmModuleEntry.h */ public class";

%csmethodmodifiers  gdcm::ModuleEntry::GetDescription " /** const
Description& gdcm::ModuleEntry::GetDescription() const  */ public";

%csmethodmodifiers  gdcm::ModuleEntry::GetName " /** const char*
gdcm::ModuleEntry::GetName() const  */ public";

%csmethodmodifiers  gdcm::ModuleEntry::GetType " /** const Type&
gdcm::ModuleEntry::GetType() const  */ public";

%csmethodmodifiers  gdcm::ModuleEntry::ModuleEntry " /**
gdcm::ModuleEntry::ModuleEntry(const char *name=\"\", const char
*type=\"3\", const char *description=\"\")  */ public";

%csmethodmodifiers  gdcm::ModuleEntry::SetDescription " /** void
gdcm::ModuleEntry::SetDescription(const char *d)  */ public";

%csmethodmodifiers  gdcm::ModuleEntry::SetName " /** void
gdcm::ModuleEntry::SetName(const char *name)  */ public";

%csmethodmodifiers  gdcm::ModuleEntry::SetType " /** void
gdcm::ModuleEntry::SetType(const Type &type)  */ public";

%csmethodmodifiers  gdcm::ModuleEntry::~ModuleEntry " /** virtual
gdcm::ModuleEntry::~ModuleEntry()  */ public";


// File: classgdcm_1_1Modules.xml
%typemap("csclassmodifiers") gdcm::Modules " /** Class for
representing a Modules.

bla

See:   Module

C++ includes: gdcmModules.h */ public class";

%csmethodmodifiers  gdcm::Modules::AddModule " /** void
gdcm::Modules::AddModule(const char *ref, const Module &module)  */
public";

%csmethodmodifiers  gdcm::Modules::Clear " /** void
gdcm::Modules::Clear()  */ public";

%csmethodmodifiers  gdcm::Modules::GetModule " /** const Module&
gdcm::Modules::GetModule(const char *name) const  */ public";

%csmethodmodifiers  gdcm::Modules::IsEmpty " /** bool
gdcm::Modules::IsEmpty() const  */ public";

%csmethodmodifiers  gdcm::Modules::Modules " /**
gdcm::Modules::Modules()  */ public";


// File: classstd_1_1multimap.xml
%typemap("csclassmodifiers") std::multimap " /** STL class.

*/ public class";


// File: classstd_1_1multimap_1_1const__iterator.xml
%typemap("csclassmodifiers") std::multimap::const_iterator " /** STL
iterator class.

*/ public class";


// File: classstd_1_1multimap_1_1const__reverse__iterator.xml
%typemap("csclassmodifiers") std::multimap::const_reverse_iterator "
/** STL iterator class.

*/ public class";


// File: classstd_1_1multimap_1_1iterator.xml
%typemap("csclassmodifiers") std::multimap::iterator " /** STL
iterator class.

*/ public class";


// File: classstd_1_1multimap_1_1reverse__iterator.xml
%typemap("csclassmodifiers") std::multimap::reverse_iterator " /** STL
iterator class.

*/ public class";


// File: classstd_1_1multiset.xml
%typemap("csclassmodifiers") std::multiset " /** STL class.

*/ public class";


// File: classstd_1_1multiset_1_1const__iterator.xml
%typemap("csclassmodifiers") std::multiset::const_iterator " /** STL
iterator class.

*/ public class";


// File: classstd_1_1multiset_1_1const__reverse__iterator.xml
%typemap("csclassmodifiers") std::multiset::const_reverse_iterator "
/** STL iterator class.

*/ public class";


// File: classstd_1_1multiset_1_1iterator.xml
%typemap("csclassmodifiers") std::multiset::iterator " /** STL
iterator class.

*/ public class";


// File: classstd_1_1multiset_1_1reverse__iterator.xml
%typemap("csclassmodifiers") std::multiset::reverse_iterator " /** STL
iterator class.

*/ public class";


// File: classgdcm_1_1NestedModuleEntries.xml
%typemap("csclassmodifiers") gdcm::NestedModuleEntries " /** Class for
representing a NestedModuleEntries.

bla

See:   ModuleEntry

C++ includes: gdcmNestedModuleEntries.h */ public class";

%csmethodmodifiers  gdcm::NestedModuleEntries::AddModuleEntry " /**
void gdcm::NestedModuleEntries::AddModuleEntry(const ModuleEntry &me)
*/ public";

%csmethodmodifiers  gdcm::NestedModuleEntries::GetModuleEntry " /**
ModuleEntry& gdcm::NestedModuleEntries::GetModuleEntry(unsigned int
idx)  */ public";

%csmethodmodifiers  gdcm::NestedModuleEntries::GetModuleEntry " /**
const ModuleEntry& gdcm::NestedModuleEntries::GetModuleEntry(unsigned
int idx) const  */ public";

%csmethodmodifiers
gdcm::NestedModuleEntries::GetNumberOfModuleEntries " /** unsigned int
gdcm::NestedModuleEntries::GetNumberOfModuleEntries()  */ public";

%csmethodmodifiers  gdcm::NestedModuleEntries::NestedModuleEntries "
/** gdcm::NestedModuleEntries::NestedModuleEntries(const char
*name=\"\", const char *type=\"3\", const char *description=\"\")  */
public";


// File: classgdcm_1_1Object.xml
%typemap("csclassmodifiers") gdcm::Object " /**  Object.

main superclass for object that want to use SmartPointer invasive ref
counting system

See:   SmartPointer

C++ includes: gdcmObject.h */ public class";

%csmethodmodifiers  gdcm::Object::Object " /**
gdcm::Object::Object(const Object &)

Special requirement for copy/cstor, assigment operator.

*/ public";

%csmethodmodifiers  gdcm::Object::Object " /** gdcm::Object::Object()
*/ public";

%csmethodmodifiers  gdcm::Object::Print " /** virtual void
gdcm::Object::Print(std::ostream &) const  */ public";

%csmethodmodifiers  gdcm::Object::~Object " /** virtual
gdcm::Object::~Object()  */ public";


// File: classstd_1_1ofstream.xml
%typemap("csclassmodifiers") std::ofstream " /** STL class.

*/ public class";


// File: classgdcm_1_1Orientation.xml
%typemap("csclassmodifiers") gdcm::Orientation " /** class to handle
Orientation

C++ includes: gdcmOrientation.h */ public class";

%csmethodmodifiers  gdcm::Orientation::Orientation " /**
gdcm::Orientation::Orientation()  */ public";

%csmethodmodifiers  gdcm::Orientation::Print " /** void
gdcm::Orientation::Print(std::ostream &) const

Print.

*/ public";

%csmethodmodifiers  gdcm::Orientation::~Orientation " /**
gdcm::Orientation::~Orientation()  */ public";


// File: classstd_1_1ostream.xml
%typemap("csclassmodifiers") std::ostream " /** STL class.

*/ public class";


// File: classstd_1_1ostringstream.xml
%typemap("csclassmodifiers") std::ostringstream " /** STL class.

*/ public class";


// File: classstd_1_1out__of__range.xml
%typemap("csclassmodifiers") std::out_of_range " /** STL class.

*/ public class";


// File: classstd_1_1overflow__error.xml
%typemap("csclassmodifiers") std::overflow_error " /** STL class.

*/ public class";


// File: classgdcm_1_1Overlay.xml
%typemap("csclassmodifiers") gdcm::Overlay " /**  Overlay class.

see AreOverlaysInPixelData Todo Is there actually any way to recognize
an overlay ? On images with multiple overlay I do not see any way to
differenciate them (other than the group tag). Example:

C++ includes: gdcmOverlay.h */ public class";

%csmethodmodifiers  gdcm::Overlay::Decode " /** void
gdcm::Overlay::Decode(std::istream &is, std::ostream &os)  */ public";

%csmethodmodifiers  gdcm::Overlay::Decompress " /** void
gdcm::Overlay::Decompress(std::ostream &os) const  */ public";

%csmethodmodifiers  gdcm::Overlay::GetBitPosition " /** unsigned short
gdcm::Overlay::GetBitPosition() const

return bit position

*/ public";

%csmethodmodifiers  gdcm::Overlay::GetBitsAllocated " /** unsigned
short gdcm::Overlay::GetBitsAllocated() const

return bits allocated

*/ public";

%csmethodmodifiers  gdcm::Overlay::GetBuffer " /** bool
gdcm::Overlay::GetBuffer(char *buffer) const  */ public";

%csmethodmodifiers  gdcm::Overlay::GetColumns " /** unsigned short
gdcm::Overlay::GetColumns() const

get columns

*/ public";

%csmethodmodifiers  gdcm::Overlay::GetDescription " /** const char*
gdcm::Overlay::GetDescription() const

get description

*/ public";

%csmethodmodifiers  gdcm::Overlay::GetGroup " /** unsigned short
gdcm::Overlay::GetGroup() const

Get Group number.

*/ public";

%csmethodmodifiers  gdcm::Overlay::GetOrigin " /** const signed short*
gdcm::Overlay::GetOrigin() const

get origin

*/ public";

%csmethodmodifiers  gdcm::Overlay::GetOverlayData " /** const
ByteValue& gdcm::Overlay::GetOverlayData() const  */ public";

%csmethodmodifiers  gdcm::Overlay::GetRows " /** unsigned short
gdcm::Overlay::GetRows() const

get rows

*/ public";

%csmethodmodifiers  gdcm::Overlay::GetType " /** const char*
gdcm::Overlay::GetType() const

get type

*/ public";

%csmethodmodifiers  gdcm::Overlay::GetUnpackBuffer " /** bool
gdcm::Overlay::GetUnpackBuffer(unsigned char *buffer) const  */
public";

%csmethodmodifiers  gdcm::Overlay::GrabOverlayFromPixelData " /** bool
gdcm::Overlay::GrabOverlayFromPixelData(DataSet const &ds)  */
public";

%csmethodmodifiers  gdcm::Overlay::IsEmpty " /** bool
gdcm::Overlay::IsEmpty() const  */ public";

%csmethodmodifiers  gdcm::Overlay::IsInPixelData " /** void
gdcm::Overlay::IsInPixelData(bool b)  */ public";

%csmethodmodifiers  gdcm::Overlay::IsInPixelData " /** bool
gdcm::Overlay::IsInPixelData() const  */ public";

%csmethodmodifiers  gdcm::Overlay::IsZero " /** bool
gdcm::Overlay::IsZero() const

return true if all bits are set to 0

*/ public";

%csmethodmodifiers  gdcm::Overlay::Overlay " /**
gdcm::Overlay::Overlay(Overlay const &ov)  */ public";

%csmethodmodifiers  gdcm::Overlay::Overlay " /**
gdcm::Overlay::Overlay()  */ public";

%csmethodmodifiers  gdcm::Overlay::Print " /** void
gdcm::Overlay::Print(std::ostream &) const

Print.

*/ public";

%csmethodmodifiers  gdcm::Overlay::SetBitPosition " /** void
gdcm::Overlay::SetBitPosition(unsigned short bitposition)

set bit position

*/ public";

%csmethodmodifiers  gdcm::Overlay::SetBitsAllocated " /** void
gdcm::Overlay::SetBitsAllocated(unsigned short bitsallocated)

set bits allocated

*/ public";

%csmethodmodifiers  gdcm::Overlay::SetColumns " /** void
gdcm::Overlay::SetColumns(unsigned short columns)

set columns

*/ public";

%csmethodmodifiers  gdcm::Overlay::SetDescription " /** void
gdcm::Overlay::SetDescription(const char *description)

set description

*/ public";

%csmethodmodifiers  gdcm::Overlay::SetFrameOrigin " /** void
gdcm::Overlay::SetFrameOrigin(unsigned short frameorigin)

set frame origin

*/ public";

%csmethodmodifiers  gdcm::Overlay::SetGroup " /** void
gdcm::Overlay::SetGroup(unsigned short group)

Set Group number.

*/ public";

%csmethodmodifiers  gdcm::Overlay::SetNumberOfFrames " /** void
gdcm::Overlay::SetNumberOfFrames(unsigned int numberofframes)

set number of frames

*/ public";

%csmethodmodifiers  gdcm::Overlay::SetOrigin " /** void
gdcm::Overlay::SetOrigin(const signed short *origin)

set origin

*/ public";

%csmethodmodifiers  gdcm::Overlay::SetOverlay " /** void
gdcm::Overlay::SetOverlay(const char *array, unsigned int length)

set overlay from byte array + length

*/ public";

%csmethodmodifiers  gdcm::Overlay::SetRows " /** void
gdcm::Overlay::SetRows(unsigned short rows)

set rows

*/ public";

%csmethodmodifiers  gdcm::Overlay::SetType " /** void
gdcm::Overlay::SetType(const char *type)

set type

*/ public";

%csmethodmodifiers  gdcm::Overlay::Update " /** void
gdcm::Overlay::Update(const DataElement &de)

Update overlay from data element de:.

*/ public";

%csmethodmodifiers  gdcm::Overlay::~Overlay " /**
gdcm::Overlay::~Overlay()  */ public";


// File: classgdcm_1_1ParseException.xml
%typemap("csclassmodifiers") gdcm::ParseException " /** Standard
exception handling object.

C++ includes: gdcmParseException.h */ public class";

%csmethodmodifiers  gdcm::ParseException::GetLastElement " /** const
DataElement& gdcm::ParseException::GetLastElement() const  */ public";

%csmethodmodifiers  gdcm::ParseException::ParseException " /**
gdcm::ParseException::ParseException()  */ public";

%csmethodmodifiers  gdcm::ParseException::SetLastElement " /** void
gdcm::ParseException::SetLastElement(DataElement &de)

Equivalence operator.

*/ public";

%csmethodmodifiers  gdcm::ParseException::~ParseException " /**
virtual gdcm::ParseException::~ParseException()  throw () */ public";


// File: classgdcm_1_1Parser.xml
%typemap("csclassmodifiers") gdcm::Parser " /**  Parser ala XML_Parser
from expat (SAX).

Detailled description here Simple API for DICOM

C++ includes: gdcmParser.h */ public class";

%csmethodmodifiers  gdcm::Parser::GetCurrentByteIndex " /** unsigned
long gdcm::Parser::GetCurrentByteIndex() const  */ public";

%csmethodmodifiers  gdcm::Parser::GetErrorCode " /** ErrorType
gdcm::Parser::GetErrorCode() const  */ public";

%csmethodmodifiers  gdcm::Parser::GetUserData " /** void*
gdcm::Parser::GetUserData() const  */ public";

%csmethodmodifiers  gdcm::Parser::Parse " /** bool
gdcm::Parser::Parse(const char *s, int len, bool isFinal)  */ public";

%csmethodmodifiers  gdcm::Parser::Parser " /** gdcm::Parser::Parser()
*/ public";

%csmethodmodifiers  gdcm::Parser::SetElementHandler " /** void
gdcm::Parser::SetElementHandler(StartElementHandler start,
EndElementHandler end)  */ public";

%csmethodmodifiers  gdcm::Parser::SetUserData " /** void
gdcm::Parser::SetUserData(void *userData)  */ public";

%csmethodmodifiers  gdcm::Parser::~Parser " /**
gdcm::Parser::~Parser()  */ public";


// File: classgdcm_1_1Patient.xml
%typemap("csclassmodifiers") gdcm::Patient " /** See PS 3.3 - 2007
DICOM MODEL OF THE REAL-WORLD, p 54.

C++ includes: gdcmPatient.h */ public class";

%csmethodmodifiers  gdcm::Patient::Patient " /**
gdcm::Patient::Patient()  */ public";


// File: classgdcm_1_1PDBElement.xml
%typemap("csclassmodifiers") gdcm::PDBElement " /** Class to represent
a PDB Element.

C++ includes: gdcmPDBElement.h */ public class";

%csmethodmodifiers  gdcm::PDBElement::GetName " /** const char*
gdcm::PDBElement::GetName() const

Set/Get Name.

*/ public";

%csmethodmodifiers  gdcm::PDBElement::GetValue " /** const char*
gdcm::PDBElement::GetValue() const

Set/Get Value.

*/ public";

%csmethodmodifiers  gdcm::PDBElement::PDBElement " /**
gdcm::PDBElement::PDBElement()  */ public";

%csmethodmodifiers  gdcm::PDBElement::SetName " /** void
gdcm::PDBElement::SetName(const char *name)  */ public";

%csmethodmodifiers  gdcm::PDBElement::SetValue " /** void
gdcm::PDBElement::SetValue(const char *value)  */ public";


// File: classgdcm_1_1PDBHeader.xml
%typemap("csclassmodifiers") gdcm::PDBHeader " /** Class for
PDBHeader.

GEMS MR Image have an Attribute (0025,1b,GEMS_SERS_01) which store the
Acquisition parameter of the MR Image. It is compressed and can
therefore not be used as is. This class de- encapsulated the Protocol
Data Block and allow users to query element by name.

WARNING:  : Everything you do with this code is at your own risk,
since decoding process was not written from specification documents.
: the API of this class might change.

C++ includes: gdcmPDBHeader.h */ public class";

%csmethodmodifiers  gdcm::PDBHeader::FindPDBElementByName " /** bool
gdcm::PDBHeader::FindPDBElementByName(const char *name)

Return true if the PDB element matching name is found or not.

*/ public";

%csmethodmodifiers  gdcm::PDBHeader::GetPDBElementByName " /** const
PDBElement& gdcm::PDBHeader::GetPDBElementByName(const char *name)

Lookup in the PDB header if a PDB element match the name 'name':
WARNING:  Case Sensitive

*/ public";

%csmethodmodifiers  gdcm::PDBHeader::LoadFromDataElement " /** bool
gdcm::PDBHeader::LoadFromDataElement(DataElement const &de)

Load the PDB Header from a DataElement of a DataSet.

*/ public";

%csmethodmodifiers  gdcm::PDBHeader::PDBHeader " /**
gdcm::PDBHeader::PDBHeader()  */ public";

%csmethodmodifiers  gdcm::PDBHeader::Print " /** void
gdcm::PDBHeader::Print(std::ostream &os) const

Print.

*/ public";

%csmethodmodifiers  gdcm::PDBHeader::~PDBHeader " /**
gdcm::PDBHeader::~PDBHeader()  */ public";


// File: classgdcm_1_1PDFCodec.xml
%typemap("csclassmodifiers") gdcm::PDFCodec " /**  PDFCodec class.

C++ includes: gdcmPDFCodec.h */ public class";

%csmethodmodifiers  gdcm::PDFCodec::CanCode " /** bool
gdcm::PDFCodec::CanCode(TransferSyntax const &) const

Return whether this coder support this transfer syntax (can code it).

*/ public";

%csmethodmodifiers  gdcm::PDFCodec::CanDecode " /** bool
gdcm::PDFCodec::CanDecode(TransferSyntax const &) const

Return whether this decoder support this transfer syntax (can decode
it).

*/ public";

%csmethodmodifiers  gdcm::PDFCodec::Decode " /** bool
gdcm::PDFCodec::Decode(DataElement const &is, DataElement &os)

Decode.

*/ public";

%csmethodmodifiers  gdcm::PDFCodec::PDFCodec " /**
gdcm::PDFCodec::PDFCodec()  */ public";

%csmethodmodifiers  gdcm::PDFCodec::~PDFCodec " /**
gdcm::PDFCodec::~PDFCodec()  */ public";


// File: classgdcm_1_1PersonName.xml
%typemap("csclassmodifiers") gdcm::PersonName " /**  PersonName class.

C++ includes: gdcmPersonName.h */ public class";

%csmethodmodifiers  gdcm::PersonName::GetMaxLength " /** unsigned int
gdcm::PersonName::GetMaxLength() const  */ public";

%csmethodmodifiers  gdcm::PersonName::GetNumberOfComponents " /**
unsigned int gdcm::PersonName::GetNumberOfComponents() const  */
public";

%csmethodmodifiers  gdcm::PersonName::Print " /** void
gdcm::PersonName::Print(std::ostream &os) const  */ public";

%csmethodmodifiers  gdcm::PersonName::SetBlob " /** void
gdcm::PersonName::SetBlob(const std::vector< char > &v)  */ public";

%csmethodmodifiers  gdcm::PersonName::SetComponents " /** void
gdcm::PersonName::SetComponents(const char *components[])  */ public";

%csmethodmodifiers  gdcm::PersonName::SetComponents " /** void
gdcm::PersonName::SetComponents(const char *comp1=\"\", const char
*comp2=\"\", const char *comp3=\"\", const char *comp4=\"\", const
char *comp5=\"\")  */ public";


// File: classgdcm_1_1PhotometricInterpretation.xml
%typemap("csclassmodifiers") gdcm::PhotometricInterpretation " /**
Class to represent an PhotometricInterpretation.

C++ includes: gdcmPhotometricInterpretation.h */ public class";

%csmethodmodifiers
gdcm::PhotometricInterpretation::GetSamplesPerPixel " /** unsigned
short gdcm::PhotometricInterpretation::GetSamplesPerPixel() const

return the value for Sample Per Pixel associated with a particular
Photometric Interpretation

*/ public";

%csmethodmodifiers  gdcm::PhotometricInterpretation::GetString " /**
const char* gdcm::PhotometricInterpretation::GetString() const  */
public";

%csmethodmodifiers  gdcm::PhotometricInterpretation::IsLossless " /**
bool gdcm::PhotometricInterpretation::IsLossless() const  */ public";

%csmethodmodifiers  gdcm::PhotometricInterpretation::IsLossy " /**
bool gdcm::PhotometricInterpretation::IsLossy() const  */ public";

%csmethodmodifiers  gdcm::PhotometricInterpretation::IsSameColorSpace
" /** bool
gdcm::PhotometricInterpretation::IsSameColorSpace(PhotometricInterpretation
const &pi) const  */ public";

%csmethodmodifiers
gdcm::PhotometricInterpretation::PhotometricInterpretation " /**
gdcm::PhotometricInterpretation::PhotometricInterpretation(PIType
pi=UNKNOW)  */ public";


// File: classgdcm_1_1PixelFormat.xml
%typemap("csclassmodifiers") gdcm::PixelFormat " /**  PixelFormat.

By default the Pixel Type will be instanciated with the following
parameters: SamplesPerPixel : 1

BitsAllocated : 8

BitsStored : 8

HighBit : 7

PixelRepresentation : 0

C++ includes: gdcmPixelFormat.h */ public class";

%csmethodmodifiers  gdcm::PixelFormat::GetBitsAllocated " /** unsigned
short gdcm::PixelFormat::GetBitsAllocated() const

BitsAllocated.

*/ public";

%csmethodmodifiers  gdcm::PixelFormat::GetBitsStored " /** unsigned
short gdcm::PixelFormat::GetBitsStored() const

BitsStored.

*/ public";

%csmethodmodifiers  gdcm::PixelFormat::GetHighBit " /** unsigned short
gdcm::PixelFormat::GetHighBit() const

HighBit.

*/ public";

%csmethodmodifiers  gdcm::PixelFormat::GetMax " /** int64_t
gdcm::PixelFormat::GetMax() const

return the max possible of the pixel

*/ public";

%csmethodmodifiers  gdcm::PixelFormat::GetMin " /** int64_t
gdcm::PixelFormat::GetMin() const

return the min possible of the pixel

*/ public";

%csmethodmodifiers  gdcm::PixelFormat::GetPixelRepresentation " /**
unsigned short gdcm::PixelFormat::GetPixelRepresentation() const

PixelRepresentation.

*/ public";

%csmethodmodifiers  gdcm::PixelFormat::GetPixelSize " /** uint8_t
gdcm::PixelFormat::GetPixelSize() const

return the size of the pixel This is the number of words it would take
to store one pixel WARNING:  the return value takes into account the
SamplesPerPixel

in the rare case when BitsAllocated == 12, the function assume word
padding and value returned will be identical as if BitsAllocated == 16

*/ public";

%csmethodmodifiers  gdcm::PixelFormat::GetSamplesPerPixel " /**
unsigned short gdcm::PixelFormat::GetSamplesPerPixel() const

Samples Per Pixel.

*/ public";

%csmethodmodifiers  gdcm::PixelFormat::GetScalarType " /** ScalarType
gdcm::PixelFormat::GetScalarType() const

ScalarType does not take into account the sample per pixel.

*/ public";

%csmethodmodifiers  gdcm::PixelFormat::GetScalarTypeAsString " /**
const char* gdcm::PixelFormat::GetScalarTypeAsString() const  */
public";

%csmethodmodifiers  gdcm::PixelFormat::PixelFormat " /**
gdcm::PixelFormat::PixelFormat(ScalarType st)  */ public";

%csmethodmodifiers  gdcm::PixelFormat::PixelFormat " /**
gdcm::PixelFormat::PixelFormat(unsigned short samplesperpixel=1,
unsigned short bitsallocated=8, unsigned short bitsstored=8, unsigned
short highbit=7, unsigned short pixelrepresentation=0)  */ public";

%csmethodmodifiers  gdcm::PixelFormat::Print " /** void
gdcm::PixelFormat::Print(std::ostream &os) const

Print.

*/ public";

%csmethodmodifiers  gdcm::PixelFormat::SetBitsAllocated " /** void
gdcm::PixelFormat::SetBitsAllocated(unsigned short ba)  */ public";

%csmethodmodifiers  gdcm::PixelFormat::SetBitsStored " /** void
gdcm::PixelFormat::SetBitsStored(unsigned short bs)  */ public";

%csmethodmodifiers  gdcm::PixelFormat::SetHighBit " /** void
gdcm::PixelFormat::SetHighBit(unsigned short hb)  */ public";

%csmethodmodifiers  gdcm::PixelFormat::SetPixelRepresentation " /**
void gdcm::PixelFormat::SetPixelRepresentation(unsigned short pr)  */
public";

%csmethodmodifiers  gdcm::PixelFormat::SetSamplesPerPixel " /** void
gdcm::PixelFormat::SetSamplesPerPixel(unsigned short spp)  */ public";

%csmethodmodifiers  gdcm::PixelFormat::SetScalarType " /** void
gdcm::PixelFormat::SetScalarType(ScalarType st)  */ public";

%csmethodmodifiers  gdcm::PixelFormat::~PixelFormat " /**
gdcm::PixelFormat::~PixelFormat()  */ public";


// File: classgdcm_1_1Pixmap.xml
%typemap("csclassmodifiers") gdcm::Pixmap " /**  Pixmap class A bitmap
based image. Used as parent for both IconImage and the main Pixel Data
Image It does not contains any World Space information (IPP, IOP).

C++ includes: gdcmPixmap.h */ public class";

%csmethodmodifiers  gdcm::Pixmap::AreOverlaysInPixelData " /** bool
gdcm::Pixmap::AreOverlaysInPixelData() const

returns if Overlays are stored in the unused bit of the pixel data:

*/ public";

%csmethodmodifiers  gdcm::Pixmap::GetCurve " /** const Curve&
gdcm::Pixmap::GetCurve(unsigned int i=0) const  */ public";

%csmethodmodifiers  gdcm::Pixmap::GetCurve " /** Curve&
gdcm::Pixmap::GetCurve(unsigned int i=0)

Curve: group 50xx.

*/ public";

%csmethodmodifiers  gdcm::Pixmap::GetIconImage " /** IconImage&
gdcm::Pixmap::GetIconImage()  */ public";

%csmethodmodifiers  gdcm::Pixmap::GetIconImage " /** const IconImage&
gdcm::Pixmap::GetIconImage() const

Set/Get Icon Image.

*/ public";

%csmethodmodifiers  gdcm::Pixmap::GetNumberOfCurves " /** unsigned int
gdcm::Pixmap::GetNumberOfCurves() const  */ public";

%csmethodmodifiers  gdcm::Pixmap::GetNumberOfOverlays " /** unsigned
int gdcm::Pixmap::GetNumberOfOverlays() const  */ public";

%csmethodmodifiers  gdcm::Pixmap::GetOverlay " /** const Overlay&
gdcm::Pixmap::GetOverlay(unsigned int i=0) const  */ public";

%csmethodmodifiers  gdcm::Pixmap::GetOverlay " /** Overlay&
gdcm::Pixmap::GetOverlay(unsigned int i=0)

Overlay: group 60xx.

*/ public";

%csmethodmodifiers  gdcm::Pixmap::Pixmap " /** gdcm::Pixmap::Pixmap()
*/ public";

%csmethodmodifiers  gdcm::Pixmap::Print " /** void
gdcm::Pixmap::Print(std::ostream &) const  */ public";

%csmethodmodifiers  gdcm::Pixmap::SetNumberOfCurves " /** void
gdcm::Pixmap::SetNumberOfCurves(unsigned int n)  */ public";

%csmethodmodifiers  gdcm::Pixmap::SetNumberOfOverlays " /** void
gdcm::Pixmap::SetNumberOfOverlays(unsigned int n)  */ public";

%csmethodmodifiers  gdcm::Pixmap::~Pixmap " /**
gdcm::Pixmap::~Pixmap()  */ public";


// File: classgdcm_1_1PixmapReader.xml
%typemap("csclassmodifiers") gdcm::PixmapReader " /**  PixmapReader.

its role is to convert the DICOM DataSet into a gdcm::Pixmap
representation By default it is also loading the lookup table and
overlay when found as they impact the rendering or the image  See PS
3.3-2008, Table C.7-11b IMAGE PIXEL MACRO ATTRIBUTES for the list of
attribute that belong to what gdcm calls a 'Pixmap'

C++ includes: gdcmPixmapReader.h */ public class";

%csmethodmodifiers  gdcm::PixmapReader::GetPixmap " /** Pixmap&
gdcm::PixmapReader::GetPixmap()  */ public";

%csmethodmodifiers  gdcm::PixmapReader::GetPixmap " /** const Pixmap&
gdcm::PixmapReader::GetPixmap() const

Return the read image.

*/ public";

%csmethodmodifiers  gdcm::PixmapReader::PixmapReader " /**
gdcm::PixmapReader::PixmapReader()  */ public";

%csmethodmodifiers  gdcm::PixmapReader::Read " /** bool
gdcm::PixmapReader::Read()

Read the DICOM image. There are two reason for failure: 1. The input
filename is not DICOM 2. The input DICOM file does not contains an
Pixmap.

*/ public";

%csmethodmodifiers  gdcm::PixmapReader::~PixmapReader " /**
gdcm::PixmapReader::~PixmapReader()  */ public";


// File: classgdcm_1_1PixmapToPixmapFilter.xml
%typemap("csclassmodifiers") gdcm::PixmapToPixmapFilter " /**
PixmapToPixmapFilter class Super class for all filter taking an image
and producing an output image.

C++ includes: gdcmPixmapToPixmapFilter.h */ public class";

%csmethodmodifiers  gdcm::PixmapToPixmapFilter::GetOutput " /** const
Pixmap& gdcm::PixmapToPixmapFilter::GetOutput() const

Get Output image.

*/ public";

%csmethodmodifiers  gdcm::PixmapToPixmapFilter::PixmapToPixmapFilter "
/** gdcm::PixmapToPixmapFilter::PixmapToPixmapFilter()  */ public";

%csmethodmodifiers  gdcm::PixmapToPixmapFilter::SetInput " /** void
gdcm::PixmapToPixmapFilter::SetInput(const Pixmap &image)

Set input image.

*/ public";

%csmethodmodifiers  gdcm::PixmapToPixmapFilter::~PixmapToPixmapFilter
" /** gdcm::PixmapToPixmapFilter::~PixmapToPixmapFilter()  */ public";


// File: classgdcm_1_1PixmapWriter.xml
%typemap("csclassmodifiers") gdcm::PixmapWriter " /**  PixmapWriter.

C++ includes: gdcmPixmapWriter.h */ public class";

%csmethodmodifiers  gdcm::PixmapWriter::GetImage " /** virtual Pixmap&
gdcm::PixmapWriter::GetImage()  */ public";

%csmethodmodifiers  gdcm::PixmapWriter::GetImage " /** virtual const
Pixmap& gdcm::PixmapWriter::GetImage() const

Set/Get Pixmap to be written It will overwrite anything Pixmap infos
found in DataSet (see parent class to see how to pass dataset)

*/ public";

%csmethodmodifiers  gdcm::PixmapWriter::GetPixmap " /** Pixmap&
gdcm::PixmapWriter::GetPixmap()  */ public";

%csmethodmodifiers  gdcm::PixmapWriter::GetPixmap " /** const Pixmap&
gdcm::PixmapWriter::GetPixmap() const  */ public";

%csmethodmodifiers  gdcm::PixmapWriter::PixmapWriter " /**
gdcm::PixmapWriter::PixmapWriter()  */ public";

%csmethodmodifiers  gdcm::PixmapWriter::SetImage " /** virtual void
gdcm::PixmapWriter::SetImage(Pixmap const &img)  */ public";

%csmethodmodifiers  gdcm::PixmapWriter::SetPixmap " /** void
gdcm::PixmapWriter::SetPixmap(Pixmap const &img)  */ public";

%csmethodmodifiers  gdcm::PixmapWriter::Write " /** bool
gdcm::PixmapWriter::Write()

Write.

*/ public";

%csmethodmodifiers  gdcm::PixmapWriter::~PixmapWriter " /**
gdcm::PixmapWriter::~PixmapWriter()  */ public";


// File: classgdcm_1_1PKCS7.xml
%typemap("csclassmodifiers") gdcm::PKCS7 " /**C++ includes:
gdcmPKCS7.h */ public class";

%csmethodmodifiers  gdcm::PKCS7::Decrypt " /** bool
gdcm::PKCS7::Decrypt(char *output, size_t &outlen, const char *array,
size_t len) const  */ public";

%csmethodmodifiers  gdcm::PKCS7::Encrypt " /** bool
gdcm::PKCS7::Encrypt(char *output, size_t &outlen, const char *array,
size_t len) const  */ public";

%csmethodmodifiers  gdcm::PKCS7::GetCertificate " /** const X509*
gdcm::PKCS7::GetCertificate() const  */ public";

%csmethodmodifiers  gdcm::PKCS7::GetCipherType " /** CipherTypes
gdcm::PKCS7::GetCipherType() const  */ public";

%csmethodmodifiers  gdcm::PKCS7::PKCS7 " /** gdcm::PKCS7::PKCS7()  */
public";

%csmethodmodifiers  gdcm::PKCS7::SetCertificate " /** void
gdcm::PKCS7::SetCertificate(X509 *cert)  */ public";

%csmethodmodifiers  gdcm::PKCS7::SetCipherType " /** void
gdcm::PKCS7::SetCipherType(CipherTypes type)  */ public";

%csmethodmodifiers  gdcm::PKCS7::~PKCS7 " /** gdcm::PKCS7::~PKCS7()
*/ public";


// File: classgdcm_1_1PNMCodec.xml
%typemap("csclassmodifiers") gdcm::PNMCodec " /** Class to do PNM PNM
is the Portable anymap file format. The main web page can be found
at:http://netpbm.sourceforge.net/.

Only support P5 & P6 PNM file (binary grayscale and binary rgb)

C++ includes: gdcmPNMCodec.h */ public class";

%csmethodmodifiers  gdcm::PNMCodec::CanCode " /** bool
gdcm::PNMCodec::CanCode(TransferSyntax const &ts) const

Return whether this coder support this transfer syntax (can code it).

*/ public";

%csmethodmodifiers  gdcm::PNMCodec::CanDecode " /** bool
gdcm::PNMCodec::CanDecode(TransferSyntax const &ts) const

Return whether this decoder support this transfer syntax (can decode
it).

*/ public";

%csmethodmodifiers  gdcm::PNMCodec::GetBufferLength " /** unsigned
long gdcm::PNMCodec::GetBufferLength() const  */ public";

%csmethodmodifiers  gdcm::PNMCodec::GetHeaderInfo " /** bool
gdcm::PNMCodec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
*/ public";

%csmethodmodifiers  gdcm::PNMCodec::PNMCodec " /**
gdcm::PNMCodec::PNMCodec()  */ public";

%csmethodmodifiers  gdcm::PNMCodec::Read " /** bool
gdcm::PNMCodec::Read(const char *filename, DataElement &out) const  */
public";

%csmethodmodifiers  gdcm::PNMCodec::SetBufferLength " /** void
gdcm::PNMCodec::SetBufferLength(unsigned long l)  */ public";

%csmethodmodifiers  gdcm::PNMCodec::Write " /** bool
gdcm::PNMCodec::Write(const char *filename, const DataElement &out)
const  */ public";

%csmethodmodifiers  gdcm::PNMCodec::~PNMCodec " /**
gdcm::PNMCodec::~PNMCodec()  */ public";


// File: classgdcm_1_1Preamble.xml
%typemap("csclassmodifiers") gdcm::Preamble " /** DICOM Preamble (Part
10).

C++ includes: gdcmPreamble.h */ public class";

%csmethodmodifiers  gdcm::Preamble::Clear " /** void
gdcm::Preamble::Clear()  */ public";

%csmethodmodifiers  gdcm::Preamble::Create " /** void
gdcm::Preamble::Create()  */ public";

%csmethodmodifiers  gdcm::Preamble::GetInternal " /** const char*
gdcm::Preamble::GetInternal() const  */ public";

%csmethodmodifiers  gdcm::Preamble::IsEmpty " /** bool
gdcm::Preamble::IsEmpty() const  */ public";

%csmethodmodifiers  gdcm::Preamble::Preamble " /**
gdcm::Preamble::Preamble(Preamble const &preamble)  */ public";

%csmethodmodifiers  gdcm::Preamble::Preamble " /**
gdcm::Preamble::Preamble()  */ public";

%csmethodmodifiers  gdcm::Preamble::Print " /** void
gdcm::Preamble::Print(std::ostream &os) const  */ public";

%csmethodmodifiers  gdcm::Preamble::Read " /** std::istream&
gdcm::Preamble::Read(std::istream &is)  */ public";

%csmethodmodifiers  gdcm::Preamble::Remove " /** void
gdcm::Preamble::Remove()  */ public";

%csmethodmodifiers  gdcm::Preamble::Valid " /** void
gdcm::Preamble::Valid()  */ public";

%csmethodmodifiers  gdcm::Preamble::Write " /** std::ostream const&
gdcm::Preamble::Write(std::ostream &os) const  */ public";

%csmethodmodifiers  gdcm::Preamble::~Preamble " /**
gdcm::Preamble::~Preamble()  */ public";


// File: classgdcm_1_1Printer.xml
%typemap("csclassmodifiers") gdcm::Printer " /**  Printer class.

C++ includes: gdcmPrinter.h */ public class";

%csmethodmodifiers  gdcm::Printer::GetPrintStyle " /** PrintStyles
gdcm::Printer::GetPrintStyle() const  */ public";

%csmethodmodifiers  gdcm::Printer::Print " /** void
gdcm::Printer::Print(std::ostream &os)  */ public";

%csmethodmodifiers  gdcm::Printer::Printer " /**
gdcm::Printer::Printer()  */ public";

%csmethodmodifiers  gdcm::Printer::SetColor " /** void
gdcm::Printer::SetColor(bool c)  */ public";

%csmethodmodifiers  gdcm::Printer::SetFile " /** void
gdcm::Printer::SetFile(File const &f)  */ public";

%csmethodmodifiers  gdcm::Printer::SetStyle " /** void
gdcm::Printer::SetStyle(PrintStyles ps)  */ public";

%csmethodmodifiers  gdcm::Printer::~Printer " /**
gdcm::Printer::~Printer()  */ public";


// File: classstd_1_1priority__queue.xml
%typemap("csclassmodifiers") std::priority_queue " /** STL class.

*/ public class";


// File: classgdcm_1_1PrivateDict.xml
%typemap("csclassmodifiers") gdcm::PrivateDict " /** Private Dict.

C++ includes: gdcmDict.h */ public class";

%csmethodmodifiers  gdcm::PrivateDict::AddDictEntry " /** void
gdcm::PrivateDict::AddDictEntry(const PrivateTag &tag, const DictEntry
&de)  */ public";

%csmethodmodifiers  gdcm::PrivateDict::GetDictEntry " /** const
DictEntry& gdcm::PrivateDict::GetDictEntry(const PrivateTag &tag)
const  */ public";

%csmethodmodifiers  gdcm::PrivateDict::IsEmpty " /** bool
gdcm::PrivateDict::IsEmpty() const  */ public";

%csmethodmodifiers  gdcm::PrivateDict::PrintXML " /** void
gdcm::PrivateDict::PrintXML() const  */ public";

%csmethodmodifiers  gdcm::PrivateDict::PrivateDict " /**
gdcm::PrivateDict::PrivateDict()  */ public";

%csmethodmodifiers  gdcm::PrivateDict::~PrivateDict " /**
gdcm::PrivateDict::~PrivateDict()  */ public";


// File: classgdcm_1_1PrivateTag.xml
%typemap("csclassmodifiers") gdcm::PrivateTag " /** Class to represent
a Private DICOM Data Element ( Attribute) Tag (Group, Element, Owner).

private tag have element value in: [0x10,0xff], for instance
0x0009,0x0000 is NOT a private tag

C++ includes: gdcmPrivateTag.h */ public class";

%csmethodmodifiers  gdcm::PrivateTag::GetOwner " /** const char*
gdcm::PrivateTag::GetOwner() const  */ public";

%csmethodmodifiers  gdcm::PrivateTag::PrivateTag " /**
gdcm::PrivateTag::PrivateTag(uint16_t group=0, uint16_t element=0,
const char *owner=\"\")  */ public";

%csmethodmodifiers  gdcm::PrivateTag::SetOwner " /** void
gdcm::PrivateTag::SetOwner(const char *owner)  */ public";


// File: classgdcm_1_1PVRGCodec.xml
%typemap("csclassmodifiers") gdcm::PVRGCodec " /**  PVRGCodec.

pvrg is a broken implementation of the JPEG standard. It is known to
have a bug in the 16bits lossless implementation of the standard.  In
an ideal world, you should not need this codec at all. But to support
some broken file such as:

PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm

we have to...

C++ includes: gdcmPVRGCodec.h */ public class";

%csmethodmodifiers  gdcm::PVRGCodec::CanCode " /** bool
gdcm::PVRGCodec::CanCode(TransferSyntax const &ts) const

Return whether this coder support this transfer syntax (can code it).

*/ public";

%csmethodmodifiers  gdcm::PVRGCodec::CanDecode " /** bool
gdcm::PVRGCodec::CanDecode(TransferSyntax const &ts) const

Return whether this decoder support this transfer syntax (can decode
it).

*/ public";

%csmethodmodifiers  gdcm::PVRGCodec::Code " /** bool
gdcm::PVRGCodec::Code(DataElement const &in, DataElement &out)

Code.

*/ public";

%csmethodmodifiers  gdcm::PVRGCodec::Decode " /** bool
gdcm::PVRGCodec::Decode(DataElement const &is, DataElement &os)

Decode.

*/ public";

%csmethodmodifiers  gdcm::PVRGCodec::PVRGCodec " /**
gdcm::PVRGCodec::PVRGCodec()  */ public";

%csmethodmodifiers  gdcm::PVRGCodec::~PVRGCodec " /**
gdcm::PVRGCodec::~PVRGCodec()  */ public";


// File: classgdcm_1_1PythonFilter.xml
%typemap("csclassmodifiers") gdcm::PythonFilter " /**  PythonFilter
PythonFilter is the class that make gdcm2.x looks more like gdcm1 and
transform the binary blob contained in a DataElement into a string,
typically this is a nice feature to have for wrapped language.

C++ includes: gdcmPythonFilter.h */ public class";

%csmethodmodifiers  gdcm::PythonFilter::GetFile " /** const File&
gdcm::PythonFilter::GetFile() const  */ public";

%csmethodmodifiers  gdcm::PythonFilter::GetFile " /** File&
gdcm::PythonFilter::GetFile()  */ public";

%csmethodmodifiers  gdcm::PythonFilter::PythonFilter " /**
gdcm::PythonFilter::PythonFilter()  */ public";

%csmethodmodifiers  gdcm::PythonFilter::SetDicts " /** void
gdcm::PythonFilter::SetDicts(const Dicts &dicts)  */ public";

%csmethodmodifiers  gdcm::PythonFilter::SetFile " /** void
gdcm::PythonFilter::SetFile(const File &f)  */ public";

%csmethodmodifiers  gdcm::PythonFilter::ToPyObject " /** PyObject*
gdcm::PythonFilter::ToPyObject(const Tag &t) const  */ public";

%csmethodmodifiers  gdcm::PythonFilter::UseDictAlways " /** void
gdcm::PythonFilter::UseDictAlways(bool use)  */ public";

%csmethodmodifiers  gdcm::PythonFilter::~PythonFilter " /**
gdcm::PythonFilter::~PythonFilter()  */ public";


// File: classstd_1_1queue.xml
%typemap("csclassmodifiers") std::queue " /** STL class.

*/ public class";


// File: classstd_1_1range__error.xml
%typemap("csclassmodifiers") std::range_error " /** STL class.

*/ public class";


// File: classgdcm_1_1RAWCodec.xml
%typemap("csclassmodifiers") gdcm::RAWCodec " /**  RAWCodec class.

C++ includes: gdcmRAWCodec.h */ public class";

%csmethodmodifiers  gdcm::RAWCodec::CanCode " /** bool
gdcm::RAWCodec::CanCode(TransferSyntax const &ts) const

Return whether this coder support this transfer syntax (can code it).

*/ public";

%csmethodmodifiers  gdcm::RAWCodec::CanDecode " /** bool
gdcm::RAWCodec::CanDecode(TransferSyntax const &ts) const

Return whether this decoder support this transfer syntax (can decode
it).

*/ public";

%csmethodmodifiers  gdcm::RAWCodec::Code " /** bool
gdcm::RAWCodec::Code(DataElement const &in, DataElement &out)

Code.

*/ public";

%csmethodmodifiers  gdcm::RAWCodec::Decode " /** bool
gdcm::RAWCodec::Decode(DataElement const &is, DataElement &os)

Decode.

*/ public";

%csmethodmodifiers  gdcm::RAWCodec::GetHeaderInfo " /** bool
gdcm::RAWCodec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
*/ public";

%csmethodmodifiers  gdcm::RAWCodec::RAWCodec " /**
gdcm::RAWCodec::RAWCodec()  */ public";

%csmethodmodifiers  gdcm::RAWCodec::~RAWCodec " /**
gdcm::RAWCodec::~RAWCodec()  */ public";


// File: classgdcm_1_1Reader.xml
%typemap("csclassmodifiers") gdcm::Reader " /**  Reader ala DOM
(Document Object Model).

This class is a non-validating reader, it will only performs well-
formedness check only, and to some extent catch known error (non well-
formed document).

Detailled description here

A DataSet DOES NOT contains group 0x0002

This is really a DataSet reader. This will not make sure the dataset
conform to any IOD at all. This is a completely different step. The
reasoning was that user could control the IOD there lib would handle
and thus we would not be able to read a DataSet if the IOD was not
found Instead we separate the reading from the validation.

NOTE: From GDCM1.x. Users will realize that one feature is missing
from this DOM implementation. In GDCM 1.x user used to be able to
control the size of the Value to be read. By default it was 0xfff. The
main author of GDCM2 thought this was too dangerous and harmful and
therefore this feature did not make it into GDCM2

WARNING: GDCM will not produce warning for unorder (non-alphabetical
order). See gdcm::Writer for more info

C++ includes: gdcmReader.h */ public class";

%csmethodmodifiers  gdcm::Reader::GetFile " /** File&
gdcm::Reader::GetFile()  */ public";

%csmethodmodifiers  gdcm::Reader::GetFile " /** const File&
gdcm::Reader::GetFile() const  */ public";

%csmethodmodifiers  gdcm::Reader::Read " /** virtual bool
gdcm::Reader::Read()  */ public";

%csmethodmodifiers  gdcm::Reader::Reader " /** gdcm::Reader::Reader()
*/ public";

%csmethodmodifiers  gdcm::Reader::ReadUpToTag " /** bool
gdcm::Reader::ReadUpToTag(const Tag &tag, std::set< Tag > const
&skiptags)  */ public";

%csmethodmodifiers  gdcm::Reader::SetFile " /** void
gdcm::Reader::SetFile(File &file)  */ public";

%csmethodmodifiers  gdcm::Reader::SetFileName " /** void
gdcm::Reader::SetFileName(const char *filename)  */ public";

%csmethodmodifiers  gdcm::Reader::SetStream " /** void
gdcm::Reader::SetStream(std::istream &input_stream)  */ public";

%csmethodmodifiers  gdcm::Reader::~Reader " /** virtual
gdcm::Reader::~Reader()  */ public";


// File: classgdcm_1_1Rescaler.xml
%typemap("csclassmodifiers") gdcm::Rescaler " /** Rescale class.

WARNING:  internally any time a floating point value is found either
in the Rescale Slope or the Rescale Intercept it is assumed that the
best matching output pixel type if FLOAT64 in previous implementation
it was FLOAT32. Because VR:DS is closer to a 64bits floating point
type FLOAT64 is thus a best matching pixel type for the floating point
transformation.

handle floating point transformation back and forth to integer
properly (no loss)

C++ includes: gdcmRescaler.h */ public class";

%csmethodmodifiers  gdcm::Rescaler::ComputeInterceptSlopePixelType "
/** PixelFormat::ScalarType
gdcm::Rescaler::ComputeInterceptSlopePixelType()

Compute the Pixel Format of the output data Used for direct
transformation

*/ public";

%csmethodmodifiers  gdcm::Rescaler::ComputePixelTypeFromMinMax " /**
PixelFormat gdcm::Rescaler::ComputePixelTypeFromMinMax()

Compute the Pixel Format of the output data Used for inverse
transformation

*/ public";

%csmethodmodifiers  gdcm::Rescaler::InverseRescale " /** bool
gdcm::Rescaler::InverseRescale(char *out, const char *in, size_t n)

Inverse transform.

*/ public";

%csmethodmodifiers  gdcm::Rescaler::Rescale " /** bool
gdcm::Rescaler::Rescale(char *out, const char *in, size_t n)

Direct transform.

*/ public";

%csmethodmodifiers  gdcm::Rescaler::Rescaler " /**
gdcm::Rescaler::Rescaler()  */ public";

%csmethodmodifiers  gdcm::Rescaler::SetIntercept " /** void
gdcm::Rescaler::SetIntercept(double i)

Set Intercept: used for both direct&inverse transformation.

*/ public";

%csmethodmodifiers  gdcm::Rescaler::SetMinMaxForPixelType " /** void
gdcm::Rescaler::SetMinMaxForPixelType(double min, double max)

Set target interval for output data. A best match will be computed (if
possible) Used for inverse transformation

*/ public";

%csmethodmodifiers  gdcm::Rescaler::SetPixelFormat " /** void
gdcm::Rescaler::SetPixelFormat(PixelFormat const &pf)

Set Pixel Format of input data.

*/ public";

%csmethodmodifiers  gdcm::Rescaler::SetSlope " /** void
gdcm::Rescaler::SetSlope(double s)

Set Slope: user for both direct&inverse transformation.

*/ public";

%csmethodmodifiers  gdcm::Rescaler::~Rescaler " /**
gdcm::Rescaler::~Rescaler()  */ public";


// File: classgdcm_1_1RLECodec.xml
%typemap("csclassmodifiers") gdcm::RLECodec " /** Class to do RLE.

ANSI X3.9 A.4.2 RLE Compression Annex G defines a RLE Compression
Transfer Syntax. This transfer Syntax is identified by the UID value
\"1.2.840.10008.1.2.5\". If the object allows multi-frame images in
the pixel data field, then each frame shall be encoded separately.
Each frame shall be encoded in one and only one Fragment (see PS
3.5.8.2).

C++ includes: gdcmRLECodec.h */ public class";

%csmethodmodifiers  gdcm::RLECodec::CanCode " /** bool
gdcm::RLECodec::CanCode(TransferSyntax const &ts) const

Return whether this coder support this transfer syntax (can code it).

*/ public";

%csmethodmodifiers  gdcm::RLECodec::CanDecode " /** bool
gdcm::RLECodec::CanDecode(TransferSyntax const &ts) const

Return whether this decoder support this transfer syntax (can decode
it).

*/ public";

%csmethodmodifiers  gdcm::RLECodec::Code " /** bool
gdcm::RLECodec::Code(DataElement const &in, DataElement &out)

Code.

*/ public";

%csmethodmodifiers  gdcm::RLECodec::Decode " /** bool
gdcm::RLECodec::Decode(DataElement const &is, DataElement &os)

Decode.

*/ public";

%csmethodmodifiers  gdcm::RLECodec::GetBufferLength " /** unsigned
long gdcm::RLECodec::GetBufferLength() const  */ public";

%csmethodmodifiers  gdcm::RLECodec::GetHeaderInfo " /** bool
gdcm::RLECodec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
*/ public";

%csmethodmodifiers  gdcm::RLECodec::RLECodec " /**
gdcm::RLECodec::RLECodec()  */ public";

%csmethodmodifiers  gdcm::RLECodec::SetBufferLength " /** void
gdcm::RLECodec::SetBufferLength(unsigned long l)  */ public";

%csmethodmodifiers  gdcm::RLECodec::SetLength " /** void
gdcm::RLECodec::SetLength(unsigned long l)  */ public";

%csmethodmodifiers  gdcm::RLECodec::~RLECodec " /**
gdcm::RLECodec::~RLECodec()  */ public";


// File: classgdcm_1_1RSA.xml
%typemap("csclassmodifiers") gdcm::RSA " /**C++ includes: gdcmRSA.h */
public class";

%csmethodmodifiers  gdcm::RSA::CheckPrivkey " /** int
gdcm::RSA::CheckPrivkey() const

Check a private RSA key.

Parameters:
-----------

ctx:   RSA context to be checked

0 if successful, or an POLARSSL_ERR_RSA_XXX error code

*/ public";

%csmethodmodifiers  gdcm::RSA::CheckPubkey " /** int
gdcm::RSA::CheckPubkey() const

Check a public RSA key.

Parameters:
-----------

ctx:   RSA context to be checked

0 if successful, or an POLARSSL_ERR_RSA_XXX error code

*/ public";

%csmethodmodifiers  gdcm::RSA::GetLenkey " /** unsigned int
gdcm::RSA::GetLenkey() const

Return the length of the key:.

*/ public";

%csmethodmodifiers  gdcm::RSA::Pkcs1Decrypt " /** int
gdcm::RSA::Pkcs1Decrypt(int mode, unsigned int &olen, const char
*input, char *output, unsigned int output_max_len)

Do an RSA operation, then remove the message padding.

Parameters:
-----------

ctx:   RSA context

mode:  RSA_PUBLIC or RSA_PRIVATE

input:  buffer holding the encrypted data

output:  buffer that will hold the plaintext

olen:  will contain the plaintext length

output_max_len:  maximum length of the output buffer

0 if successful, or an POLARSSL_ERR_RSA_XXX error code

The output buffer must be as large as the size of ctx->N (eg. 128
bytes if RSA-1024 is used) otherwise an error is thrown.

*/ public";

%csmethodmodifiers  gdcm::RSA::Pkcs1Encrypt " /** int
gdcm::RSA::Pkcs1Encrypt(int mode, unsigned int ilen, const char
*input, char *output) const

Add the message padding, then do an RSA operation.

Parameters:
-----------

ctx:   RSA context

mode:  RSA_PUBLIC or RSA_PRIVATE

ilen:  contains the plaintext length

input:  buffer holding the data to be encrypted

output:  buffer that will hold the ciphertext

0 if successful, or an XYSSL_ERR_RSA_XXX error code

The output buffer must be as large as the size of ctx->N (eg. 128
bytes if RSA-1024 is used).

*/ public";

%csmethodmodifiers  gdcm::RSA::RSA " /** gdcm::RSA::RSA()  */ public";

%csmethodmodifiers  gdcm::RSA::X509ParseKey " /** int
gdcm::RSA::X509ParseKey(const char *buf, unsigned int buflen, const
char *pwd=0, unsigned int pwdlen=0)

Parse a private RSA key.

Parameters:
-----------

rsa:   RSA context to be initialized

buf:  input buffer

buflen:  size of the buffer

pwd:  password for decryption (optional)

pwdlen:  size of the password

0 if successful, or a specific X509 error code

*/ public";

%csmethodmodifiers  gdcm::RSA::X509ParseKeyfile " /** int
gdcm::RSA::X509ParseKeyfile(const char *path, const char *password=0)

Load and parse a private RSA key.

Parameters:
-----------

rsa:   RSA context to be initialized

path:  filename to read the private key from

pwd:  password to decrypt the file (can be NULL)

0 if successful, or a specific X509 error code

*/ public";

%csmethodmodifiers  gdcm::RSA::X509WriteKeyfile " /** int
gdcm::RSA::X509WriteKeyfile(const char *path, int format=OUTPUT_PEM)
*/ public";

%csmethodmodifiers  gdcm::RSA::~RSA " /** gdcm::RSA::~RSA()  */
public";


// File: classstd_1_1runtime__error.xml
%typemap("csclassmodifiers") std::runtime_error " /** STL class.

*/ public class";


// File: classgdcm_1_1Scanner.xml
%typemap("csclassmodifiers") gdcm::Scanner " /**  Scanner.

Todo This filter is dealing with both VRASCII and VRBINARY element,
thanks to the help of gdcm::StringFilter

WARNING:  : IMPORTANT In case of file where tags are not ordered, the
output will be garbage

: implementation details. All values are stored in a std::set of
std::string. Then the *address* of the cstring underlying the
std::string is used in the std::map

C++ includes: gdcmScanner.h */ public class";

%csmethodmodifiers  gdcm::Scanner::AddSkipTag " /** void
gdcm::Scanner::AddSkipTag(Tag const &t)

Add a tag that will need to be skipped. Those are root level skip
tags.

*/ public";

%csmethodmodifiers  gdcm::Scanner::AddTag " /** void
gdcm::Scanner::AddTag(Tag const &t)

Add a tag that will need to be read. Those are root level skip tags.

*/ public";

%csmethodmodifiers  gdcm::Scanner::Begin " /** ConstIterator
gdcm::Scanner::Begin() const  */ public";

%csmethodmodifiers  gdcm::Scanner::ClearSkipTags " /** void
gdcm::Scanner::ClearSkipTags()  */ public";

%csmethodmodifiers  gdcm::Scanner::ClearTags " /** void
gdcm::Scanner::ClearTags()  */ public";

%csmethodmodifiers  gdcm::Scanner::End " /** ConstIterator
gdcm::Scanner::End() const  */ public";

%csmethodmodifiers  gdcm::Scanner::GetFilenames " /**
Directory::FilenamesType const& gdcm::Scanner::GetFilenames() const
*/ public";

%csmethodmodifiers  gdcm::Scanner::GetKeys " /**
Directory::FilenamesType gdcm::Scanner::GetKeys() const

Return the list of filename that are key in the internal map, which
means those filename were properly parsed

*/ public";

%csmethodmodifiers  gdcm::Scanner::GetMapping " /** TagToValue const&
gdcm::Scanner::GetMapping(const char *filename) const

Get the std::map mapping filenames to value for file 'filename'.

*/ public";

%csmethodmodifiers  gdcm::Scanner::GetMappings " /** MappingType
const& gdcm::Scanner::GetMappings() const

Mappings are the mapping from a particular tag to the map, mapping
filename to value:.

*/ public";

%csmethodmodifiers  gdcm::Scanner::GetValue " /** const char*
gdcm::Scanner::GetValue(const char *filename, Tag const &t) const

Retrieve the value found for tag: t associated with file: filename
This is meant for a single short call. If multiple calls (multiple
tags) should be done, prefer the GetMapping function, and then reuse
the TagToValue hash table. WARNING:   Tag 't' should have been added
via AddTag() prior to the Scan() call !

*/ public";

%csmethodmodifiers  gdcm::Scanner::GetValues " /** ValuesType
gdcm::Scanner::GetValues(Tag const &t) const

Get all the values found (in lexicographic order) associated with Tag
't'.

*/ public";

%csmethodmodifiers  gdcm::Scanner::GetValues " /** ValuesType const&
gdcm::Scanner::GetValues() const

Get all the values found (in lexicographic order).

*/ public";

%csmethodmodifiers  gdcm::Scanner::IsKey " /** bool
gdcm::Scanner::IsKey(const char *filename) const

Check if filename is a key in the Mapping table. returns true only of
file can be found, which means the file was indeed a DICOM file that
could be processed

*/ public";

%csmethodmodifiers  gdcm::Scanner::Print " /** void
gdcm::Scanner::Print(std::ostream &os) const

Print result.

*/ public";

%csmethodmodifiers  gdcm::Scanner::Scan " /** bool
gdcm::Scanner::Scan(Directory::FilenamesType const &filenames)

Start the scan !

*/ public";

%csmethodmodifiers  gdcm::Scanner::Scanner " /**
gdcm::Scanner::Scanner()  */ public";

%csmethodmodifiers  gdcm::Scanner::~Scanner " /**
gdcm::Scanner::~Scanner()  */ public";


// File: structgdcm_1_1Scanner_1_1ltstr.xml
%typemap("csclassmodifiers") gdcm::Scanner::ltstr " /**C++ includes:
gdcmScanner.h */ public class";


// File: classgdcm_1_1SegmentedPaletteColorLookupTable.xml
%typemap("csclassmodifiers") gdcm::SegmentedPaletteColorLookupTable "
/**  SegmentedPaletteColorLookupTable class.

C++ includes: gdcmSegmentedPaletteColorLookupTable.h */ public class";

%csmethodmodifiers  gdcm::SegmentedPaletteColorLookupTable::Print "
/** void gdcm::SegmentedPaletteColorLookupTable::Print(std::ostream &)
const  */ public";

%csmethodmodifiers
gdcm::SegmentedPaletteColorLookupTable::SegmentedPaletteColorLookupTable
" /**
gdcm::SegmentedPaletteColorLookupTable::SegmentedPaletteColorLookupTable()
*/ public";

%csmethodmodifiers  gdcm::SegmentedPaletteColorLookupTable::SetLUT "
/** void
gdcm::SegmentedPaletteColorLookupTable::SetLUT(LookupTableType type,
const unsigned char *array, unsigned int length)

Initialize a SegmentedPaletteColorLookupTable.

*/ public";

%csmethodmodifiers
gdcm::SegmentedPaletteColorLookupTable::~SegmentedPaletteColorLookupTable
" /**
gdcm::SegmentedPaletteColorLookupTable::~SegmentedPaletteColorLookupTable()
*/ public";


// File: classgdcm_1_1SequenceOfFragments.xml
%typemap("csclassmodifiers") gdcm::SequenceOfFragments " /** Class to
represent a Sequence Of Fragments.

Todo I do not enforce that Sequence of Fragments ends with a SQ end
del

C++ includes: gdcmSequenceOfFragments.h */ public class";

%csmethodmodifiers  gdcm::SequenceOfFragments::AddFragment " /** void
gdcm::SequenceOfFragments::AddFragment(Fragment const &item)

Appends a Fragment to the already added ones.

*/ public";

%csmethodmodifiers  gdcm::SequenceOfFragments::Clear " /** void
gdcm::SequenceOfFragments::Clear()  */ public";

%csmethodmodifiers  gdcm::SequenceOfFragments::ComputeByteLength " /**
unsigned long gdcm::SequenceOfFragments::ComputeByteLength() const  */
public";

%csmethodmodifiers  gdcm::SequenceOfFragments::ComputeLength " /** VL
gdcm::SequenceOfFragments::ComputeLength() const  */ public";

%csmethodmodifiers  gdcm::SequenceOfFragments::GetBuffer " /** bool
gdcm::SequenceOfFragments::GetBuffer(char *buffer, unsigned long
length) const  */ public";

%csmethodmodifiers  gdcm::SequenceOfFragments::GetFragBuffer " /**
bool gdcm::SequenceOfFragments::GetFragBuffer(unsigned int fragNb,
char *buffer, unsigned long &length) const  */ public";

%csmethodmodifiers  gdcm::SequenceOfFragments::GetFragment " /** const
Fragment& gdcm::SequenceOfFragments::GetFragment(unsigned int num)
const  */ public";

%csmethodmodifiers  gdcm::SequenceOfFragments::GetLength " /** VL
gdcm::SequenceOfFragments::GetLength() const

Returns the SQ length, as read from disk.

*/ public";

%csmethodmodifiers  gdcm::SequenceOfFragments::GetNumberOfFragments "
/** unsigned int gdcm::SequenceOfFragments::GetNumberOfFragments()
const  */ public";

%csmethodmodifiers  gdcm::SequenceOfFragments::GetTable " /**
BasicOffsetTable& gdcm::SequenceOfFragments::GetTable()  */ public";

%csmethodmodifiers  gdcm::SequenceOfFragments::GetTable " /** const
BasicOffsetTable& gdcm::SequenceOfFragments::GetTable() const  */
public";

%csmethodmodifiers  gdcm::SequenceOfFragments::Print " /** void
gdcm::SequenceOfFragments::Print(std::ostream &os) const  */ public";

%csmethodmodifiers  gdcm::SequenceOfFragments::Read " /**
std::istream& gdcm::SequenceOfFragments::Read(std::istream &is)  */
public";

%csmethodmodifiers  gdcm::SequenceOfFragments::SequenceOfFragments "
/** gdcm::SequenceOfFragments::SequenceOfFragments()

constructor (UndefinedLength by default)

*/ public";

%csmethodmodifiers  gdcm::SequenceOfFragments::SetLength " /** void
gdcm::SequenceOfFragments::SetLength(VL length)

Sets the actual SQ length.

*/ public";

%csmethodmodifiers  gdcm::SequenceOfFragments::Write " /**
std::ostream const& gdcm::SequenceOfFragments::Write(std::ostream &os)
const  */ public";

%csmethodmodifiers  gdcm::SequenceOfFragments::WriteBuffer " /** bool
gdcm::SequenceOfFragments::WriteBuffer(std::ostream &os) const  */
public";


// File: classgdcm_1_1SequenceOfItems.xml
%typemap("csclassmodifiers") gdcm::SequenceOfItems " /** Class to
represent a Sequence Of Items (value representation : SQ) a Value
Representation for Data Elements that contains a sequence of Data
Sets.

Sequence of Item allows for Nested Data Sets.

See PS 3.5, 7.4.6 Data Element Type Within a Sequence SEQUENCE OF
ITEMS (VALUE REPRESENTATION SQ) A Value Representation for Data
Elements that contain a sequence of Data Sets. Sequence of Items
allows for Nested Data Sets.

C++ includes: gdcmSequenceOfItems.h */ public class";

%csmethodmodifiers  gdcm::SequenceOfItems::AddItem " /** void
gdcm::SequenceOfItems::AddItem(Item const &item)

Appends an Item to the already added ones.

*/ public";

%csmethodmodifiers  gdcm::SequenceOfItems::Begin " /** ConstIterator
gdcm::SequenceOfItems::Begin() const  */ public";

%csmethodmodifiers  gdcm::SequenceOfItems::Begin " /** Iterator
gdcm::SequenceOfItems::Begin()  */ public";

%csmethodmodifiers  gdcm::SequenceOfItems::Clear " /** void
gdcm::SequenceOfItems::Clear()  */ public";

%csmethodmodifiers  gdcm::SequenceOfItems::ComputeLength " /** VL
gdcm::SequenceOfItems::ComputeLength() const  */ public";

%csmethodmodifiers  gdcm::SequenceOfItems::End " /** ConstIterator
gdcm::SequenceOfItems::End() const  */ public";

%csmethodmodifiers  gdcm::SequenceOfItems::End " /** Iterator
gdcm::SequenceOfItems::End()  */ public";

%csmethodmodifiers  gdcm::SequenceOfItems::FindDataElement " /** bool
gdcm::SequenceOfItems::FindDataElement(const Tag &t) const  */
public";

%csmethodmodifiers  gdcm::SequenceOfItems::GetItem " /** Item&
gdcm::SequenceOfItems::GetItem(unsigned int position)  */ public";

%csmethodmodifiers  gdcm::SequenceOfItems::GetItem " /** const Item&
gdcm::SequenceOfItems::GetItem(unsigned int position) const  */
public";

%csmethodmodifiers  gdcm::SequenceOfItems::GetLength " /** VL
gdcm::SequenceOfItems::GetLength() const

Returns the SQ length, as read from disk.

*/ public";

%csmethodmodifiers  gdcm::SequenceOfItems::GetNumberOfItems " /**
unsigned int gdcm::SequenceOfItems::GetNumberOfItems() const  */
public";

%csmethodmodifiers  gdcm::SequenceOfItems::IsUndefinedLength " /**
bool gdcm::SequenceOfItems::IsUndefinedLength() const

return if Value Length if of undefined length

*/ public";

%csmethodmodifiers  gdcm::SequenceOfItems::Print " /** void
gdcm::SequenceOfItems::Print(std::ostream &os) const  */ public";

%csmethodmodifiers  gdcm::SequenceOfItems::Read " /** std::istream&
gdcm::SequenceOfItems::Read(std::istream &is)  */ public";

%csmethodmodifiers  gdcm::SequenceOfItems::SequenceOfItems " /**
gdcm::SequenceOfItems::SequenceOfItems()

constructor (UndefinedLength by default)

*/ public";

%csmethodmodifiers  gdcm::SequenceOfItems::SetLength " /** void
gdcm::SequenceOfItems::SetLength(VL length)

Sets the actual SQ length.

*/ public";

%csmethodmodifiers  gdcm::SequenceOfItems::SetLengthToUndefined " /**
void gdcm::SequenceOfItems::SetLengthToUndefined()  */ public";

%csmethodmodifiers  gdcm::SequenceOfItems::Write " /** std::ostream
const& gdcm::SequenceOfItems::Write(std::ostream &os) const  */
public";


// File: classgdcm_1_1SerieHelper.xml
%typemap("csclassmodifiers") gdcm::SerieHelper " /** DO NOT USE this
class, it is only a temporary solution for ITK migration from GDCM 1.x
to GDCM 2.x It will disapear soon, you've been warned.

Instead see gdcm::ImageHelper or gdcm::IPPSorter

C++ includes: gdcmSerieHelper.h */ public class";

%csmethodmodifiers  gdcm::SerieHelper::AddRestriction " /** void
gdcm::SerieHelper::AddRestriction(uint16_t group, uint16_t elem,
std::string const &value, int op)  */ public";

%csmethodmodifiers  gdcm::SerieHelper::AddRestriction " /** void
gdcm::SerieHelper::AddRestriction(const std::string &tag)  */ public";

%csmethodmodifiers  gdcm::SerieHelper::Clear " /** void
gdcm::SerieHelper::Clear()  */ public";

%csmethodmodifiers
gdcm::SerieHelper::CreateDefaultUniqueSeriesIdentifier " /** void
gdcm::SerieHelper::CreateDefaultUniqueSeriesIdentifier()  */ public";

%csmethodmodifiers  gdcm::SerieHelper::CreateUniqueSeriesIdentifier "
/** std::string gdcm::SerieHelper::CreateUniqueSeriesIdentifier(File
*inFile)  */ public";

%csmethodmodifiers  gdcm::SerieHelper::GetFirstSingleSerieUIDFileSet "
/** FileList* gdcm::SerieHelper::GetFirstSingleSerieUIDFileSet()  */
public";

%csmethodmodifiers  gdcm::SerieHelper::GetNextSingleSerieUIDFileSet "
/** FileList* gdcm::SerieHelper::GetNextSingleSerieUIDFileSet()  */
public";

%csmethodmodifiers  gdcm::SerieHelper::OrderFileList " /** void
gdcm::SerieHelper::OrderFileList(FileList *fileSet)  */ public";

%csmethodmodifiers  gdcm::SerieHelper::SerieHelper " /**
gdcm::SerieHelper::SerieHelper()  */ public";

%csmethodmodifiers  gdcm::SerieHelper::SetDirectory " /** void
gdcm::SerieHelper::SetDirectory(std::string const &dir, bool
recursive=false)  */ public";

%csmethodmodifiers  gdcm::SerieHelper::SetLoadMode " /** void
gdcm::SerieHelper::SetLoadMode(int)  */ public";

%csmethodmodifiers  gdcm::SerieHelper::SetUseSeriesDetails " /** void
gdcm::SerieHelper::SetUseSeriesDetails(bool useSeriesDetails)  */
public";

%csmethodmodifiers  gdcm::SerieHelper::~SerieHelper " /**
gdcm::SerieHelper::~SerieHelper()  */ public";


// File: structgdcm_1_1SerieHelper_1_1Rule.xml


// File: classgdcm_1_1Series.xml
%typemap("csclassmodifiers") gdcm::Series " /**  Series.

C++ includes: gdcmSeries.h */ public class";

%csmethodmodifiers  gdcm::Series::Series " /** gdcm::Series::Series()
*/ public";


// File: classstd_1_1set.xml
%typemap("csclassmodifiers") std::set " /** STL class.

*/ public class";


// File: classstd_1_1set_1_1const__iterator.xml
%typemap("csclassmodifiers") std::set::const_iterator " /** STL
iterator class.

*/ public class";


// File: classstd_1_1set_1_1const__reverse__iterator.xml
%typemap("csclassmodifiers") std::set::const_reverse_iterator " /**
STL iterator class.

*/ public class";


// File: classstd_1_1set_1_1iterator.xml
%typemap("csclassmodifiers") std::set::iterator " /** STL iterator
class.

*/ public class";


// File: classstd_1_1set_1_1reverse__iterator.xml
%typemap("csclassmodifiers") std::set::reverse_iterator " /** STL
iterator class.

*/ public class";


// File: classgdcm_1_1SHA1.xml
%typemap("csclassmodifiers") gdcm::SHA1 " /**C++ includes: gdcmSHA1.h
*/ public class";

%csmethodmodifiers  gdcm::SHA1::SHA1 " /** gdcm::SHA1::SHA1()  */
public";

%csmethodmodifiers  gdcm::SHA1::~SHA1 " /** gdcm::SHA1::~SHA1()  */
public";


// File: classgdcm_1_1SmartPointer.xml
%typemap("csclassmodifiers") gdcm::SmartPointer " /** Class for Smart
Pointer.

Will only work for subclass of gdcm::Object See tr1/shared_ptr for a
more general approach (not invasive) include <tr1/memory> {
shared_ptr<Bla> b(new Bla); } Class partly based on post by Bill
Hubauer:http://groups.google.com/group/comp.lang.c++/msg/173ddc38a827a930

See:  http://www.davethehat.com/articles/smartp.htm  and
itk::SmartPointer

C++ includes: gdcmSmartPointer.h */ public class";

%csmethodmodifiers  gdcm::SmartPointer::GetPointer " /** ObjectType*
gdcm::SmartPointer< ObjectType >::GetPointer() const

Explicit function to retrieve the pointer.

*/ public";

%csmethodmodifiers  gdcm::SmartPointer::SmartPointer " /**
gdcm::SmartPointer< ObjectType >::SmartPointer(ObjectType const &p)
*/ public";

%csmethodmodifiers  gdcm::SmartPointer::SmartPointer " /**
gdcm::SmartPointer< ObjectType >::SmartPointer(ObjectType *p)  */
public";

%csmethodmodifiers  gdcm::SmartPointer::SmartPointer " /**
gdcm::SmartPointer< ObjectType >::SmartPointer(const SmartPointer<
ObjectType > &p)  */ public";

%csmethodmodifiers  gdcm::SmartPointer::SmartPointer " /**
gdcm::SmartPointer< ObjectType >::SmartPointer()  */ public";

%csmethodmodifiers  gdcm::SmartPointer::~SmartPointer " /**
gdcm::SmartPointer< ObjectType >::~SmartPointer()  */ public";


// File: classgdcm_1_1SOPClassUIDToIOD.xml
%typemap("csclassmodifiers") gdcm::SOPClassUIDToIOD " /** Class
convert a class uid into IOD.

C++ includes: gdcmSOPClassUIDToIOD.h */ public class";


// File: classgdcm_1_1Sorter.xml
%typemap("csclassmodifiers") gdcm::Sorter " /**C++ includes:
gdcmSorter.h */ public class";

%csmethodmodifiers  gdcm::Sorter::AddSelect " /** bool
gdcm::Sorter::AddSelect(Tag const &tag, const char *value)

UNSUPPORTED FOR NOW.

*/ public";

%csmethodmodifiers  gdcm::Sorter::GetFilenames " /** const
std::vector<std::string>& gdcm::Sorter::GetFilenames() const

Return the list of filenames as sorted by the specific algorithm used.
Empty by default (before Sort() is called)

*/ public";

%csmethodmodifiers  gdcm::Sorter::Print " /** void
gdcm::Sorter::Print(std::ostream &os) const

Print.

*/ public";

%csmethodmodifiers  gdcm::Sorter::SetSortFunction " /** void
gdcm::Sorter::SetSortFunction(SortFunction f)  */ public";

%csmethodmodifiers  gdcm::Sorter::Sort " /** virtual bool
gdcm::Sorter::Sort(std::vector< std::string > const &filenames)

Typically the output of gdcm::Directory::GetFilenames().

*/ public";

%csmethodmodifiers  gdcm::Sorter::Sorter " /** gdcm::Sorter::Sorter()
*/ public";

%csmethodmodifiers  gdcm::Sorter::StableSort " /** virtual bool
gdcm::Sorter::StableSort(std::vector< std::string > const &filenames)
*/ public";

%csmethodmodifiers  gdcm::Sorter::~Sorter " /** virtual
gdcm::Sorter::~Sorter()  */ public";


// File: classgdcm_1_1Spacing.xml
%typemap("csclassmodifiers") gdcm::Spacing " /** Class for Spacing.

See PS 3.3-2008, Table C.7-11b IMAGE PIXEL MACRO ATTRIBUTES

Ratio of the vertical size and horizontal size of the pixels in the
image specified by a pair of integer values where the first value is
the vertical pixel size, and the second value is the horizontal pixel
size. Required if the aspect ratio values do not have a ratio of 1:1
and the physical pixel spacing is not specified by Pixel Spacing
(0028,0030), or Imager Pixel Spacing (0018,1164) or Nominal Scanned
Pixel Spacing (0018,2010), either for the entire Image or per-frame in
a Functional Group Macro. See C.7.6.3.1.7.

PS 3.3-2008 10.7.1.3 Pixel Spacing Value Order and Valid Values All
pixel spacing related attributes shall have non-zero values, except
when there is only a single row or column or pixel of data present, in
which case the corresponding value may be zero.

Ref:http://apps.sourceforge.net/mediawiki/gdcm/index.php?title=Imager_Pixel_Spacing

C++ includes: gdcmSpacing.h */ public class";

%csmethodmodifiers  gdcm::Spacing::Spacing " /**
gdcm::Spacing::Spacing()  */ public";

%csmethodmodifiers  gdcm::Spacing::~Spacing " /**
gdcm::Spacing::~Spacing()  */ public";


// File: classgdcm_1_1Spectroscopy.xml
%typemap("csclassmodifiers") gdcm::Spectroscopy " /**  Spectroscopy
class.

C++ includes: gdcmSpectroscopy.h */ public class";

%csmethodmodifiers  gdcm::Spectroscopy::Spectroscopy " /**
gdcm::Spectroscopy::Spectroscopy()  */ public";


// File: classgdcm_1_1SplitMosaicFilter.xml
%typemap("csclassmodifiers") gdcm::SplitMosaicFilter " /**
SplitMosaicFilter class Class to reshuffle bytes for a SIEMENS Mosaic
image.

C++ includes: gdcmSplitMosaicFilter.h */ public class";

%csmethodmodifiers  gdcm::SplitMosaicFilter::GetFile " /** const File&
gdcm::SplitMosaicFilter::GetFile() const  */ public";

%csmethodmodifiers  gdcm::SplitMosaicFilter::GetFile " /** File&
gdcm::SplitMosaicFilter::GetFile()  */ public";

%csmethodmodifiers  gdcm::SplitMosaicFilter::GetImage " /** Image&
gdcm::SplitMosaicFilter::GetImage()  */ public";

%csmethodmodifiers  gdcm::SplitMosaicFilter::GetImage " /** const
Image& gdcm::SplitMosaicFilter::GetImage() const  */ public";

%csmethodmodifiers  gdcm::SplitMosaicFilter::SetFile " /** void
gdcm::SplitMosaicFilter::SetFile(const File &f)  */ public";

%csmethodmodifiers  gdcm::SplitMosaicFilter::SetImage " /** void
gdcm::SplitMosaicFilter::SetImage(const Image &image)  */ public";

%csmethodmodifiers  gdcm::SplitMosaicFilter::Split " /** bool
gdcm::SplitMosaicFilter::Split()

Split the SIEMENS MOSAIC image.

*/ public";

%csmethodmodifiers  gdcm::SplitMosaicFilter::SplitMosaicFilter " /**
gdcm::SplitMosaicFilter::SplitMosaicFilter()  */ public";

%csmethodmodifiers  gdcm::SplitMosaicFilter::~SplitMosaicFilter " /**
gdcm::SplitMosaicFilter::~SplitMosaicFilter()  */ public";


// File: classstd_1_1stack.xml
%typemap("csclassmodifiers") std::stack " /** STL class.

*/ public class";


// File: structgdcm_1_1static__assert__test.xml
%typemap("csclassmodifiers") gdcm::static_assert_test " /**C++
includes: gdcmStaticAssert.h */ public class";


// File: structgdcm_1_1STATIC__ASSERTION__FAILURE_3_01true_01_4.xml
%typemap("csclassmodifiers") gdcm::STATIC_ASSERTION_FAILURE< true > "
/**C++ includes: gdcmStaticAssert.h */ public class";


// File: classgdcm_1_1String.xml
%typemap("csclassmodifiers") gdcm::String " /**  String.

TDelimiter template parameter is used to separate multiple String (VM1
>) TMaxLength is only a hint. Noone actually respect the max length
TPadChar is the string padding (0 or space)

C++ includes: gdcmString.h */ public class";

%csmethodmodifiers  gdcm::String::IsValid " /** bool gdcm::String<
TDelimiter, TMaxLength, TPadChar >::IsValid() const

return if string is valid

*/ public";

%csmethodmodifiers  gdcm::String::String " /** gdcm::String<
TDelimiter, TMaxLength, TPadChar >::String(const std::string &s,
size_type pos=0, size_type n=npos)  */ public";

%csmethodmodifiers  gdcm::String::String " /** gdcm::String<
TDelimiter, TMaxLength, TPadChar >::String(const value_type *s,
size_type n)  */ public";

%csmethodmodifiers  gdcm::String::String " /** gdcm::String<
TDelimiter, TMaxLength, TPadChar >::String(const value_type *s)  */
public";

%csmethodmodifiers  gdcm::String::String " /** gdcm::String<
TDelimiter, TMaxLength, TPadChar >::String()

String constructors.

*/ public";

%csmethodmodifiers  gdcm::String::Trim " /** std::string gdcm::String<
TDelimiter, TMaxLength, TPadChar >::Trim() const

Trim function is required to return a std::string object, otherwise we
could not create a gdcm::String object with an odd number of bytes...

*/ public";


// File: classstd_1_1string.xml
%typemap("csclassmodifiers") std::string " /** STL class.

*/ public class";


// File: classstd_1_1string_1_1const__iterator.xml
%typemap("csclassmodifiers") std::string::const_iterator " /** STL
iterator class.

*/ public class";


// File: classstd_1_1string_1_1const__reverse__iterator.xml
%typemap("csclassmodifiers") std::string::const_reverse_iterator " /**
STL iterator class.

*/ public class";


// File: classstd_1_1string_1_1iterator.xml
%typemap("csclassmodifiers") std::string::iterator " /** STL iterator
class.

*/ public class";


// File: classstd_1_1string_1_1reverse__iterator.xml
%typemap("csclassmodifiers") std::string::reverse_iterator " /** STL
iterator class.

*/ public class";


// File: classgdcm_1_1StringFilter.xml
%typemap("csclassmodifiers") gdcm::StringFilter " /**  StringFilter
StringFilter is the class that make gdcm2.x looks more like gdcm1 and
transform the binary blob contained in a DataElement into a string,
typically this is a nice feature to have for wrapped language.

C++ includes: gdcmStringFilter.h */ public class";

%csmethodmodifiers  gdcm::StringFilter::FromString " /** std::string
gdcm::StringFilter::FromString(const Tag &t, const char *value, VL
const &vl)  */ public";

%csmethodmodifiers  gdcm::StringFilter::GetFile " /** const File&
gdcm::StringFilter::GetFile() const  */ public";

%csmethodmodifiers  gdcm::StringFilter::GetFile " /** File&
gdcm::StringFilter::GetFile()  */ public";

%csmethodmodifiers  gdcm::StringFilter::SetDicts " /** void
gdcm::StringFilter::SetDicts(const Dicts &dicts)

Allow user to pass in there own dicts.

*/ public";

%csmethodmodifiers  gdcm::StringFilter::SetFile " /** void
gdcm::StringFilter::SetFile(const File &f)

Set/Get File.

*/ public";

%csmethodmodifiers  gdcm::StringFilter::StringFilter " /**
gdcm::StringFilter::StringFilter()  */ public";

%csmethodmodifiers  gdcm::StringFilter::ToString " /** std::string
gdcm::StringFilter::ToString(const Tag &t) const

Convert to string the ByteValue contained in a DataElement.

*/ public";

%csmethodmodifiers  gdcm::StringFilter::ToStringPair " /**
std::pair<std::string, std::string>
gdcm::StringFilter::ToStringPair(const Tag &t) const

Convert to string the ByteValue contained in a DataElement the
returned elements are: pair.first : the name as found in the
dictionary of DataElement pari.second : the value encoded into a
string (US,UL...) are properly converted

*/ public";

%csmethodmodifiers  gdcm::StringFilter::UseDictAlways " /** void
gdcm::StringFilter::UseDictAlways(bool)  */ public";

%csmethodmodifiers  gdcm::StringFilter::~StringFilter " /**
gdcm::StringFilter::~StringFilter()  */ public";


// File: classstd_1_1stringstream.xml
%typemap("csclassmodifiers") std::stringstream " /** STL class.

*/ public class";


// File: classgdcm_1_1Study.xml
%typemap("csclassmodifiers") gdcm::Study " /**  Study.

C++ includes: gdcmStudy.h */ public class";

%csmethodmodifiers  gdcm::Study::Study " /** gdcm::Study::Study()  */
public";


// File: classgdcm_1_1SwapCode.xml
%typemap("csclassmodifiers") gdcm::SwapCode " /**  SwapCode
representation.

C++ includes: gdcmSwapCode.h */ public class";

%csmethodmodifiers  gdcm::SwapCode::SwapCode " /**
gdcm::SwapCode::SwapCode(SwapCodeType sc=Unknown)  */ public";


// File: classgdcm_1_1SwapperDoOp.xml
%typemap("csclassmodifiers") gdcm::SwapperDoOp " /**C++ includes:
gdcmSwapper.h */ public class";


// File: classgdcm_1_1SwapperNoOp.xml
%typemap("csclassmodifiers") gdcm::SwapperNoOp " /**C++ includes:
gdcmSwapper.h */ public class";


// File: classgdcm_1_1System.xml
%typemap("csclassmodifiers") gdcm::System " /** Class to do system
operation.

OS independant functionalities

C++ includes: gdcmSystem.h */ public class";


// File: classgdcm_1_1Table.xml
%typemap("csclassmodifiers") gdcm::Table " /**  Table.

C++ includes: gdcmTable.h */ public class";

%csmethodmodifiers  gdcm::Table::GetTableEntry " /** const TableEntry&
gdcm::Table::GetTableEntry(const Tag &tag) const  */ public";

%csmethodmodifiers  gdcm::Table::InsertEntry " /** void
gdcm::Table::InsertEntry(Tag const &tag, TableEntry const &te)  */
public";

%csmethodmodifiers  gdcm::Table::Table " /** gdcm::Table::Table()  */
public";

%csmethodmodifiers  gdcm::Table::~Table " /** gdcm::Table::~Table()
*/ public";


// File: classgdcm_1_1TableEntry.xml
%typemap("csclassmodifiers") gdcm::TableEntry " /**  TableEntry.

C++ includes: gdcmTableEntry.h */ public class";

%csmethodmodifiers  gdcm::TableEntry::TableEntry " /**
gdcm::TableEntry::TableEntry(const char *attribute=0, Type const
&type=Type(), const char *des=0)  */ public";

%csmethodmodifiers  gdcm::TableEntry::~TableEntry " /**
gdcm::TableEntry::~TableEntry()  */ public";


// File: classgdcm_1_1TableReader.xml
%typemap("csclassmodifiers") gdcm::TableReader " /** Class for
representing a TableReader.

This class is an empty shell meant to be derived

C++ includes: gdcmTableReader.h */ public class";

%csmethodmodifiers  gdcm::TableReader::CharacterDataHandler " /**
virtual void gdcm::TableReader::CharacterDataHandler(const char *data,
int length)  */ public";

%csmethodmodifiers  gdcm::TableReader::EndElement " /** virtual void
gdcm::TableReader::EndElement(const char *name)  */ public";

%csmethodmodifiers  gdcm::TableReader::GetDefs " /** const Defs&
gdcm::TableReader::GetDefs() const  */ public";

%csmethodmodifiers  gdcm::TableReader::GetFilename " /** const char*
gdcm::TableReader::GetFilename()  */ public";

%csmethodmodifiers  gdcm::TableReader::HandleIOD " /** void
gdcm::TableReader::HandleIOD(const char **atts)  */ public";

%csmethodmodifiers  gdcm::TableReader::HandleIODEntry " /** void
gdcm::TableReader::HandleIODEntry(const char **atts)  */ public";

%csmethodmodifiers  gdcm::TableReader::HandleMacro " /** void
gdcm::TableReader::HandleMacro(const char **atts)  */ public";

%csmethodmodifiers  gdcm::TableReader::HandleMacroEntry " /** void
gdcm::TableReader::HandleMacroEntry(const char **atts)  */ public";

%csmethodmodifiers  gdcm::TableReader::HandleMacroEntryDescription "
/** void gdcm::TableReader::HandleMacroEntryDescription(const char
**atts)  */ public";

%csmethodmodifiers  gdcm::TableReader::HandleModule " /** void
gdcm::TableReader::HandleModule(const char **atts)  */ public";

%csmethodmodifiers  gdcm::TableReader::HandleModuleEntry " /** void
gdcm::TableReader::HandleModuleEntry(const char **atts)  */ public";

%csmethodmodifiers  gdcm::TableReader::HandleModuleEntryDescription "
/** void gdcm::TableReader::HandleModuleEntryDescription(const char
**atts)  */ public";

%csmethodmodifiers  gdcm::TableReader::Read " /** int
gdcm::TableReader::Read()  */ public";

%csmethodmodifiers  gdcm::TableReader::SetFilename " /** void
gdcm::TableReader::SetFilename(const char *filename)  */ public";

%csmethodmodifiers  gdcm::TableReader::StartElement " /** virtual void
gdcm::TableReader::StartElement(const char *name, const char **atts)
*/ public";

%csmethodmodifiers  gdcm::TableReader::TableReader " /**
gdcm::TableReader::TableReader(Defs &defs)  */ public";

%csmethodmodifiers  gdcm::TableReader::~TableReader " /** virtual
gdcm::TableReader::~TableReader()  */ public";


// File: classgdcm_1_1Tag.xml
%typemap("csclassmodifiers") gdcm::Tag " /** Class to represent a
DICOM Data Element ( Attribute) Tag (Group, Element). Basically an
uint32_t which can also be expressed as two uint16_t (group and
element).

DATA ELEMENT TAG: A unique identifier for a Data Element composed of
an ordered pair of numbers (a Group Number followed by an Element
Number). GROUP NUMBER: The first number in the ordered pair of numbers
that makes up a Data Element Tag. ELEMENT NUMBER: The second number in
the ordered pair of numbers that makes up a Data Element Tag.

C++ includes: gdcmTag.h */ public class";

%csmethodmodifiers  gdcm::Tag::GetElement " /** uint16_t
gdcm::Tag::GetElement() const

Returns the 'Element number' of the given Tag.

*/ public";

%csmethodmodifiers  gdcm::Tag::GetElementTag " /** uint32_t
gdcm::Tag::GetElementTag() const

Returns the full tag value of the given Tag.

*/ public";

%csmethodmodifiers  gdcm::Tag::GetGroup " /** uint16_t
gdcm::Tag::GetGroup() const

Returns the 'Group number' of the given Tag.

*/ public";

%csmethodmodifiers  gdcm::Tag::GetLength " /** uint32_t
gdcm::Tag::GetLength() const

return the length of tag (read: size on disk)

*/ public";

%csmethodmodifiers  gdcm::Tag::GetPrivateCreator " /** Tag
gdcm::Tag::GetPrivateCreator() const

Return the Private Creator Data Element tag of a private data element.

*/ public";

%csmethodmodifiers  gdcm::Tag::IsGroupLength " /** bool
gdcm::Tag::IsGroupLength() const

return whether the tag correspond to a group length tag:

*/ public";

%csmethodmodifiers  gdcm::Tag::IsGroupXX " /** bool
gdcm::Tag::IsGroupXX(const Tag &t) const

e.g 6002,3000 belong to groupXX: 6000,3000

*/ public";

%csmethodmodifiers  gdcm::Tag::IsIllegal " /** bool
gdcm::Tag::IsIllegal() const

return if the tag is considered to be an illegal tag

*/ public";

%csmethodmodifiers  gdcm::Tag::IsPrivate " /** bool
gdcm::Tag::IsPrivate() const

PRIVATE DATA ELEMENT: Additional Data Element, defined by an
implementor, to communicate information that is not contained in
Standard Data Elements. Private Data elements have odd Group Numbers.

*/ public";

%csmethodmodifiers  gdcm::Tag::IsPrivateCreator " /** bool
gdcm::Tag::IsPrivateCreator() const

Returns if tag is a Private Creator (xxxx,00yy), where xxxx is odd
number and yy in [0x10,0xFF].

*/ public";

%csmethodmodifiers  gdcm::Tag::IsPublic " /** bool
gdcm::Tag::IsPublic() const

STANDARD DATA ELEMENT: A Data Element defined in the DICOM Standard,
and therefore listed in the DICOM Data Element Dictionary in PS 3.6.
Is the Tag from the Public dict...well the implementation is buggy it
does not prove the element is indeed in the dict...

*/ public";

%csmethodmodifiers  gdcm::Tag::PrintAsPipeSeparatedString " /**
std::string gdcm::Tag::PrintAsPipeSeparatedString() const

Print as a pipe separated string (GDCM 1.x compat only). Do not use in
newer code See:   ReadFromPipeSeparatedString

*/ public";

%csmethodmodifiers  gdcm::Tag::Read " /** std::istream&
gdcm::Tag::Read(std::istream &is)

Read a tag from binary representation.

*/ public";

%csmethodmodifiers  gdcm::Tag::ReadFromCommaSeparatedString " /** bool
gdcm::Tag::ReadFromCommaSeparatedString(const char *str)

Read from a comma separated string. This is a highly user oriented
function, the string should be formated as: 1234,5678 to specify the
tag (0x1234,0x5678) The notation comes from the DICOM standard, and is
handy to use from a command line program

*/ public";

%csmethodmodifiers  gdcm::Tag::ReadFromPipeSeparatedString " /** bool
gdcm::Tag::ReadFromPipeSeparatedString(const char *str)

Read from a pipe separated string (GDCM 1.x compat only). Do not use
in newer code See:   ReadFromCommaSeparatedString

*/ public";

%csmethodmodifiers  gdcm::Tag::SetElement " /** void
gdcm::Tag::SetElement(uint16_t element)

Sets the 'Element number' of the given Tag.

*/ public";

%csmethodmodifiers  gdcm::Tag::SetElementTag " /** void
gdcm::Tag::SetElementTag(uint32_t tag)

Sets the full tag value of the given Tag.

*/ public";

%csmethodmodifiers  gdcm::Tag::SetElementTag " /** void
gdcm::Tag::SetElementTag(uint16_t group, uint16_t element)

Sets the 'Group number' & 'Element number' of the given Tag.

*/ public";

%csmethodmodifiers  gdcm::Tag::SetGroup " /** void
gdcm::Tag::SetGroup(uint16_t group)

Sets the 'Group number' of the given Tag.

*/ public";

%csmethodmodifiers  gdcm::Tag::SetPrivateCreator " /** void
gdcm::Tag::SetPrivateCreator(Tag const &t)

Set private creator:.

*/ public";

%csmethodmodifiers  gdcm::Tag::Tag " /** gdcm::Tag::Tag(const Tag
&_val)  */ public";

%csmethodmodifiers  gdcm::Tag::Tag " /** gdcm::Tag::Tag(uint32_t
tag=0)

Constructor with 1*uint32_t Prefer the cstor that takes two uint16_t.

*/ public";

%csmethodmodifiers  gdcm::Tag::Tag " /** gdcm::Tag::Tag(uint16_t
group, uint16_t element)

Constructor with 2*uint16_t.

*/ public";

%csmethodmodifiers  gdcm::Tag::Write " /** const std::ostream&
gdcm::Tag::Write(std::ostream &os) const

Write a tag in binary rep.

*/ public";


// File: classgdcm_1_1TagPath.xml
%typemap("csclassmodifiers") gdcm::TagPath " /** class to handle a
path of tag.

Any Resemblance to Existing XPath is Purely Coincidental

C++ includes: gdcmTagPath.h */ public class";

%csmethodmodifiers  gdcm::TagPath::ConstructFromString " /** bool
gdcm::TagPath::ConstructFromString(const char *path)

\"/0018,0018/\"... No space allowed, comma is use to separate tag
group from tag element and slash is used to separate tag return false
if invalid

*/ public";

%csmethodmodifiers  gdcm::TagPath::ConstructFromTagList " /** bool
gdcm::TagPath::ConstructFromTagList(Tag const *l, unsigned int n)

Construct from a list of tags.

*/ public";

%csmethodmodifiers  gdcm::TagPath::Print " /** void
gdcm::TagPath::Print(std::ostream &) const  */ public";

%csmethodmodifiers  gdcm::TagPath::Push " /** bool
gdcm::TagPath::Push(unsigned int itemnum)  */ public";

%csmethodmodifiers  gdcm::TagPath::Push " /** bool
gdcm::TagPath::Push(Tag const &t)  */ public";

%csmethodmodifiers  gdcm::TagPath::TagPath " /**
gdcm::TagPath::TagPath()  */ public";

%csmethodmodifiers  gdcm::TagPath::~TagPath " /**
gdcm::TagPath::~TagPath()  */ public";


// File: classgdcm_1_1Testing.xml
%typemap("csclassmodifiers") gdcm::Testing " /** class for testing

this class is used for the nightly regression system for GDCM It makes
heavily use of md5 computation

See:   gdcm::MD5 class for md5 computation

C++ includes: gdcmTesting.h */ public class";

%csmethodmodifiers  gdcm::Testing::Print " /** void
gdcm::Testing::Print(std::ostream &os=std::cout)

Print.

*/ public";

%csmethodmodifiers  gdcm::Testing::Testing " /**
gdcm::Testing::Testing()  */ public";

%csmethodmodifiers  gdcm::Testing::~Testing " /**
gdcm::Testing::~Testing()  */ public";


// File: classgdcm_1_1Trace.xml
%typemap("csclassmodifiers") gdcm::Trace " /**  Trace.

Debug / Warning and Error are encapsulated in this class

C++ includes: gdcmTrace.h */ public class";

%csmethodmodifiers  gdcm::Trace::Trace " /** gdcm::Trace::Trace()  */
public";

%csmethodmodifiers  gdcm::Trace::~Trace " /** gdcm::Trace::~Trace()
*/ public";


// File: classgdcm_1_1TransferSyntax.xml
%typemap("csclassmodifiers") gdcm::TransferSyntax " /** Class to
manipulate Transfer Syntax.

TRANSFER SYNTAX (Standard and Private): A set of encoding rules that
allow Application Entities to unambiguously negotiate the encoding
techniques (e.g., Data Element structure, byte ordering, compression)
they are able to support, thereby allowing these Application Entities
to communicate. Todo : The implementation is completely retarded ->
see gdcm::UIDs for a replacement We need: IsSupported We need
preprocess of raw/xml file We need GetFullName() Need a notion of
Private Syntax. As defined in Ps 3.5. Section 9.2

C++ includes: gdcmTransferSyntax.h */ public class";

%csmethodmodifiers  gdcm::TransferSyntax::GetNegociatedType " /**
NegociatedType gdcm::TransferSyntax::GetNegociatedType() const  */
public";

%csmethodmodifiers  gdcm::TransferSyntax::GetString " /** const char*
gdcm::TransferSyntax::GetString() const  */ public";

%csmethodmodifiers  gdcm::TransferSyntax::GetSwapCode " /** SwapCode
gdcm::TransferSyntax::GetSwapCode() const  */ public";

%csmethodmodifiers  gdcm::TransferSyntax::IsEncapsulated " /** bool
gdcm::TransferSyntax::IsEncapsulated() const  */ public";

%csmethodmodifiers  gdcm::TransferSyntax::IsEncoded " /** bool
gdcm::TransferSyntax::IsEncoded() const  */ public";

%csmethodmodifiers  gdcm::TransferSyntax::IsExplicit " /** bool
gdcm::TransferSyntax::IsExplicit() const  */ public";

%csmethodmodifiers  gdcm::TransferSyntax::IsImplicit " /** bool
gdcm::TransferSyntax::IsImplicit() const  */ public";

%csmethodmodifiers  gdcm::TransferSyntax::IsLossless " /** bool
gdcm::TransferSyntax::IsLossless() const  */ public";

%csmethodmodifiers  gdcm::TransferSyntax::IsLossy " /** bool
gdcm::TransferSyntax::IsLossy() const

Return whether the Transfer Syntax contains a lossy or lossless
Encapsulated stream WARNING:  IsLossy is NOT !IsLossless since JPEG
2000 Transfer Syntax is dual the stream can be either lossy or
lossless compressed.

*/ public";

%csmethodmodifiers  gdcm::TransferSyntax::IsValid " /** bool
gdcm::TransferSyntax::IsValid() const  */ public";

%csmethodmodifiers  gdcm::TransferSyntax::TransferSyntax " /**
gdcm::TransferSyntax::TransferSyntax(TSType
type=ImplicitVRLittleEndian)  */ public";


// File: classgdcm_1_1Type.xml
%typemap("csclassmodifiers") gdcm::Type " /**  Type.

PS 3.5 7.4 DATA ELEMENT TYPE 7.4.1 TYPE 1 REQUIRED DATA ELEMENTS 7.4.2
TYPE 1C CONDITIONAL DATA ELEMENTS 7.4.3 TYPE 2 REQUIRED DATA ELEMENTS
7.4.4 TYPE 2C CONDITIONAL DATA ELEMENTS 7.4.5 TYPE 3 OPTIONAL DATA
ELEMENTS  The intent of Type 2 Data Elements is to allow a zero length
to be conveyed when the operator or application does not know its
value or has a specific reason for not specifying its value. It is the
intent that the device should support these Data Elements.

C++ includes: gdcmType.h */ public class";

%csmethodmodifiers  gdcm::Type::Type " /** gdcm::Type::Type(TypeType
type=UNKNOWN)  */ public";


// File: structgdcm_1_1UI.xml
%typemap("csclassmodifiers") gdcm::UI " /**C++ includes: gdcmVR.h */
public class";


// File: classgdcm_1_1UIDGenerator.xml
%typemap("csclassmodifiers") gdcm::UIDGenerator " /** Class for
generating unique UID.

bla Usage: When constructing a Series or Study UID, user *has* to keep
around the UID, otherwise the UID Generator will simply forget the
value and create a new UID.

C++ includes: gdcmUIDGenerator.h */ public class";

%csmethodmodifiers  gdcm::UIDGenerator::Generate " /** const char*
gdcm::UIDGenerator::Generate()

Internally uses a std::string, so two calls have the same pointer !
save into a std::string In summary do not write code like that: const
char *uid1 = uid.Generate(); const char *uid2 = uid.Generate(); since
uid1 == uid2

*/ public";

%csmethodmodifiers  gdcm::UIDGenerator::UIDGenerator " /**
gdcm::UIDGenerator::UIDGenerator()

By default the root of a UID is a GDCM Root...

*/ public";


// File: classgdcm_1_1UIDs.xml
%typemap("csclassmodifiers") gdcm::UIDs " /** all known uids

C++ includes: gdcmUIDs.h */ public class";

%csmethodmodifiers  gdcm::UIDs::GetName " /** const char*
gdcm::UIDs::GetName() const

When object is Initialize function return the well known name
associated with uid return NULL when not initialized

*/ public";

%csmethodmodifiers  gdcm::UIDs::GetString " /** const char*
gdcm::UIDs::GetString() const

When object is Initialize function return the uid return NULL when not
initialized

*/ public";

%csmethodmodifiers  gdcm::UIDs::SetFromUID " /** bool
gdcm::UIDs::SetFromUID(const char *str)

Initialize object from a string (a uid number) return false on error,
and internal state is set to 0

*/ public";


// File: classstd_1_1underflow__error.xml
%typemap("csclassmodifiers") std::underflow_error " /** STL class.

*/ public class";


// File: classgdcm_1_1UNExplicitDataElement.xml
%typemap("csclassmodifiers") gdcm::UNExplicitDataElement " /** Class
to read/write a DataElement as UNExplicit Data Element.

bla

C++ includes: gdcmUNExplicitDataElement.h */ public class";

%csmethodmodifiers  gdcm::UNExplicitDataElement::GetLength " /** VL
gdcm::UNExplicitDataElement::GetLength() const  */ public";

%csmethodmodifiers  gdcm::UNExplicitDataElement::Read " /**
std::istream& gdcm::UNExplicitDataElement::Read(std::istream &is)  */
public";

%csmethodmodifiers  gdcm::UNExplicitDataElement::ReadWithLength " /**
std::istream& gdcm::UNExplicitDataElement::ReadWithLength(std::istream
&is, VL &length)  */ public";


// File: classgdcm_1_1UNExplicitImplicitDataElement.xml
%typemap("csclassmodifiers") gdcm::UNExplicitImplicitDataElement " /**
Class to read/write a DataElement as ExplicitImplicit Data Element
This class gather two known bugs: 1. GDCM 1.2.0 would rewrite VR=UN
Value Length on 2 bytes instead of 4 bytes 2. GDCM 1.2.0 would also
rewrite DataElement as Implicit when the VR would not be known this
would only happen in some very rare cases. gdcm 2.X design could
handle bug #1 or #2 exclusively, this class can now handle file which
have both issues. See: gdcmData/TheralysGDCM120Bug.dcm.

C++ includes: gdcmUNExplicitImplicitDataElement.h */ public class";

%csmethodmodifiers  gdcm::UNExplicitImplicitDataElement::GetLength "
/** VL gdcm::UNExplicitImplicitDataElement::GetLength() const  */
public";

%csmethodmodifiers  gdcm::UNExplicitImplicitDataElement::Read " /**
std::istream& gdcm::UNExplicitImplicitDataElement::Read(std::istream
&is)  */ public";


// File: classgdcm_1_1Unpacker12Bits.xml
%typemap("csclassmodifiers") gdcm::Unpacker12Bits " /** Pack/Unpack 12
bits pixel into 16bits You can only pack an even number of 16bits,
which means a multiple of 4 (expressed in bytes)

You can only unpack a multiple of 3 bytes.

C++ includes: gdcmUnpacker12Bits.h */ public class";


// File: classgdcm_1_1Usage.xml
%typemap("csclassmodifiers") gdcm::Usage " /**  Usage.

A.1.3 IOD Module Table and Functional Group Macro Table This Section
of each IOD defines in a tabular form the Modules comprising the IOD.
The following information must be specified for each Module in the
table: The name of the Module or Functional Group

A reference to the Section in Annex C which defines the Module or
Functional Group

The usage of the Module or Functional Group; whether it is:

Mandatory (see A.1.3.1) , abbreviated M

Conditional (see A.1.3.2) , abbreviated C

User Option (see A.1.3.3) , abbreviated U The Modules referenced are
defined in Annex C. A.1.3.1 MANDATORY MODULES For each IOD, Mandatory
Modules shall be supported per the definitions, semantics and
requirements defined in Annex C.

A.1.3.2 CONDITIONAL MODULES Conditional Modules are Mandatory Modules
if specific conditions are met. If the specified conditions are not
met, this Module shall not be supported; that is, no information
defined in that Module shall be sent. A.1.3.3 USER OPTION MODULES User
Option Modules may or may not be supported. If an optional Module is
supported, the Attribute Types specified in the Modules in Annex C
shall be supported.

C++ includes: gdcmUsage.h */ public class";

%csmethodmodifiers  gdcm::Usage::Usage " /**
gdcm::Usage::Usage(UsageType type=Invalid)  */ public";


// File: classstd_1_1valarray.xml
%typemap("csclassmodifiers") std::valarray " /** STL class.

*/ public class";


// File: classgdcm_1_1Validate.xml
%typemap("csclassmodifiers") gdcm::Validate " /**  Validate class.

C++ includes: gdcmValidate.h */ public class";

%csmethodmodifiers  gdcm::Validate::GetValidatedFile " /** const File&
gdcm::Validate::GetValidatedFile()  */ public";

%csmethodmodifiers  gdcm::Validate::SetFile " /** void
gdcm::Validate::SetFile(File const &f)  */ public";

%csmethodmodifiers  gdcm::Validate::Validate " /**
gdcm::Validate::Validate()  */ public";

%csmethodmodifiers  gdcm::Validate::Validation " /** void
gdcm::Validate::Validation()  */ public";

%csmethodmodifiers  gdcm::Validate::~Validate " /**
gdcm::Validate::~Validate()  */ public";


// File: classgdcm_1_1Value.xml
%typemap("csclassmodifiers") gdcm::Value " /** Class to represent the
value of a Data Element.

VALUE: A component of a Value Field. A Value Field may consist of one
or more of these components.

C++ includes: gdcmValue.h */ public class";

%csmethodmodifiers  gdcm::Value::Clear " /** virtual void
gdcm::Value::Clear()=0  */ public";

%csmethodmodifiers  gdcm::Value::GetLength " /** virtual VL
gdcm::Value::GetLength() const =0  */ public";

%csmethodmodifiers  gdcm::Value::SetLength " /** virtual void
gdcm::Value::SetLength(VL l)=0  */ public";

%csmethodmodifiers  gdcm::Value::Value " /** gdcm::Value::Value()  */
public";

%csmethodmodifiers  gdcm::Value::~Value " /** gdcm::Value::~Value()
*/ public";


// File: classgdcm_1_1ValueIO.xml
%typemap("csclassmodifiers") gdcm::ValueIO " /** Class to represent
the value of a Data Element.

VALUE: A component of a Value Field. A Value Field may consist of one
or more of these components.

C++ includes: gdcmValueIO.h */ public class";


// File: classstd_1_1vector.xml
%typemap("csclassmodifiers") std::vector " /** STL class.

*/ public class";


// File: classstd_1_1vector_1_1const__iterator.xml
%typemap("csclassmodifiers") std::vector::const_iterator " /** STL
iterator class.

*/ public class";


// File: classstd_1_1vector_1_1const__reverse__iterator.xml
%typemap("csclassmodifiers") std::vector::const_reverse_iterator " /**
STL iterator class.

*/ public class";


// File: classstd_1_1vector_1_1iterator.xml
%typemap("csclassmodifiers") std::vector::iterator " /** STL iterator
class.

*/ public class";


// File: classstd_1_1vector_1_1reverse__iterator.xml
%typemap("csclassmodifiers") std::vector::reverse_iterator " /** STL
iterator class.

*/ public class";


// File: classgdcm_1_1Version.xml
%typemap("csclassmodifiers") gdcm::Version " /** major/minor and build
version

C++ includes: gdcmVersion.h */ public class";


// File: classgdcm_1_1VL.xml
%typemap("csclassmodifiers") gdcm::VL " /**  Value Length.

WARNING:  this is a 4bytes value ! Do not try to use it for 2bytes
value length

C++ includes: gdcmVL.h */ public class";

%csmethodmodifiers  gdcm::VL::GetLength " /** VL gdcm::VL::GetLength()
const  */ public";

%csmethodmodifiers  gdcm::VL::IsOdd " /** bool gdcm::VL::IsOdd() const
*/ public";

%csmethodmodifiers  gdcm::VL::IsUndefined " /** bool
gdcm::VL::IsUndefined() const  */ public";

%csmethodmodifiers  gdcm::VL::Read " /** std::istream&
gdcm::VL::Read(std::istream &is)  */ public";

%csmethodmodifiers  gdcm::VL::Read16 " /** std::istream&
gdcm::VL::Read16(std::istream &is)  */ public";

%csmethodmodifiers  gdcm::VL::SetToUndefined " /** void
gdcm::VL::SetToUndefined()  */ public";

%csmethodmodifiers  gdcm::VL::VL " /** gdcm::VL::VL(uint32_t vl=0)  */
public";

%csmethodmodifiers  gdcm::VL::Write " /** const std::ostream&
gdcm::VL::Write(std::ostream &os) const  */ public";

%csmethodmodifiers  gdcm::VL::Write16 " /** const std::ostream&
gdcm::VL::Write16(std::ostream &os) const  */ public";


// File: classgdcm_1_1VM.xml
%typemap("csclassmodifiers") gdcm::VM " /**  Value Multiplicity
Looking at the DICOMV3 dict only there is very few cases: 1 2 3 4 5 6
8 16 24 1-2 1-3 1-8 1-32 1-99 1-n 2-2n 2-n 3-3n 3-n.

Some private dict define some more: 4-4n 1-4 1-5 256 9 3-4

even more:

7-7n 10 18 12 35 47_47n 30_30n 28

6-6n

C++ includes: gdcmVM.h */ public class";

%csmethodmodifiers  gdcm::VM::Compatible " /** bool
gdcm::VM::Compatible(VM const &vm) const

WARNING: Implementation deficiency The Compatible function is poorly
implemented, the reference vm should be coming from the dictionary,
while the passed in value is the value guess from the file.

*/ public";

%csmethodmodifiers  gdcm::VM::GetLength " /** unsigned int
gdcm::VM::GetLength() const  */ public";

%csmethodmodifiers  gdcm::VM::VM " /** gdcm::VM::VM(VMType type=VM0)
*/ public";


// File: classgdcm_1_1VR.xml
%typemap("csclassmodifiers") gdcm::VR " /**  VR class This is adapted
from DICOM standard The biggest difference is the INVALID VR and the
composite one that differ from standard (more like an addition) This
allow us to represent all the possible case express in the DICOMV3
dict.

VALUE REPRESENTATION ( VR) Specifies the data type and format of the
Value(s) contained in the Value Field of a Data Element. VALUE
REPRESENTATION FIELD: The field where the Value Representation of a
Data Element is stored in the encoding of a Data Element structure
with explicit VR.

C++ includes: gdcmVR.h */ public class";

%csmethodmodifiers  gdcm::VR::Compatible " /** bool
gdcm::VR::Compatible(VR const &vr) const  */ public";

%csmethodmodifiers  gdcm::VR::GetLength " /** int
gdcm::VR::GetLength() const  */ public";

%csmethodmodifiers  gdcm::VR::GetSize " /** unsigned int
gdcm::VR::GetSize() const  */ public";

%csmethodmodifiers  gdcm::VR::GetSizeof " /** unsigned int
gdcm::VR::GetSizeof() const  */ public";

%csmethodmodifiers  gdcm::VR::IsDual " /** bool gdcm::VR::IsDual()
const  */ public";

%csmethodmodifiers  gdcm::VR::IsVRFile " /** bool gdcm::VR::IsVRFile()
const  */ public";

%csmethodmodifiers  gdcm::VR::Read " /** std::istream&
gdcm::VR::Read(std::istream &is)  */ public";

%csmethodmodifiers  gdcm::VR::VR " /** gdcm::VR::VR(VRType vr=INVALID)
*/ public";

%csmethodmodifiers  gdcm::VR::Write " /** const std::ostream&
gdcm::VR::Write(std::ostream &os) const  */ public";


// File: classgdcm_1_1VR16ExplicitDataElement.xml
%typemap("csclassmodifiers") gdcm::VR16ExplicitDataElement " /** Class
to read/write a DataElement as Explicit Data Element.

This class support 16 bits when finding an unkown VR: For instance:
Siemens_CT_Sensation64_has_VR_RT.dcm

C++ includes: gdcmVR16ExplicitDataElement.h */ public class";

%csmethodmodifiers  gdcm::VR16ExplicitDataElement::GetLength " /** VL
gdcm::VR16ExplicitDataElement::GetLength() const  */ public";

%csmethodmodifiers  gdcm::VR16ExplicitDataElement::Read " /**
std::istream& gdcm::VR16ExplicitDataElement::Read(std::istream &is)
*/ public";

%csmethodmodifiers  gdcm::VR16ExplicitDataElement::ReadWithLength "
/** std::istream&
gdcm::VR16ExplicitDataElement::ReadWithLength(std::istream &is, VL
&length)  */ public";


// File: classgdcm_1_1VRVLSize_3_010_01_4.xml
%typemap("csclassmodifiers") gdcm::VRVLSize< 0 > " /**C++ includes:
gdcmAttribute.h */ public class";


// File: classgdcm_1_1VRVLSize_3_011_01_4.xml
%typemap("csclassmodifiers") gdcm::VRVLSize< 1 > " /**C++ includes:
gdcmAttribute.h */ public class";


// File: classvtkGDCMImageReader.xml
%typemap("csclassmodifiers") vtkGDCMImageReader " /**C++ includes:
vtkGDCMImageReader.h */ public class";

%csmethodmodifiers  vtkGDCMImageReader::CanReadFile " /** virtual int
vtkGDCMImageReader::CanReadFile(const char *fname)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::GetDescriptiveName " /**
virtual const char* vtkGDCMImageReader::GetDescriptiveName()  */
public";

%csmethodmodifiers  vtkGDCMImageReader::GetFileExtensions " /**
virtual const char* vtkGDCMImageReader::GetFileExtensions()  */
public";

%csmethodmodifiers  vtkGDCMImageReader::GetIconImage " /**
vtkImageData* vtkGDCMImageReader::GetIconImage()  */ public";

%csmethodmodifiers  vtkGDCMImageReader::GetOverlay " /** vtkImageData*
vtkGDCMImageReader::GetOverlay(int i)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::PrintSelf " /** virtual void
vtkGDCMImageReader::PrintSelf(ostream &os, vtkIndent indent)  */
public";

%csmethodmodifiers  vtkGDCMImageReader::SetCurve " /** virtual void
vtkGDCMImageReader::SetCurve(vtkPolyData *pd)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::SetFileNames " /** virtual
void vtkGDCMImageReader::SetFileNames(vtkStringArray *)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkBooleanMacro " /** int
vtkGDCMImageReader::vtkBooleanMacro(ApplyYBRToRGB, int)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkBooleanMacro " /**
vtkGDCMImageReader::vtkBooleanMacro(ApplyLookupTable, int)  */
public";

%csmethodmodifiers  vtkGDCMImageReader::vtkBooleanMacro " /**
vtkGDCMImageReader::vtkBooleanMacro(LossyFlag, int)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkBooleanMacro " /**
vtkGDCMImageReader::vtkBooleanMacro(LoadIconImage, int)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkBooleanMacro " /**
vtkGDCMImageReader::vtkBooleanMacro(LoadOverlays, int)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetMacro " /**
vtkGDCMImageReader::vtkGetMacro(Scale, double)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetMacro " /**
vtkGDCMImageReader::vtkGetMacro(Shift, double)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetMacro " /**
vtkGDCMImageReader::vtkGetMacro(PlanarConfiguration, int)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetMacro " /**
vtkGDCMImageReader::vtkGetMacro(ImageFormat, int)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetMacro " /**
vtkGDCMImageReader::vtkGetMacro(ApplyYBRToRGB, int)
vtkSetMacro(ApplyYBRToRGB  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetMacro " /**
vtkGDCMImageReader::vtkGetMacro(ApplyLookupTable, int)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetMacro " /**
vtkGDCMImageReader::vtkGetMacro(NumberOfIconImages, int)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetMacro " /**
vtkGDCMImageReader::vtkGetMacro(NumberOfOverlays, int)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetMacro " /**
vtkGDCMImageReader::vtkGetMacro(LossyFlag, int)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetMacro " /**
vtkGDCMImageReader::vtkGetMacro(LoadIconImage, int)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetMacro " /**
vtkGDCMImageReader::vtkGetMacro(LoadOverlays, int)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetObjectMacro " /**
vtkGDCMImageReader::vtkGetObjectMacro(Curve, vtkPolyData)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetObjectMacro " /**
vtkGDCMImageReader::vtkGetObjectMacro(FileNames, vtkStringArray)  */
public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetObjectMacro " /**
vtkGDCMImageReader::vtkGetObjectMacro(MedicalImageProperties,
vtkMedicalImageProperties)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetObjectMacro " /**
vtkGDCMImageReader::vtkGetObjectMacro(DirectionCosines, vtkMatrix4x4)
*/ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetVector3Macro " /**
vtkGDCMImageReader::vtkGetVector3Macro(ImagePositionPatient, double)
*/ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkGetVector6Macro " /**
vtkGDCMImageReader::vtkGetVector6Macro(ImageOrientationPatient,
double)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkSetMacro " /**
vtkGDCMImageReader::vtkSetMacro(ApplyLookupTable, int)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkSetMacro " /**
vtkGDCMImageReader::vtkSetMacro(LossyFlag, int)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkSetMacro " /**
vtkGDCMImageReader::vtkSetMacro(LoadIconImage, int)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkSetMacro " /**
vtkGDCMImageReader::vtkSetMacro(LoadOverlays, int)  */ public";

%csmethodmodifiers  vtkGDCMImageReader::vtkTypeRevisionMacro " /**
vtkGDCMImageReader::vtkTypeRevisionMacro(vtkGDCMImageReader,
vtkMedicalImageReader2)  */ public";


// File: classvtkGDCMImageWriter.xml
%typemap("csclassmodifiers") vtkGDCMImageWriter " /**C++ includes:
vtkGDCMImageWriter.h */ public class";

%csmethodmodifiers  vtkGDCMImageWriter::GetDescriptiveName " /**
virtual const char* vtkGDCMImageWriter::GetDescriptiveName()  */
public";

%csmethodmodifiers  vtkGDCMImageWriter::GetFileExtensions " /**
virtual const char* vtkGDCMImageWriter::GetFileExtensions()  */
public";

%csmethodmodifiers  vtkGDCMImageWriter::PrintSelf " /** virtual void
vtkGDCMImageWriter::PrintSelf(ostream &os, vtkIndent indent)  */
public";

%csmethodmodifiers  vtkGDCMImageWriter::SetDirectionCosines " /**
virtual void vtkGDCMImageWriter::SetDirectionCosines(vtkMatrix4x4
*matrix)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::SetFileNames " /** virtual
void vtkGDCMImageWriter::SetFileNames(vtkStringArray *)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::SetMedicalImageProperties "
/** virtual void
vtkGDCMImageWriter::SetMedicalImageProperties(vtkMedicalImageProperties
*)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkBooleanMacro " /**
vtkGDCMImageWriter::vtkBooleanMacro(FileLowerLeft, int)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkBooleanMacro " /**
vtkGDCMImageWriter::vtkBooleanMacro(LossyFlag, int)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkGetMacro " /**
vtkGDCMImageWriter::vtkGetMacro(FileLowerLeft, int)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkGetMacro " /**
vtkGDCMImageWriter::vtkGetMacro(ImageFormat, int)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkGetMacro " /**
vtkGDCMImageWriter::vtkGetMacro(Scale, double)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkGetMacro " /**
vtkGDCMImageWriter::vtkGetMacro(Shift, double)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkGetMacro " /**
vtkGDCMImageWriter::vtkGetMacro(LossyFlag, int)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkGetObjectMacro " /**
vtkGDCMImageWriter::vtkGetObjectMacro(DirectionCosines, vtkMatrix4x4)
*/ public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkGetObjectMacro " /**
vtkGDCMImageWriter::vtkGetObjectMacro(FileNames, vtkStringArray)  */
public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkGetObjectMacro " /**
vtkGDCMImageWriter::vtkGetObjectMacro(MedicalImageProperties,
vtkMedicalImageProperties)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkSetMacro " /**
vtkGDCMImageWriter::vtkSetMacro(PlanarConfiguration, int)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkSetMacro " /**
vtkGDCMImageWriter::vtkSetMacro(FileLowerLeft, int)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkSetMacro " /**
vtkGDCMImageWriter::vtkSetMacro(ImageFormat, int)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkSetMacro " /**
vtkGDCMImageWriter::vtkSetMacro(Scale, double)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkSetMacro " /**
vtkGDCMImageWriter::vtkSetMacro(Shift, double)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkSetMacro " /**
vtkGDCMImageWriter::vtkSetMacro(LossyFlag, int)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::vtkTypeRevisionMacro " /**
vtkGDCMImageWriter::vtkTypeRevisionMacro(vtkGDCMImageWriter,
vtkImageWriter)  */ public";

%csmethodmodifiers  vtkGDCMImageWriter::Write " /** virtual void
vtkGDCMImageWriter::Write()  */ public";


// File: classvtkGDCMPolyDataReader.xml
%typemap("csclassmodifiers") vtkGDCMPolyDataReader " /**C++ includes:
vtkGDCMPolyDataReader.h */ public class";

%csmethodmodifiers  vtkGDCMPolyDataReader::PrintSelf " /** virtual
void vtkGDCMPolyDataReader::PrintSelf(ostream &os, vtkIndent indent)
*/ public";

%csmethodmodifiers  vtkGDCMPolyDataReader::vtkGetObjectMacro " /**
vtkGDCMPolyDataReader::vtkGetObjectMacro(MedicalImageProperties,
vtkMedicalImageProperties)  */ public";

%csmethodmodifiers  vtkGDCMPolyDataReader::vtkGetStringMacro " /**
vtkGDCMPolyDataReader::vtkGetStringMacro(FileName)  */ public";

%csmethodmodifiers  vtkGDCMPolyDataReader::vtkSetStringMacro " /**
vtkGDCMPolyDataReader::vtkSetStringMacro(FileName)  */ public";

%csmethodmodifiers  vtkGDCMPolyDataReader::vtkTypeRevisionMacro " /**
vtkGDCMPolyDataReader::vtkTypeRevisionMacro(vtkGDCMPolyDataReader,
vtkPolyDataAlgorithm)  */ public";


// File: classvtkGDCMThreadedImageReader.xml
%typemap("csclassmodifiers") vtkGDCMThreadedImageReader " /**C++
includes: vtkGDCMThreadedImageReader.h */ public class";

%csmethodmodifiers  vtkGDCMThreadedImageReader::PrintSelf " /**
virtual void vtkGDCMThreadedImageReader::PrintSelf(ostream &os,
vtkIndent indent)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader::vtkBooleanMacro " /**
vtkGDCMThreadedImageReader::vtkBooleanMacro(UseShiftScale, int)  */
public";

%csmethodmodifiers  vtkGDCMThreadedImageReader::vtkGetMacro " /**
vtkGDCMThreadedImageReader::vtkGetMacro(UseShiftScale, int)  */
public";

%csmethodmodifiers  vtkGDCMThreadedImageReader::vtkSetMacro " /**
vtkGDCMThreadedImageReader::vtkSetMacro(UseShiftScale, int)  */
public";

%csmethodmodifiers  vtkGDCMThreadedImageReader::vtkSetMacro " /**
vtkGDCMThreadedImageReader::vtkSetMacro(Scale, double)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader::vtkSetMacro " /**
vtkGDCMThreadedImageReader::vtkSetMacro(Shift, double)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader::vtkTypeRevisionMacro "
/**
vtkGDCMThreadedImageReader::vtkTypeRevisionMacro(vtkGDCMThreadedImageReader,
vtkGDCMImageReader)  */ public";


// File: classvtkGDCMThreadedImageReader2.xml
%typemap("csclassmodifiers") vtkGDCMThreadedImageReader2 " /**C++
includes: vtkGDCMThreadedImageReader2.h */ public class";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::GetFileName " /**
virtual const char* vtkGDCMThreadedImageReader2::GetFileName(int i=0)
*/ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::PrintSelf " /**
virtual void vtkGDCMThreadedImageReader2::PrintSelf(ostream &os,
vtkIndent indent)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::SetFileName " /**
virtual void vtkGDCMThreadedImageReader2::SetFileName(const char
*filename)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::SetFileNames " /**
virtual void vtkGDCMThreadedImageReader2::SetFileNames(vtkStringArray
*)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::SplitExtent " /** int
vtkGDCMThreadedImageReader2::SplitExtent(int splitExt[6], int
startExt[6], int num, int total)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkBooleanMacro " /**
vtkGDCMThreadedImageReader2::vtkBooleanMacro(UseShiftScale, int)  */
public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkBooleanMacro " /**
vtkGDCMThreadedImageReader2::vtkBooleanMacro(LoadOverlays, int)  */
public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkBooleanMacro " /**
vtkGDCMThreadedImageReader2::vtkBooleanMacro(FileLowerLeft, int)  */
public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkGetMacro " /**
vtkGDCMThreadedImageReader2::vtkGetMacro(UseShiftScale, int)  */
public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkGetMacro " /**
vtkGDCMThreadedImageReader2::vtkGetMacro(Scale, double)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkGetMacro " /**
vtkGDCMThreadedImageReader2::vtkGetMacro(Shift, double)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkGetMacro " /**
vtkGDCMThreadedImageReader2::vtkGetMacro(LoadOverlays, int)  */
public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkGetMacro " /**
vtkGDCMThreadedImageReader2::vtkGetMacro(NumberOfScalarComponents,
int)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkGetMacro " /**
vtkGDCMThreadedImageReader2::vtkGetMacro(DataScalarType, int)  */
public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkGetMacro " /**
vtkGDCMThreadedImageReader2::vtkGetMacro(NumberOfOverlays, int)  */
public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkGetMacro " /**
vtkGDCMThreadedImageReader2::vtkGetMacro(FileLowerLeft, int)  */
public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkGetObjectMacro "
/** vtkGDCMThreadedImageReader2::vtkGetObjectMacro(FileNames,
vtkStringArray)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkGetVector3Macro "
/** vtkGDCMThreadedImageReader2::vtkGetVector3Macro(DataSpacing,
double)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkGetVector3Macro "
/** vtkGDCMThreadedImageReader2::vtkGetVector3Macro(DataOrigin,
double)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkGetVector6Macro "
/** vtkGDCMThreadedImageReader2::vtkGetVector6Macro(DataExtent, int)
*/ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkSetMacro " /**
vtkGDCMThreadedImageReader2::vtkSetMacro(UseShiftScale, int)  */
public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkSetMacro " /**
vtkGDCMThreadedImageReader2::vtkSetMacro(Scale, double)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkSetMacro " /**
vtkGDCMThreadedImageReader2::vtkSetMacro(Shift, double)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkSetMacro " /**
vtkGDCMThreadedImageReader2::vtkSetMacro(LoadOverlays, int)  */
public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkSetMacro " /**
vtkGDCMThreadedImageReader2::vtkSetMacro(NumberOfScalarComponents,
int)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkSetMacro " /**
vtkGDCMThreadedImageReader2::vtkSetMacro(DataScalarType, int)  */
public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkSetMacro " /**
vtkGDCMThreadedImageReader2::vtkSetMacro(FileLowerLeft, int)  */
public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkSetVector3Macro "
/** vtkGDCMThreadedImageReader2::vtkSetVector3Macro(DataSpacing,
double)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkSetVector3Macro "
/** vtkGDCMThreadedImageReader2::vtkSetVector3Macro(DataOrigin,
double)  */ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkSetVector6Macro "
/** vtkGDCMThreadedImageReader2::vtkSetVector6Macro(DataExtent, int)
*/ public";

%csmethodmodifiers  vtkGDCMThreadedImageReader2::vtkTypeRevisionMacro
" /**
vtkGDCMThreadedImageReader2::vtkTypeRevisionMacro(vtkGDCMThreadedImageReader2,
vtkThreadedImageAlgorithm)  */ public";


// File: classvtkImageColorViewer.xml
%typemap("csclassmodifiers") vtkImageColorViewer " /**C++ includes:
vtkImageColorViewer.h */ public class";

%csmethodmodifiers  vtkImageColorViewer::AddInput " /** virtual void
vtkImageColorViewer::AddInput(vtkImageData *input)  */ public";

%csmethodmodifiers  vtkImageColorViewer::AddInputConnection " /**
virtual void
vtkImageColorViewer::AddInputConnection(vtkAlgorithmOutput *input)  */
public";

%csmethodmodifiers  vtkImageColorViewer::GetColorLevel " /** virtual
double vtkImageColorViewer::GetColorLevel()  */ public";

%csmethodmodifiers  vtkImageColorViewer::GetColorWindow " /** virtual
double vtkImageColorViewer::GetColorWindow()  */ public";

%csmethodmodifiers  vtkImageColorViewer::GetInput " /** virtual
vtkImageData* vtkImageColorViewer::GetInput()  */ public";

%csmethodmodifiers  vtkImageColorViewer::GetOffScreenRendering " /**
virtual int vtkImageColorViewer::GetOffScreenRendering()  */ public";

%csmethodmodifiers  vtkImageColorViewer::GetOverlayVisibility " /**
double vtkImageColorViewer::GetOverlayVisibility()  */ public";

%csmethodmodifiers  vtkImageColorViewer::GetPosition " /** virtual
int* vtkImageColorViewer::GetPosition()  */ public";

%csmethodmodifiers  vtkImageColorViewer::GetSize " /** virtual int*
vtkImageColorViewer::GetSize()  */ public";

%csmethodmodifiers  vtkImageColorViewer::GetSliceMax " /** virtual int
vtkImageColorViewer::GetSliceMax()  */ public";

%csmethodmodifiers  vtkImageColorViewer::GetSliceMin " /** virtual int
vtkImageColorViewer::GetSliceMin()  */ public";

%csmethodmodifiers  vtkImageColorViewer::GetSliceRange " /** virtual
int* vtkImageColorViewer::GetSliceRange()  */ public";

%csmethodmodifiers  vtkImageColorViewer::GetSliceRange " /** virtual
void vtkImageColorViewer::GetSliceRange(int &min, int &max)  */
public";

%csmethodmodifiers  vtkImageColorViewer::GetSliceRange " /** virtual
void vtkImageColorViewer::GetSliceRange(int range[2])  */ public";

%csmethodmodifiers  vtkImageColorViewer::GetWindowName " /** virtual
const char* vtkImageColorViewer::GetWindowName()  */ public";

%csmethodmodifiers  vtkImageColorViewer::PrintSelf " /** void
vtkImageColorViewer::PrintSelf(ostream &os, vtkIndent indent)  */
public";

%csmethodmodifiers  vtkImageColorViewer::Render " /** virtual void
vtkImageColorViewer::Render(void)  */ public";

%csmethodmodifiers  vtkImageColorViewer::SetColorLevel " /** virtual
void vtkImageColorViewer::SetColorLevel(double s)  */ public";

%csmethodmodifiers  vtkImageColorViewer::SetColorWindow " /** virtual
void vtkImageColorViewer::SetColorWindow(double s)  */ public";

%csmethodmodifiers  vtkImageColorViewer::SetDisplayId " /** virtual
void vtkImageColorViewer::SetDisplayId(void *a)  */ public";

%csmethodmodifiers  vtkImageColorViewer::SetInput " /** virtual void
vtkImageColorViewer::SetInput(vtkImageData *in)  */ public";

%csmethodmodifiers  vtkImageColorViewer::SetInputConnection " /**
virtual void
vtkImageColorViewer::SetInputConnection(vtkAlgorithmOutput *input)  */
public";

%csmethodmodifiers  vtkImageColorViewer::SetOffScreenRendering " /**
virtual void vtkImageColorViewer::SetOffScreenRendering(int)  */
public";

%csmethodmodifiers  vtkImageColorViewer::SetOverlayVisibility " /**
void vtkImageColorViewer::SetOverlayVisibility(double vis)  */
public";

%csmethodmodifiers  vtkImageColorViewer::SetParentId " /** virtual
void vtkImageColorViewer::SetParentId(void *a)  */ public";

%csmethodmodifiers  vtkImageColorViewer::SetPosition " /** virtual
void vtkImageColorViewer::SetPosition(int a[2])  */ public";

%csmethodmodifiers  vtkImageColorViewer::SetPosition " /** virtual
void vtkImageColorViewer::SetPosition(int a, int b)  */ public";

%csmethodmodifiers  vtkImageColorViewer::SetRenderer " /** virtual
void vtkImageColorViewer::SetRenderer(vtkRenderer *arg)  */ public";

%csmethodmodifiers  vtkImageColorViewer::SetRenderWindow " /** virtual
void vtkImageColorViewer::SetRenderWindow(vtkRenderWindow *arg)  */
public";

%csmethodmodifiers  vtkImageColorViewer::SetSize " /** virtual void
vtkImageColorViewer::SetSize(int a[2])  */ public";

%csmethodmodifiers  vtkImageColorViewer::SetSize " /** virtual void
vtkImageColorViewer::SetSize(int a, int b)  */ public";

%csmethodmodifiers  vtkImageColorViewer::SetSlice " /** virtual void
vtkImageColorViewer::SetSlice(int s)  */ public";

%csmethodmodifiers  vtkImageColorViewer::SetSliceOrientation " /**
virtual void vtkImageColorViewer::SetSliceOrientation(int orientation)
*/ public";

%csmethodmodifiers  vtkImageColorViewer::SetSliceOrientationToXY " /**
virtual void vtkImageColorViewer::SetSliceOrientationToXY()  */
public";

%csmethodmodifiers  vtkImageColorViewer::SetSliceOrientationToXZ " /**
virtual void vtkImageColorViewer::SetSliceOrientationToXZ()  */
public";

%csmethodmodifiers  vtkImageColorViewer::SetSliceOrientationToYZ " /**
virtual void vtkImageColorViewer::SetSliceOrientationToYZ()  */
public";

%csmethodmodifiers  vtkImageColorViewer::SetupInteractor " /** virtual
void vtkImageColorViewer::SetupInteractor(vtkRenderWindowInteractor *)
*/ public";

%csmethodmodifiers  vtkImageColorViewer::SetWindowId " /** virtual
void vtkImageColorViewer::SetWindowId(void *a)  */ public";

%csmethodmodifiers  vtkImageColorViewer::UpdateDisplayExtent " /**
virtual void vtkImageColorViewer::UpdateDisplayExtent()  */ public";

%csmethodmodifiers  vtkImageColorViewer::VTK_LEGACY " /**
vtkImageColorViewer::VTK_LEGACY(void SetZSlice(int))  */ public";

%csmethodmodifiers  vtkImageColorViewer::VTK_LEGACY " /**
vtkImageColorViewer::VTK_LEGACY(int GetZSlice())  */ public";

%csmethodmodifiers  vtkImageColorViewer::VTK_LEGACY " /**
vtkImageColorViewer::VTK_LEGACY(int GetWholeZMax())  */ public";

%csmethodmodifiers  vtkImageColorViewer::VTK_LEGACY " /**
vtkImageColorViewer::VTK_LEGACY(int GetWholeZMin())  */ public";

%csmethodmodifiers  vtkImageColorViewer::vtkBooleanMacro " /**
vtkImageColorViewer::vtkBooleanMacro(OffScreenRendering, int)  */
public";

%csmethodmodifiers  vtkImageColorViewer::vtkGetMacro " /**
vtkImageColorViewer::vtkGetMacro(Slice, int)  */ public";

%csmethodmodifiers  vtkImageColorViewer::vtkGetMacro " /**
vtkImageColorViewer::vtkGetMacro(SliceOrientation, int)  */ public";

%csmethodmodifiers  vtkImageColorViewer::vtkGetObjectMacro " /**
vtkImageColorViewer::vtkGetObjectMacro(InteractorStyle,
vtkInteractorStyleImage)  */ public";

%csmethodmodifiers  vtkImageColorViewer::vtkGetObjectMacro " /**
vtkImageColorViewer::vtkGetObjectMacro(WindowLevel,
vtkImageMapToWindowLevelColors2)  */ public";

%csmethodmodifiers  vtkImageColorViewer::vtkGetObjectMacro " /**
vtkImageColorViewer::vtkGetObjectMacro(ImageActor, vtkImageActor)  */
public";

%csmethodmodifiers  vtkImageColorViewer::vtkGetObjectMacro " /**
vtkImageColorViewer::vtkGetObjectMacro(Renderer, vtkRenderer)  */
public";

%csmethodmodifiers  vtkImageColorViewer::vtkGetObjectMacro " /**
vtkImageColorViewer::vtkGetObjectMacro(RenderWindow, vtkRenderWindow)
*/ public";

%csmethodmodifiers  vtkImageColorViewer::vtkTypeRevisionMacro " /**
vtkImageColorViewer::vtkTypeRevisionMacro(vtkImageColorViewer,
vtkObject)  */ public";


// File: classvtkImageMapToColors16.xml
%typemap("csclassmodifiers") vtkImageMapToColors16 " /**C++ includes:
vtkImageMapToColors16.h */ public class";

%csmethodmodifiers  vtkImageMapToColors16::GetMTime " /** virtual
unsigned long vtkImageMapToColors16::GetMTime()  */ public";

%csmethodmodifiers  vtkImageMapToColors16::PrintSelf " /** void
vtkImageMapToColors16::PrintSelf(ostream &os, vtkIndent indent)  */
public";

%csmethodmodifiers  vtkImageMapToColors16::SetLookupTable " /**
virtual void vtkImageMapToColors16::SetLookupTable(vtkScalarsToColors
*)  */ public";

%csmethodmodifiers  vtkImageMapToColors16::SetOutputFormatToLuminance
" /** void vtkImageMapToColors16::SetOutputFormatToLuminance()  */
public";

%csmethodmodifiers
vtkImageMapToColors16::SetOutputFormatToLuminanceAlpha " /** void
vtkImageMapToColors16::SetOutputFormatToLuminanceAlpha()  */ public";

%csmethodmodifiers  vtkImageMapToColors16::SetOutputFormatToRGB " /**
void vtkImageMapToColors16::SetOutputFormatToRGB()  */ public";

%csmethodmodifiers  vtkImageMapToColors16::SetOutputFormatToRGBA " /**
void vtkImageMapToColors16::SetOutputFormatToRGBA()  */ public";

%csmethodmodifiers  vtkImageMapToColors16::vtkBooleanMacro " /**
vtkImageMapToColors16::vtkBooleanMacro(PassAlphaToOutput, int)  */
public";

%csmethodmodifiers  vtkImageMapToColors16::vtkGetMacro " /**
vtkImageMapToColors16::vtkGetMacro(PassAlphaToOutput, int)  */
public";

%csmethodmodifiers  vtkImageMapToColors16::vtkGetMacro " /**
vtkImageMapToColors16::vtkGetMacro(ActiveComponent, int)  */ public";

%csmethodmodifiers  vtkImageMapToColors16::vtkGetMacro " /**
vtkImageMapToColors16::vtkGetMacro(OutputFormat, int)  */ public";

%csmethodmodifiers  vtkImageMapToColors16::vtkGetObjectMacro " /**
vtkImageMapToColors16::vtkGetObjectMacro(LookupTable,
vtkScalarsToColors)  */ public";

%csmethodmodifiers  vtkImageMapToColors16::vtkSetMacro " /**
vtkImageMapToColors16::vtkSetMacro(PassAlphaToOutput, int)  */
public";

%csmethodmodifiers  vtkImageMapToColors16::vtkSetMacro " /**
vtkImageMapToColors16::vtkSetMacro(ActiveComponent, int)  */ public";

%csmethodmodifiers  vtkImageMapToColors16::vtkSetMacro " /**
vtkImageMapToColors16::vtkSetMacro(OutputFormat, int)  */ public";

%csmethodmodifiers  vtkImageMapToColors16::vtkTypeRevisionMacro " /**
vtkImageMapToColors16::vtkTypeRevisionMacro(vtkImageMapToColors16,
vtkThreadedImageAlgorithm)  */ public";


// File: classvtkImageMapToWindowLevelColors2.xml
%typemap("csclassmodifiers") vtkImageMapToWindowLevelColors2 " /**C++
includes: vtkImageMapToWindowLevelColors2.h */ public class";

%csmethodmodifiers  vtkImageMapToWindowLevelColors2::PrintSelf " /**
void vtkImageMapToWindowLevelColors2::PrintSelf(ostream &os, vtkIndent
indent)  */ public";

%csmethodmodifiers  vtkImageMapToWindowLevelColors2::vtkGetMacro " /**
vtkImageMapToWindowLevelColors2::vtkGetMacro(Level, double)  */
public";

%csmethodmodifiers  vtkImageMapToWindowLevelColors2::vtkGetMacro " /**
vtkImageMapToWindowLevelColors2::vtkGetMacro(Window, double)  */
public";

%csmethodmodifiers  vtkImageMapToWindowLevelColors2::vtkSetMacro " /**
vtkImageMapToWindowLevelColors2::vtkSetMacro(Level, double)  */
public";

%csmethodmodifiers  vtkImageMapToWindowLevelColors2::vtkSetMacro " /**
vtkImageMapToWindowLevelColors2::vtkSetMacro(Window, double)  */
public";

%csmethodmodifiers
vtkImageMapToWindowLevelColors2::vtkTypeRevisionMacro " /**
vtkImageMapToWindowLevelColors2::vtkTypeRevisionMacro(vtkImageMapToWindowLevelColors2,
vtkImageMapToColors)  */ public";


// File: classvtkImagePlanarComponentsToComponents.xml
%typemap("csclassmodifiers") vtkImagePlanarComponentsToComponents "
/**C++ includes: vtkImagePlanarComponentsToComponents.h */ public
class";

%csmethodmodifiers  vtkImagePlanarComponentsToComponents::PrintSelf "
/** void vtkImagePlanarComponentsToComponents::PrintSelf(ostream &os,
vtkIndent indent)  */ public";

%csmethodmodifiers
vtkImagePlanarComponentsToComponents::vtkTypeRevisionMacro " /**
vtkImagePlanarComponentsToComponents::vtkTypeRevisionMacro(vtkImagePlanarComponentsToComponents,
vtkImageAlgorithm)  */ public";


// File: classvtkImageRGBToYBR.xml
%typemap("csclassmodifiers") vtkImageRGBToYBR " /**C++ includes:
vtkImageRGBToYBR.h */ public class";

%csmethodmodifiers  vtkImageRGBToYBR::PrintSelf " /** void
vtkImageRGBToYBR::PrintSelf(ostream &os, vtkIndent indent)  */
public";

%csmethodmodifiers  vtkImageRGBToYBR::vtkTypeRevisionMacro " /**
vtkImageRGBToYBR::vtkTypeRevisionMacro(vtkImageRGBToYBR,
vtkThreadedImageAlgorithm)  */ public";


// File: classvtkImageYBRToRGB.xml
%typemap("csclassmodifiers") vtkImageYBRToRGB " /**C++ includes:
vtkImageYBRToRGB.h */ public class";

%csmethodmodifiers  vtkImageYBRToRGB::PrintSelf " /** void
vtkImageYBRToRGB::PrintSelf(ostream &os, vtkIndent indent)  */
public";

%csmethodmodifiers  vtkImageYBRToRGB::vtkTypeRevisionMacro " /**
vtkImageYBRToRGB::vtkTypeRevisionMacro(vtkImageYBRToRGB,
vtkThreadedImageAlgorithm)  */ public";


// File: classvtkLookupTable16.xml
%typemap("csclassmodifiers") vtkLookupTable16 " /**C++ includes:
vtkLookupTable16.h */ public class";

%csmethodmodifiers  vtkLookupTable16::Build " /** void
vtkLookupTable16::Build()  */ public";

%csmethodmodifiers  vtkLookupTable16::GetPointer " /** unsigned short*
vtkLookupTable16::GetPointer(const vtkIdType id)  */ public";

%csmethodmodifiers  vtkLookupTable16::PrintSelf " /** void
vtkLookupTable16::PrintSelf(ostream &os, vtkIndent indent)  */
public";

%csmethodmodifiers  vtkLookupTable16::SetNumberOfTableValues " /**
void vtkLookupTable16::SetNumberOfTableValues(vtkIdType number)  */
public";

%csmethodmodifiers  vtkLookupTable16::vtkTypeRevisionMacro " /**
vtkLookupTable16::vtkTypeRevisionMacro(vtkLookupTable16,
vtkLookupTable)  */ public";

%csmethodmodifiers  vtkLookupTable16::WritePointer " /** unsigned char
* vtkLookupTable16::WritePointer(const vtkIdType id, const int number)
*/ public";


// File: classgdcm_1_1Waveform.xml
%typemap("csclassmodifiers") gdcm::Waveform " /**  Waveform class.

C++ includes: gdcmWaveform.h */ public class";

%csmethodmodifiers  gdcm::Waveform::Waveform " /**
gdcm::Waveform::Waveform()  */ public";


// File: classstd_1_1wfstream.xml
%typemap("csclassmodifiers") std::wfstream " /** STL class.

*/ public class";


// File: classstd_1_1wifstream.xml
%typemap("csclassmodifiers") std::wifstream " /** STL class.

*/ public class";


// File: classstd_1_1wios.xml
%typemap("csclassmodifiers") std::wios " /** STL class.

*/ public class";


// File: classstd_1_1wistream.xml
%typemap("csclassmodifiers") std::wistream " /** STL class.

*/ public class";


// File: classstd_1_1wistringstream.xml
%typemap("csclassmodifiers") std::wistringstream " /** STL class.

*/ public class";


// File: classstd_1_1wofstream.xml
%typemap("csclassmodifiers") std::wofstream " /** STL class.

*/ public class";


// File: classstd_1_1wostream.xml
%typemap("csclassmodifiers") std::wostream " /** STL class.

*/ public class";


// File: classstd_1_1wostringstream.xml
%typemap("csclassmodifiers") std::wostringstream " /** STL class.

*/ public class";


// File: classgdcm_1_1Writer.xml
%typemap("csclassmodifiers") gdcm::Writer " /**  Writer ala DOM
(Document Object Model) This class is a non-validating writer, it will
only performs well- formedness check only.

Detailled description here To avoid GDCM being yet another broken
DICOM lib we try to be user level and avoid writing illegal stuff (odd
length, non-zero value for Item start/end length ...) Therefore you
cannot (well unless you are really smart) write DICOM with even length
tag. All the checks are consider basics: Correct Meta Information
Header (see gdcm::FileMetaInformation)

Zero value for Item Length (0xfffe, 0xe00d/0xe0dd)

Even length for any elements

Alphabetical order for elements (garanteed by design of internals)

32bits VR will be rewritten with 00

WARNING: gdcm::Writer cannot write a DataSet if no SOP Instance UID
(0008,0018) is found

C++ includes: gdcmWriter.h */ public class";

%csmethodmodifiers  gdcm::Writer::CheckFileMetaInformationOff " /**
void gdcm::Writer::CheckFileMetaInformationOff()  */ public";

%csmethodmodifiers  gdcm::Writer::CheckFileMetaInformationOn " /**
void gdcm::Writer::CheckFileMetaInformationOn()  */ public";

%csmethodmodifiers  gdcm::Writer::GetFile " /** File&
gdcm::Writer::GetFile()  */ public";

%csmethodmodifiers  gdcm::Writer::SetCheckFileMetaInformation " /**
void gdcm::Writer::SetCheckFileMetaInformation(bool b)

Undocumented function, do not use (= leave default).

*/ public";

%csmethodmodifiers  gdcm::Writer::SetFile " /** void
gdcm::Writer::SetFile(const File &f)

Set/Get the DICOM file ( DataSet + Header).

*/ public";

%csmethodmodifiers  gdcm::Writer::SetFileName " /** void
gdcm::Writer::SetFileName(const char *filename)

Set the filename of DICOM file to write:.

*/ public";

%csmethodmodifiers  gdcm::Writer::SetStream " /** void
gdcm::Writer::SetStream(std::ostream &output_stream)

Set user ostream buffer.

*/ public";

%csmethodmodifiers  gdcm::Writer::Write " /** virtual bool
gdcm::Writer::Write()

Main function to tell the writer to write.

*/ public";

%csmethodmodifiers  gdcm::Writer::Writer " /** gdcm::Writer::Writer()
*/ public";

%csmethodmodifiers  gdcm::Writer::~Writer " /** virtual
gdcm::Writer::~Writer()  */ public";


// File: classstd_1_1wstring.xml
%typemap("csclassmodifiers") std::wstring " /** STL class.

*/ public class";


// File: classstd_1_1wstring_1_1const__iterator.xml
%typemap("csclassmodifiers") std::wstring::const_iterator " /** STL
iterator class.

*/ public class";


// File: classstd_1_1wstring_1_1const__reverse__iterator.xml
%typemap("csclassmodifiers") std::wstring::const_reverse_iterator "
/** STL iterator class.

*/ public class";


// File: classstd_1_1wstring_1_1iterator.xml
%typemap("csclassmodifiers") std::wstring::iterator " /** STL iterator
class.

*/ public class";


// File: classstd_1_1wstring_1_1reverse__iterator.xml
%typemap("csclassmodifiers") std::wstring::reverse_iterator " /** STL
iterator class.

*/ public class";


// File: classstd_1_1wstringstream.xml
%typemap("csclassmodifiers") std::wstringstream " /** STL class.

*/ public class";


// File: classgdcm_1_1X509.xml
%typemap("csclassmodifiers") gdcm::X509 " /**C++ includes: gdcmX509.h
*/ public class";

%csmethodmodifiers  gdcm::X509::GetNumberOfRecipients " /** unsigned
int gdcm::X509::GetNumberOfRecipients() const  */ public";

%csmethodmodifiers  gdcm::X509::ParseCertificateFile " /** bool
gdcm::X509::ParseCertificateFile(const char *filename)  */ public";

%csmethodmodifiers  gdcm::X509::ParseKeyFile " /** bool
gdcm::X509::ParseKeyFile(const char *filename)  */ public";

%csmethodmodifiers  gdcm::X509::X509 " /** gdcm::X509::X509()  */
public";

%csmethodmodifiers  gdcm::X509::~X509 " /** gdcm::X509::~X509()  */
public";


// File: classgdcm_1_1XMLDictReader.xml
%typemap("csclassmodifiers") gdcm::XMLDictReader " /** Class for
representing a XMLDictReader.

bla Will read the DICOMV3.xml file

C++ includes: gdcmXMLDictReader.h */ public class";

%csmethodmodifiers  gdcm::XMLDictReader::CharacterDataHandler " /**
void gdcm::XMLDictReader::CharacterDataHandler(const char *data, int
length)  */ public";

%csmethodmodifiers  gdcm::XMLDictReader::EndElement " /** void
gdcm::XMLDictReader::EndElement(const char *name)  */ public";

%csmethodmodifiers  gdcm::XMLDictReader::GetDict " /** const Dict&
gdcm::XMLDictReader::GetDict()  */ public";

%csmethodmodifiers  gdcm::XMLDictReader::StartElement " /** void
gdcm::XMLDictReader::StartElement(const char *name, const char **atts)
*/ public";

%csmethodmodifiers  gdcm::XMLDictReader::XMLDictReader " /**
gdcm::XMLDictReader::XMLDictReader()  */ public";

%csmethodmodifiers  gdcm::XMLDictReader::~XMLDictReader " /**
gdcm::XMLDictReader::~XMLDictReader()  */ public";


// File: classgdcm_1_1XMLPrivateDictReader.xml
%typemap("csclassmodifiers") gdcm::XMLPrivateDictReader " /** Class
for representing a XMLPrivateDictReader.

bla Will read the Private.xml file

C++ includes: gdcmXMLPrivateDictReader.h */ public class";

%csmethodmodifiers  gdcm::XMLPrivateDictReader::CharacterDataHandler "
/** void gdcm::XMLPrivateDictReader::CharacterDataHandler(const char
*data, int length)  */ public";

%csmethodmodifiers  gdcm::XMLPrivateDictReader::EndElement " /** void
gdcm::XMLPrivateDictReader::EndElement(const char *name)  */ public";

%csmethodmodifiers  gdcm::XMLPrivateDictReader::GetPrivateDict " /**
const PrivateDict& gdcm::XMLPrivateDictReader::GetPrivateDict()  */
public";

%csmethodmodifiers  gdcm::XMLPrivateDictReader::StartElement " /**
void gdcm::XMLPrivateDictReader::StartElement(const char *name, const
char **atts)  */ public";

%csmethodmodifiers  gdcm::XMLPrivateDictReader::XMLPrivateDictReader "
/** gdcm::XMLPrivateDictReader::XMLPrivateDictReader()  */ public";

%csmethodmodifiers  gdcm::XMLPrivateDictReader::~XMLPrivateDictReader
" /** gdcm::XMLPrivateDictReader::~XMLPrivateDictReader()  */ public";


// File: namespacegdcm.xml
%csmethodmodifiers  gdcm::terminal::to_string " std::string
gdcm::to_string(Float data)  */ public";

%csmethodmodifiers  gdcm::terminal::TYPETOENCODING "
gdcm::TYPETOENCODING(SQ, VRBINARY, unsigned char) TYPETOENCODING(UN
*/ public";


// File: namespacegdcm_1_1terminal.xml
%csmethodmodifiers  gdcm::terminal::setattribute " GDCM_EXPORT
std::string gdcm::terminal::setattribute(Attribute att)  */ public";

%csmethodmodifiers  gdcm::terminal::setbgcolor " GDCM_EXPORT
std::string gdcm::terminal::setbgcolor(Color c)  */ public";

%csmethodmodifiers  gdcm::terminal::setfgcolor " GDCM_EXPORT
std::string gdcm::terminal::setfgcolor(Color c)  */ public";

%csmethodmodifiers  gdcm::terminal::setmode " GDCM_EXPORT void
gdcm::terminal::setmode(Mode m)  */ public";


// File: namespaceitk.xml


// File: namespaceopenssl.xml


// File: namespacestd.xml


// File: namespacezlib__stream.xml
%csmethodmodifiers  zlib_stream::detail::isGZip " bool
zlib_stream::isGZip(std::istream &is)

A typedef for basic_zip_ostream<wchar_t>.

A typedef for basic_zip_istream<wchart> Helper function to check
whether stream is compressed or not.

*/ public";


// File: namespacezlib__stream_1_1detail.xml


// File: gdcm2vtk_8man.xml


// File: gdcmAES_8h.xml


// File: gdcmanon_8man.xml


// File: gdcmAnonymizer_8h.xml


// File: gdcmApplicationEntity_8h.xml


// File: gdcmASN1_8h.xml


// File: gdcmAttribute_8h.xml


// File: gdcmAudioCodec_8h.xml


// File: gdcmBase64_8h.xml


// File: gdcmBasicOffsetTable_8h.xml


// File: gdcmBitmap_8h.xml


// File: gdcmByteBuffer_8h.xml


// File: gdcmByteSwap_8h.xml


// File: gdcmByteSwapFilter_8h.xml


// File: gdcmByteValue_8h.xml


// File: gdcmCodec_8h.xml


// File: gdcmCoder_8h.xml


// File: gdcmConstCharWrapper_8h.xml


// File: gdcmconv_8man.xml


// File: gdcmCP246ExplicitDataElement_8h.xml


// File: gdcmCSAElement_8h.xml


// File: gdcmCSAHeader_8h.xml


// File: gdcmCSAHeaderDict_8h.xml


// File: gdcmCSAHeaderDictEntry_8h.xml


// File: gdcmCurve_8h.xml


// File: gdcmDataElement_8h.xml


// File: gdcmDataSet_8h.xml


// File: gdcmDataSetHelper_8h.xml


// File: gdcmDecoder_8h.xml


// File: gdcmDefinedTerms_8h.xml


// File: gdcmDeflateStream_8h.xml


// File: gdcmDefs_8h.xml


// File: gdcmDeltaEncodingCodec_8h.xml


// File: gdcmDICOMDIR_8h.xml


// File: gdcmDict_8h.xml


// File: gdcmDictConverter_8h.xml


// File: gdcmDictEntry_8h.xml


// File: gdcmDictPrinter_8h.xml


// File: gdcmDicts_8h.xml


// File: gdcmDirectionCosines_8h.xml


// File: gdcmDirectory_8h.xml


// File: gdcmDummyValueGenerator_8h.xml


// File: gdcmdump_8man.xml


// File: gdcmDumper_8h.xml


// File: gdcmElement_8h.xml


// File: gdcmEncapsulatedDocument_8h.xml


// File: gdcmEnumeratedValues_8h.xml


// File: gdcmException_8h.xml


// File: gdcmExplicitDataElement_8h.xml


// File: gdcmExplicitImplicitDataElement_8h.xml


// File: gdcmFiducials_8h.xml


// File: gdcmFile_8h.xml


// File: gdcmFileExplicitFilter_8h.xml


// File: gdcmFileMetaInformation_8h.xml


// File: gdcmFilename_8h.xml


// File: gdcmFilenameGenerator_8h.xml


// File: gdcmFileSet_8h.xml


// File: gdcmFragment_8h.xml


// File: gdcmGlobal_8h.xml


// File: gdcmGroupDict_8h.xml


// File: gdcmHAVEGE_8h.xml


// File: gdcmIconImage_8h.xml


// File: gdcmImage_8h.xml


// File: gdcmImageApplyLookupTable_8h.xml


// File: gdcmImageChangePhotometricInterpretation_8h.xml


// File: gdcmImageChangePlanarConfiguration_8h.xml


// File: gdcmImageChangeTransferSyntax_8h.xml


// File: gdcmImageCodec_8h.xml


// File: gdcmImageConverter_8h.xml


// File: gdcmImageFragmentSplitter_8h.xml


// File: gdcmImageHelper_8h.xml


// File: gdcmImageReader_8h.xml


// File: gdcmImageToImageFilter_8h.xml


// File: gdcmImageWriter_8h.xml


// File: gdcmimg_8man.xml


// File: gdcmImplicitDataElement_8h.xml


// File: gdcminfo_8man.xml


// File: gdcmIOD_8h.xml


// File: gdcmIODEntry_8h.xml


// File: gdcmIODs_8h.xml


// File: gdcmIPPSorter_8h.xml


// File: gdcmItem_8h.xml


// File: gdcmJPEG12Codec_8h.xml


// File: gdcmJPEG16Codec_8h.xml


// File: gdcmJPEG2000Codec_8h.xml


// File: gdcmJPEG8Codec_8h.xml


// File: gdcmJPEGCodec_8h.xml


// File: gdcmJPEGLSCodec_8h.xml


// File: gdcmLO_8h.xml


// File: gdcmLookupTable_8h.xml


// File: gdcmMD5_8h.xml


// File: gdcmMediaStorage_8h.xml


// File: gdcmModule_8h.xml


// File: gdcmModuleEntry_8h.xml


// File: gdcmModules_8h.xml


// File: gdcmNestedModuleEntries_8h.xml


// File: gdcmObject_8h.xml


// File: gdcmOrientation_8h.xml


// File: gdcmOverlay_8h.xml


// File: gdcmParseException_8h.xml


// File: gdcmParser_8h.xml


// File: gdcmPatient_8h.xml


// File: gdcmPDBElement_8h.xml


// File: gdcmPDBHeader_8h.xml


// File: gdcmpdf_8man.xml


// File: gdcmPDFCodec_8h.xml


// File: gdcmPersonName_8h.xml


// File: gdcmPhotometricInterpretation_8h.xml


// File: gdcmPixelFormat_8h.xml


// File: gdcmPixmap_8h.xml


// File: gdcmPixmapReader_8h.xml


// File: gdcmPixmapToPixmapFilter_8h.xml


// File: gdcmPixmapWriter_8h.xml


// File: gdcmPKCS7_8h.xml


// File: gdcmPNMCodec_8h.xml


// File: gdcmPreamble_8h.xml


// File: gdcmPrinter_8h.xml


// File: gdcmPrivateTag_8h.xml


// File: gdcmPVRGCodec_8h.xml


// File: gdcmPythonFilter_8h.xml


// File: gdcmraw_8man.xml


// File: gdcmRAWCodec_8h.xml


// File: gdcmReader_8h.xml


// File: gdcmRescaler_8h.xml


// File: gdcmRLECodec_8h.xml


// File: gdcmRSA_8h.xml


// File: gdcmScanner_8h.xml


// File: gdcmscanner_8man.xml


// File: gdcmSegmentedPaletteColorLookupTable_8h.xml


// File: gdcmSequenceOfFragments_8h.xml


// File: gdcmSequenceOfItems_8h.xml


// File: gdcmSerieHelper_8h.xml


// File: gdcmSeries_8h.xml


// File: gdcmSHA1_8h.xml


// File: gdcmSmartPointer_8h.xml


// File: gdcmSOPClassUIDToIOD_8h.xml


// File: gdcmSorter_8h.xml


// File: gdcmSpacing_8h.xml


// File: gdcmSpectroscopy_8h.xml


// File: gdcmSplitMosaicFilter_8h.xml


// File: gdcmStaticAssert_8h.xml


// File: gdcmString_8h.xml


// File: gdcmStringFilter_8h.xml


// File: gdcmStudy_8h.xml


// File: gdcmSwapCode_8h.xml


// File: gdcmSwapper_8h.xml


// File: gdcmSystem_8h.xml


// File: gdcmTable_8h.xml


// File: gdcmTableEntry_8h.xml


// File: gdcmTableReader_8h.xml


// File: gdcmTag_8h.xml


// File: gdcmTagPath_8h.xml


// File: gdcmtar_8man.xml


// File: gdcmTerminal_8h.xml


// File: gdcmTesting_8h.xml


// File: gdcmTrace_8h.xml


// File: gdcmTransferSyntax_8h.xml


// File: gdcmType_8h.xml


// File: gdcmTypes_8h.xml


// File: gdcmUIDGenerator_8h.xml


// File: gdcmUIDs_8h.xml


// File: gdcmUNExplicitDataElement_8h.xml


// File: gdcmUNExplicitImplicitDataElement_8h.xml


// File: gdcmUnpacker12Bits_8h.xml


// File: gdcmUsage_8h.xml


// File: gdcmValidate_8h.xml


// File: gdcmValue_8h.xml


// File: gdcmValueIO_8h.xml


// File: gdcmVersion_8h.xml


// File: gdcmviewer_8man.xml


// File: gdcmVL_8h.xml


// File: gdcmVM_8h.xml


// File: gdcmVR_8h.xml


// File: gdcmVR16ExplicitDataElement_8h.xml


// File: gdcmWaveform_8h.xml


// File: gdcmWin32_8h.xml


// File: gdcmWriter_8h.xml


// File: gdcmX509_8h.xml


// File: gdcmXMLDictReader_8h.xml


// File: gdcmXMLPrivateDictReader_8h.xml


// File: itkGDCMImageIO2_8h.xml


// File: README_8txt.xml


// File: vtkGDCMImageReader_8h.xml


// File: vtkGDCMImageWriter_8h.xml


// File: vtkGDCMPolyDataReader_8h.xml


// File: vtkGDCMThreadedImageReader_8h.xml


// File: vtkGDCMThreadedImageReader2_8h.xml


// File: vtkImageColorViewer_8h.xml


// File: vtkImageMapToColors16_8h.xml


// File: vtkImageMapToWindowLevelColors2_8h.xml


// File: vtkImagePlanarComponentsToComponents_8h.xml


// File: vtkImageRGBToYBR_8h.xml


// File: vtkImageYBRToRGB_8h.xml


// File: vtkLookupTable16_8h.xml


// File: zipstreamimpl_8h.xml


// File: gdcm2vtk.xml


// File: gdcmanon.xml


// File: gdcmconv.xml


// File: gdcmdump.xml


// File: gdcmimg.xml


// File: gdcminfo.xml


// File: gdcmpdf.xml


// File: gdcmraw.xml


// File: gdcmscanner.xml


// File: gdcmtar.xml


// File: gdcmviewer.xml


// File: todo.xml


// File: deprecated.xml


// File: dir_2f925b9d7bdbc7cb71effd7da9c163ec.xml


// File: dir_47bd3784b8141d2b2b57bce8132141f0.xml


// File: dir_e1b815c71b4736e0245897636405935d.xml


// File: dir_1dd5d3a73373a39d9f89a45b28af9882.xml


// File: dir_98f95006cf3d01c03f5f1dcd2796cea1.xml


// File: dir_475b1e2b8823f4954ce71ff73c30d053.xml


// File: dir_7ab25f9b685d07ed13e36a824772d506.xml


// File: dir_a588aec3e82fcf93c169e09facea5e45.xml


// File: dir_4aa7a2573b912138c18e1d44a1da3f20.xml


// File: dir_f6cd53f06c8d451cb23b075f1b152ad4.xml


// File: dir_c6ef5dad3eb4614ee3542d4c8cc27653.xml


// File: dir_abed4454057e3046a7830aeb372c696f.xml


// File: dir_4abd1d05a373ee18e10fafc144268c02.xml


// File: dir_4a9b39e6c7867f3f2bccc1f0ae75dad7.xml


// File: DecompressImage_8cs-example.xml


// File: DecompressJPEGFile_8cs-example.xml


// File: ManipulateFile_8cs-example.xml


// File: PatchFile_8cxx-example.xml


// File: ScanDirectory_8cs-example.xml


// File: TestByteSwap_8cxx-example.xml


// File: TestCSharpFilter_8cs-example.xml


// File: TestReader_8cxx-example.xml


// File: TestReader_8py-example.xml


// File: indexpage.xml
