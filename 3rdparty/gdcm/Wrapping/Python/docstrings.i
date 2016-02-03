
// File: index.xml

// File: classgdcm_1_1network_1_1AAbortPDU.xml
%feature("docstring") gdcm::network::AAbortPDU "

AAbortPDU Table 9-26 A-ABORT PDU FIELDS.

C++ includes: gdcmAAbortPDU.h ";

%feature("docstring")  gdcm::network::AAbortPDU::AAbortPDU "gdcm::network::AAbortPDU::AAbortPDU() ";

%feature("docstring")  gdcm::network::AAbortPDU::IsLastFragment "bool
gdcm::network::AAbortPDU::IsLastFragment() const ";

%feature("docstring")  gdcm::network::AAbortPDU::Print "void
gdcm::network::AAbortPDU::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::network::AAbortPDU::Read "std::istream&
gdcm::network::AAbortPDU::Read(std::istream &is) ";

%feature("docstring")  gdcm::network::AAbortPDU::SetReason "void
gdcm::network::AAbortPDU::SetReason(const uint8_t r) ";

%feature("docstring")  gdcm::network::AAbortPDU::SetSource "void
gdcm::network::AAbortPDU::SetSource(const uint8_t s) ";

%feature("docstring")  gdcm::network::AAbortPDU::Size "size_t
gdcm::network::AAbortPDU::Size() const ";

%feature("docstring")  gdcm::network::AAbortPDU::Write "const
std::ostream& gdcm::network::AAbortPDU::Write(std::ostream &os) const
";


// File: classgdcm_1_1network_1_1AAssociateACPDU.xml
%feature("docstring") gdcm::network::AAssociateACPDU "

AAssociateACPDU Table 9-17 ASSOCIATE-AC PDU fields.

C++ includes: gdcmAAssociateACPDU.h ";

%feature("docstring")  gdcm::network::AAssociateACPDU::AAssociateACPDU
"gdcm::network::AAssociateACPDU::AAssociateACPDU() ";

%feature("docstring")
gdcm::network::AAssociateACPDU::AddPresentationContextAC "void
gdcm::network::AAssociateACPDU::AddPresentationContextAC(PresentationContextAC
const &pcac) ";

%feature("docstring")
gdcm::network::AAssociateACPDU::GetNumberOfPresentationContextAC "SizeType
gdcm::network::AAssociateACPDU::GetNumberOfPresentationContextAC()
const ";

%feature("docstring")
gdcm::network::AAssociateACPDU::GetPresentationContextAC "const
PresentationContextAC&
gdcm::network::AAssociateACPDU::GetPresentationContextAC(SizeType i)
";

%feature("docstring")
gdcm::network::AAssociateACPDU::GetUserInformation "const
UserInformation& gdcm::network::AAssociateACPDU::GetUserInformation()
const ";

%feature("docstring")  gdcm::network::AAssociateACPDU::InitFromRQ "void gdcm::network::AAssociateACPDU::InitFromRQ(AAssociateRQPDU const
&rqpdu) ";

%feature("docstring")  gdcm::network::AAssociateACPDU::IsLastFragment
"bool gdcm::network::AAssociateACPDU::IsLastFragment() const ";

%feature("docstring")  gdcm::network::AAssociateACPDU::Print "void
gdcm::network::AAssociateACPDU::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::network::AAssociateACPDU::Read "std::istream& gdcm::network::AAssociateACPDU::Read(std::istream &is)
";

%feature("docstring")  gdcm::network::AAssociateACPDU::Size "SizeType
gdcm::network::AAssociateACPDU::Size() const ";

%feature("docstring")  gdcm::network::AAssociateACPDU::Write "const
std::ostream& gdcm::network::AAssociateACPDU::Write(std::ostream &os)
const ";


// File: classgdcm_1_1network_1_1AAssociateRJPDU.xml
%feature("docstring") gdcm::network::AAssociateRJPDU "

AAssociateRJPDU Table 9-21 ASSOCIATE-RJ PDU FIELDS.

C++ includes: gdcmAAssociateRJPDU.h ";

%feature("docstring")  gdcm::network::AAssociateRJPDU::AAssociateRJPDU
"gdcm::network::AAssociateRJPDU::AAssociateRJPDU() ";

%feature("docstring")  gdcm::network::AAssociateRJPDU::IsLastFragment
"bool gdcm::network::AAssociateRJPDU::IsLastFragment() const ";

%feature("docstring")  gdcm::network::AAssociateRJPDU::Print "void
gdcm::network::AAssociateRJPDU::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::network::AAssociateRJPDU::Read "std::istream& gdcm::network::AAssociateRJPDU::Read(std::istream &is)
";

%feature("docstring")  gdcm::network::AAssociateRJPDU::Size "size_t
gdcm::network::AAssociateRJPDU::Size() const ";

%feature("docstring")  gdcm::network::AAssociateRJPDU::Write "const
std::ostream& gdcm::network::AAssociateRJPDU::Write(std::ostream &os)
const ";


// File: classgdcm_1_1network_1_1AAssociateRQPDU.xml
%feature("docstring") gdcm::network::AAssociateRQPDU "

AAssociateRQPDU Table 9-11 ASSOCIATE-RQ PDU fields.

C++ includes: gdcmAAssociateRQPDU.h ";

%feature("docstring")  gdcm::network::AAssociateRQPDU::AAssociateRQPDU
"gdcm::network::AAssociateRQPDU::AAssociateRQPDU() ";

%feature("docstring")  gdcm::network::AAssociateRQPDU::AAssociateRQPDU
"gdcm::network::AAssociateRQPDU::AAssociateRQPDU(const
AAssociateRQPDU &pdu) ";

%feature("docstring")
gdcm::network::AAssociateRQPDU::AddPresentationContext "void
gdcm::network::AAssociateRQPDU::AddPresentationContext(PresentationContextRQ
const &pc) ";

%feature("docstring")
gdcm::network::AAssociateRQPDU::GetCalledAETitle "std::string
gdcm::network::AAssociateRQPDU::GetCalledAETitle() const ";

%feature("docstring")
gdcm::network::AAssociateRQPDU::GetCallingAETitle "std::string
gdcm::network::AAssociateRQPDU::GetCallingAETitle() const ";

%feature("docstring")
gdcm::network::AAssociateRQPDU::GetNumberOfPresentationContext "SizeType
gdcm::network::AAssociateRQPDU::GetNumberOfPresentationContext() const
";

%feature("docstring")
gdcm::network::AAssociateRQPDU::GetPresentationContext "PresentationContextRQ const&
gdcm::network::AAssociateRQPDU::GetPresentationContext(SizeType i)
const ";

%feature("docstring")
gdcm::network::AAssociateRQPDU::GetPresentationContextByAbstractSyntax
"const PresentationContextRQ*
gdcm::network::AAssociateRQPDU::GetPresentationContextByAbstractSyntax(AbstractSyntax
const &as) const ";

%feature("docstring")
gdcm::network::AAssociateRQPDU::GetPresentationContextByID "const
PresentationContextRQ*
gdcm::network::AAssociateRQPDU::GetPresentationContextByID(uint8_t i)
const ";

%feature("docstring")
gdcm::network::AAssociateRQPDU::GetPresentationContexts "PresentationContextArrayType const&
gdcm::network::AAssociateRQPDU::GetPresentationContexts() ";

%feature("docstring")
gdcm::network::AAssociateRQPDU::GetUserInformation "const
UserInformation& gdcm::network::AAssociateRQPDU::GetUserInformation()
const ";

%feature("docstring")  gdcm::network::AAssociateRQPDU::IsLastFragment
"bool gdcm::network::AAssociateRQPDU::IsLastFragment() const ";

%feature("docstring")  gdcm::network::AAssociateRQPDU::Print "void
gdcm::network::AAssociateRQPDU::Print(std::ostream &os) const

This function will initialize an AAssociateACPDU from the fields in
the AAssociateRQPDU structure ";

%feature("docstring")  gdcm::network::AAssociateRQPDU::Read "std::istream& gdcm::network::AAssociateRQPDU::Read(std::istream &is)
";

%feature("docstring")
gdcm::network::AAssociateRQPDU::SetCalledAETitle "void
gdcm::network::AAssociateRQPDU::SetCalledAETitle(const char
calledaetitle[16])

Set the Called AE Title. ";

%feature("docstring")
gdcm::network::AAssociateRQPDU::SetCallingAETitle "void
gdcm::network::AAssociateRQPDU::SetCallingAETitle(const char
callingaetitle[16])

Set the Calling AE Title. ";

%feature("docstring")
gdcm::network::AAssociateRQPDU::SetUserInformation "void
gdcm::network::AAssociateRQPDU::SetUserInformation(UserInformation
const &ui) ";

%feature("docstring")  gdcm::network::AAssociateRQPDU::Size "size_t
gdcm::network::AAssociateRQPDU::Size() const ";

%feature("docstring")  gdcm::network::AAssociateRQPDU::Write "const
std::ostream& gdcm::network::AAssociateRQPDU::Write(std::ostream &os)
const ";


// File: classgdcm_1_1AbortEvent.xml
%feature("docstring") gdcm::AbortEvent "C++ includes: gdcmEvent.h ";


// File: classgdcm_1_1network_1_1AbstractSyntax.xml
%feature("docstring") gdcm::network::AbstractSyntax "

AbstractSyntax Table 9-14 ABSTRACT SYNTAX SUB-ITEM FIELDS.

C++ includes: gdcmAbstractSyntax.h ";

%feature("docstring")  gdcm::network::AbstractSyntax::AbstractSyntax "gdcm::network::AbstractSyntax::AbstractSyntax() ";

%feature("docstring")  gdcm::network::AbstractSyntax::GetAsDataElement
"DataElement gdcm::network::AbstractSyntax::GetAsDataElement() const
";

%feature("docstring")  gdcm::network::AbstractSyntax::GetName "const
char* gdcm::network::AbstractSyntax::GetName() const ";

%feature("docstring")  gdcm::network::AbstractSyntax::Print "void
gdcm::network::AbstractSyntax::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::network::AbstractSyntax::Read "std::istream& gdcm::network::AbstractSyntax::Read(std::istream &is) ";

%feature("docstring")  gdcm::network::AbstractSyntax::SetName "void
gdcm::network::AbstractSyntax::SetName(const char *name) ";

%feature("docstring")  gdcm::network::AbstractSyntax::SetNameFromUID "void gdcm::network::AbstractSyntax::SetNameFromUID(UIDs::TSName
tsname) ";

%feature("docstring")  gdcm::network::AbstractSyntax::Size "size_t
gdcm::network::AbstractSyntax::Size() const ";

%feature("docstring")  gdcm::network::AbstractSyntax::Write "const
std::ostream& gdcm::network::AbstractSyntax::Write(std::ostream &os)
const ";


// File: classstd_1_1allocator.xml
%feature("docstring") std::allocator "

STL class. ";


// File: classgdcm_1_1AnonymizeEvent.xml
%feature("docstring") gdcm::AnonymizeEvent "

AnonymizeEvent Special type of event triggered during the
Anonymization process.

See:   Anonymizer

C++ includes: gdcmAnonymizeEvent.h ";

%feature("docstring")  gdcm::AnonymizeEvent::AnonymizeEvent "gdcm::AnonymizeEvent::AnonymizeEvent(Tag const &tag=0) ";

%feature("docstring")  gdcm::AnonymizeEvent::AnonymizeEvent "gdcm::AnonymizeEvent::AnonymizeEvent(const Self &s) ";

%feature("docstring")  gdcm::AnonymizeEvent::~AnonymizeEvent "virtual
gdcm::AnonymizeEvent::~AnonymizeEvent() ";

%feature("docstring")  gdcm::AnonymizeEvent::CheckEvent "virtual bool
gdcm::AnonymizeEvent::CheckEvent(const ::gdcm::Event *e) const ";

%feature("docstring")  gdcm::AnonymizeEvent::GetEventName "virtual
const char* gdcm::AnonymizeEvent::GetEventName() const

Return the StringName associated with the event. ";

%feature("docstring")  gdcm::AnonymizeEvent::GetTag "Tag const&
gdcm::AnonymizeEvent::GetTag() const ";

%feature("docstring")  gdcm::AnonymizeEvent::MakeObject "virtual
::gdcm::Event* gdcm::AnonymizeEvent::MakeObject() const

Create an Event of this type This method work as a Factory for
creating events of each particular type. ";

%feature("docstring")  gdcm::AnonymizeEvent::SetTag "void
gdcm::AnonymizeEvent::SetTag(const Tag &t) ";


// File: classgdcm_1_1Anonymizer.xml
%feature("docstring") gdcm::Anonymizer "

Anonymizer This class is a multi purpose anonymizer. It can work in 2
mode:

Full (irreversible) anonymizer (aka dumb mode)

reversible de-identifier/re-identifier (aka smart mode). This
implements the Basic Application Level Confidentiality Profile, DICOM
PS 3.15-2009

dumb mode This is a dumb anonymizer implementation. All it allows user
is simple operation such as:

Tag based functions: complete removal of DICOM attribute (Remove)

make a tag empty, ie make it's length 0 (Empty)

replace with another string-based value (Replace)

DataSet based functions: Remove all group length attribute from a
DICOM dataset (Group Length element are deprecated, DICOM 2008)

Remove all private attributes

Remove all retired attributes

All function calls actually execute the user specified request.
Previous implementation were calling a general Anonymize function but
traversing a std::set is O(n) operation, while a simple user specified
request is O(log(n)) operation. So 'm' user interaction is O(m*log(n))
which is < O(n) complexity.

smart mode this mode implements the Basic Application Level
Confidentiality Profile (DICOM PS 3.15-2008) In this case, it is
extremely important to use the same Anonymizer class when anonymizing
a FileSet. Once the Anonymizer is destroyed its memory of known
(already processed) UIDs will be lost. which will make the anonymizer
behaves incorrectly for attributes such as Series UID Study UID where
user want some consistency. When attribute is Type 1 / Type 1C, a
dummy generator will take in the existing value and produce a dummy
value (a sha1 representation). sha1 algorithm is considered to be
cryptographically strong (compared to md5sum) so that we meet the
following two conditions: Produce the same dummy value for the same
input value

do not provide an easy way to retrieve the original value from the
sha1 generated value

This class implement the Subject/Observer pattern trigger the
following event:  AnonymizeEvent

IterationEvent

StartEvent

EndEvent

See:   CryptographicMessageSyntax

C++ includes: gdcmAnonymizer.h ";

%feature("docstring")  gdcm::Anonymizer::Anonymizer "gdcm::Anonymizer::Anonymizer() ";

%feature("docstring")  gdcm::Anonymizer::~Anonymizer "gdcm::Anonymizer::~Anonymizer() ";

%feature("docstring")
gdcm::Anonymizer::BasicApplicationLevelConfidentialityProfile "bool
gdcm::Anonymizer::BasicApplicationLevelConfidentialityProfile(bool
deidentify=true)

PS 3.15 / E.1.1 De-Identifier An Application may claim conformance to
the Basic Application Level Confidentiality Profile as a deidentifier
if it protects all Attributes that might be used by unauthorized
entities to identify the patient. NOT THREAD SAFE ";

%feature("docstring")  gdcm::Anonymizer::Empty "bool
gdcm::Anonymizer::Empty(Tag const &t)

Make Tag t empty (if not found tag will be created) Warning: does not
handle SQ element ";

%feature("docstring")  gdcm::Anonymizer::GetCryptographicMessageSyntax
"const CryptographicMessageSyntax*
gdcm::Anonymizer::GetCryptographicMessageSyntax() const ";

%feature("docstring")  gdcm::Anonymizer::GetFile "File&
gdcm::Anonymizer::GetFile() ";

%feature("docstring")  gdcm::Anonymizer::Remove "bool
gdcm::Anonymizer::Remove(Tag const &t)

remove a tag (even a SQ can be removed) Return code is false when tag
t cannot be found ";

%feature("docstring")  gdcm::Anonymizer::RemoveGroupLength "bool
gdcm::Anonymizer::RemoveGroupLength()

Main function that loop over all elements and remove group length. ";

%feature("docstring")  gdcm::Anonymizer::RemovePrivateTags "bool
gdcm::Anonymizer::RemovePrivateTags()

Main function that loop over all elements and remove private tags. ";

%feature("docstring")  gdcm::Anonymizer::RemoveRetired "bool
gdcm::Anonymizer::RemoveRetired()

Main function that loop over all elements and remove retired element.
";

%feature("docstring")  gdcm::Anonymizer::Replace "bool
gdcm::Anonymizer::Replace(Tag const &t, const char *value)

Replace tag with another value, if tag is not found it will be
created: WARNING: this function can only execute if tag is a VRASCII
";

%feature("docstring")  gdcm::Anonymizer::Replace "bool
gdcm::Anonymizer::Replace(Tag const &t, const char *value, VL const
&vl)

when the value contains \\\\0, it is a good idea to specify the
length. This function is required when dealing with VRBINARY tag ";

%feature("docstring")  gdcm::Anonymizer::SetCryptographicMessageSyntax
"void
gdcm::Anonymizer::SetCryptographicMessageSyntax(CryptographicMessageSyntax
*cms)

Set/Get CMS key that will be used to encrypt the dataset within
BasicApplicationLevelConfidentialityProfile. ";

%feature("docstring")  gdcm::Anonymizer::SetFile "void
gdcm::Anonymizer::SetFile(const File &f)

Set/Get File. ";


// File: classgdcm_1_1AnyEvent.xml
%feature("docstring") gdcm::AnyEvent "C++ includes: gdcmEvent.h ";


// File: classgdcm_1_1network_1_1ApplicationContext.xml
%feature("docstring") gdcm::network::ApplicationContext "

ApplicationContext Table 9-12 APPLICATION CONTEXT ITEM FIELDS.

Todo Looks like Application Context can only be 64 bytes at max (see
Figure 9-1 / PS 3.8 - 2009 )

C++ includes: gdcmApplicationContext.h ";

%feature("docstring")
gdcm::network::ApplicationContext::ApplicationContext "gdcm::network::ApplicationContext::ApplicationContext() ";

%feature("docstring")  gdcm::network::ApplicationContext::GetName "const char* gdcm::network::ApplicationContext::GetName() const ";

%feature("docstring")  gdcm::network::ApplicationContext::Print "void
gdcm::network::ApplicationContext::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::network::ApplicationContext::Read "std::istream& gdcm::network::ApplicationContext::Read(std::istream
&is) ";

%feature("docstring")  gdcm::network::ApplicationContext::SetName "void gdcm::network::ApplicationContext::SetName(const char *name) ";

%feature("docstring")  gdcm::network::ApplicationContext::Size "size_t gdcm::network::ApplicationContext::Size() const ";

%feature("docstring")  gdcm::network::ApplicationContext::Write "const std::ostream&
gdcm::network::ApplicationContext::Write(std::ostream &os) const ";


// File: classgdcm_1_1ApplicationEntity.xml
%feature("docstring") gdcm::ApplicationEntity "

ApplicationEntity.

AE Application Entity

A string of characters that identifies an Application Entity with
leading and trailing spaces (20H) being non-significant. A value
consisting solely of spaces shall not be used.

Default Character Repertoire excluding character code 5CH (the
BACKSLASH \\\\ in ISO-IR 6), and control characters LF, FF, CR and
ESC.

16 bytes maximum

C++ includes: gdcmApplicationEntity.h ";

%feature("docstring")  gdcm::ApplicationEntity::IsValid "bool
gdcm::ApplicationEntity::IsValid() const ";

%feature("docstring")  gdcm::ApplicationEntity::Print "void
gdcm::ApplicationEntity::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::ApplicationEntity::SetBlob "void
gdcm::ApplicationEntity::SetBlob(const std::vector< char > &v) ";

%feature("docstring")  gdcm::ApplicationEntity::Squeeze "void
gdcm::ApplicationEntity::Squeeze() ";


// File: classgdcm_1_1network_1_1AReleaseRPPDU.xml
%feature("docstring") gdcm::network::AReleaseRPPDU "

AReleaseRPPDU Table 9-25 A-RELEASE-RP PDU fields.

C++ includes: gdcmAReleaseRPPDU.h ";

%feature("docstring")  gdcm::network::AReleaseRPPDU::AReleaseRPPDU "gdcm::network::AReleaseRPPDU::AReleaseRPPDU() ";

%feature("docstring")  gdcm::network::AReleaseRPPDU::IsLastFragment "bool gdcm::network::AReleaseRPPDU::IsLastFragment() const ";

%feature("docstring")  gdcm::network::AReleaseRPPDU::Print "void
gdcm::network::AReleaseRPPDU::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::network::AReleaseRPPDU::Read "std::istream& gdcm::network::AReleaseRPPDU::Read(std::istream &is) ";

%feature("docstring")  gdcm::network::AReleaseRPPDU::Size "size_t
gdcm::network::AReleaseRPPDU::Size() const ";

%feature("docstring")  gdcm::network::AReleaseRPPDU::Write "const
std::ostream& gdcm::network::AReleaseRPPDU::Write(std::ostream &os)
const ";


// File: classgdcm_1_1network_1_1AReleaseRQPDU.xml
%feature("docstring") gdcm::network::AReleaseRQPDU "

AReleaseRQPDU Table 9-24 A-RELEASE-RQ PDU FIELDS.

C++ includes: gdcmAReleaseRQPDU.h ";

%feature("docstring")  gdcm::network::AReleaseRQPDU::AReleaseRQPDU "gdcm::network::AReleaseRQPDU::AReleaseRQPDU() ";

%feature("docstring")  gdcm::network::AReleaseRQPDU::IsLastFragment "bool gdcm::network::AReleaseRQPDU::IsLastFragment() const ";

%feature("docstring")  gdcm::network::AReleaseRQPDU::Print "void
gdcm::network::AReleaseRQPDU::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::network::AReleaseRQPDU::Read "std::istream& gdcm::network::AReleaseRQPDU::Read(std::istream &is) ";

%feature("docstring")  gdcm::network::AReleaseRQPDU::Size "size_t
gdcm::network::AReleaseRQPDU::Size() const ";

%feature("docstring")  gdcm::network::AReleaseRQPDU::Write "const
std::ostream& gdcm::network::AReleaseRQPDU::Write(std::ostream &os)
const ";


// File: classstd_1_1array.xml
%feature("docstring") std::array "

STL class. ";


// File: classgdcm_1_1network_1_1ARTIMTimer.xml
%feature("docstring") gdcm::network::ARTIMTimer "

ARTIMTimer This file contains the code for the ARTIM timer.

Basically, the ARTIM timer will just get the wall time when it's
started, and then can be queried for the current time, and then can be
stopped (ie, the start time reset).

Because we're trying to do this without threading, we should be able
to 'start' the ARTIM timer by this mechanism, and then when waiting
for a particular response, tight loop that with sleep calls and
determinations of when the ARTIM timer has reached its peak. As such,
this isn't a strict 'timer' in the traditional sense of the word, but
more of a time keeper.

There can be only one ARTIM timer per connection.

C++ includes: gdcmARTIMTimer.h ";

%feature("docstring")  gdcm::network::ARTIMTimer::ARTIMTimer "gdcm::network::ARTIMTimer::ARTIMTimer() ";

%feature("docstring")  gdcm::network::ARTIMTimer::GetElapsedTime "double gdcm::network::ARTIMTimer::GetElapsedTime() const ";

%feature("docstring")  gdcm::network::ARTIMTimer::GetHasExpired "bool
gdcm::network::ARTIMTimer::GetHasExpired() const ";

%feature("docstring")  gdcm::network::ARTIMTimer::GetTimeout "double
gdcm::network::ARTIMTimer::GetTimeout() const ";

%feature("docstring")  gdcm::network::ARTIMTimer::SetTimeout "void
gdcm::network::ARTIMTimer::SetTimeout(double inTimeout) ";

%feature("docstring")  gdcm::network::ARTIMTimer::Start "void
gdcm::network::ARTIMTimer::Start() ";

%feature("docstring")  gdcm::network::ARTIMTimer::Stop "void
gdcm::network::ARTIMTimer::Stop() ";


// File: classgdcm_1_1ASN1.xml
%feature("docstring") gdcm::ASN1 "

Class for ASN1.

C++ includes: gdcmASN1.h ";

%feature("docstring")  gdcm::ASN1::ASN1 "gdcm::ASN1::ASN1() ";

%feature("docstring")  gdcm::ASN1::~ASN1 "gdcm::ASN1::~ASN1() ";


// File: classgdcm_1_1network_1_1AsynchronousOperationsWindowSub.xml
%feature("docstring") gdcm::network::AsynchronousOperationsWindowSub "

AsynchronousOperationsWindowSub PS 3.7 Table D.3-7 ASYNCHRONOUS
OPERATIONS WINDOW SUB-ITEM FIELDS (A-ASSOCIATE-RQ)

C++ includes: gdcmAsynchronousOperationsWindowSub.h ";

%feature("docstring")
gdcm::network::AsynchronousOperationsWindowSub::AsynchronousOperationsWindowSub
"gdcm::network::AsynchronousOperationsWindowSub::AsynchronousOperationsWindowSub()
";

%feature("docstring")
gdcm::network::AsynchronousOperationsWindowSub::Print "void
gdcm::network::AsynchronousOperationsWindowSub::Print(std::ostream
&os) const ";

%feature("docstring")
gdcm::network::AsynchronousOperationsWindowSub::Read "std::istream&
gdcm::network::AsynchronousOperationsWindowSub::Read(std::istream &is)
";

%feature("docstring")
gdcm::network::AsynchronousOperationsWindowSub::Size "size_t
gdcm::network::AsynchronousOperationsWindowSub::Size() const ";

%feature("docstring")
gdcm::network::AsynchronousOperationsWindowSub::Write "const
std::ostream&
gdcm::network::AsynchronousOperationsWindowSub::Write(std::ostream
&os) const ";


// File: classgdcm_1_1Attribute.xml
%feature("docstring") gdcm::Attribute "

Attribute class This class use template metaprograming tricks to let
the user know when the template instanciation does not match the
public dictionary.

Typical example that compile is: Attribute<0x0008,0x9007> a =
{\"ORIGINAL\",\"PRIMARY\",\"T1\",\"NONE\"};

Examples that will NOT compile are:

Attribute<0x0018,0x1182, VR::IS, VM::VM1> fd1 = {}; // not enough
parameters Attribute<0x0018,0x1182, VR::IS, VM::VM2> fd2 = {0,1,2}; //
too many initializers Attribute<0x0018,0x1182, VR::IS, VM::VM3> fd3 =
{0,1,2}; // VM3 is not valid Attribute<0x0018,0x1182, VR::UL, VM::VM2>
fd3 = {0,1}; // UL is not valid VR

C++ includes: gdcmAttribute.h ";

%feature("docstring")  gdcm::Attribute::GDCM_STATIC_ASSERT "gdcm::Attribute< Group, Element, TVR, TVM
>::GDCM_STATIC_ASSERT(((VR::VRType) TVR &(VR::VRType)(TagToType<
Group, Element >::VRType))) ";

%feature("docstring")  gdcm::Attribute::GDCM_STATIC_ASSERT "gdcm::Attribute< Group, Element, TVR, TVM
>::GDCM_STATIC_ASSERT(((VM::VMType) TVM &(VM::VMType)(TagToType<
Group, Element >::VMType))) ";

%feature("docstring")  gdcm::Attribute::GDCM_STATIC_ASSERT "gdcm::Attribute< Group, Element, TVR, TVM
>::GDCM_STATIC_ASSERT(((((VR::VRType) TVR &VR::VR_VM1)&&((VM::VMType)
TVM==VM::VM1))||!((VR::VRType) TVR &VR::VR_VM1))) ";

%feature("docstring")  gdcm::Attribute::GetAsDataElement "DataElement
gdcm::Attribute< Group, Element, TVR, TVM >::GetAsDataElement() const
";

%feature("docstring")  gdcm::Attribute::GetNumberOfValues "unsigned
int gdcm::Attribute< Group, Element, TVR, TVM >::GetNumberOfValues()
const ";

%feature("docstring")  gdcm::Attribute::GetValue "ArrayType&
gdcm::Attribute< Group, Element, TVR, TVM >::GetValue(unsigned int
idx=0) ";

%feature("docstring")  gdcm::Attribute::GetValue "ArrayType const&
gdcm::Attribute< Group, Element, TVR, TVM >::GetValue(unsigned int
idx=0) const ";

%feature("docstring")  gdcm::Attribute::GetValues "const ArrayType*
gdcm::Attribute< Group, Element, TVR, TVM >::GetValues() const ";

%feature("docstring")  gdcm::Attribute::Print "void gdcm::Attribute<
Group, Element, TVR, TVM >::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::Attribute::Set "void gdcm::Attribute<
Group, Element, TVR, TVM >::Set(DataSet const &ds) ";

%feature("docstring")  gdcm::Attribute::SetFromDataElement "void
gdcm::Attribute< Group, Element, TVR, TVM
>::SetFromDataElement(DataElement const &de) ";

%feature("docstring")  gdcm::Attribute::SetFromDataSet "void
gdcm::Attribute< Group, Element, TVR, TVM >::SetFromDataSet(DataSet
const &ds) ";

%feature("docstring")  gdcm::Attribute::SetValue "void
gdcm::Attribute< Group, Element, TVR, TVM >::SetValue(ArrayType v,
unsigned int idx=0) ";

%feature("docstring")  gdcm::Attribute::SetValues "void
gdcm::Attribute< Group, Element, TVR, TVM >::SetValues(const ArrayType
*array, unsigned int numel=VMType) ";


// File: classgdcm_1_1Attribute_3_01Group_00_01Element_00_01TVR_00_01VM_1_1VM1_01_4.xml
%feature("docstring") gdcm::Attribute< Group, Element, TVR, VM::VM1 >
" C++ includes: gdcmAttribute.h ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1
>::GDCM_STATIC_ASSERT " gdcm::Attribute< Group, Element, TVR, VM::VM1
>::GDCM_STATIC_ASSERT(VMToLength< VM::VM1 >::Length==1) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1
>::GDCM_STATIC_ASSERT " gdcm::Attribute< Group, Element, TVR, VM::VM1
>::GDCM_STATIC_ASSERT(((VR::VRType) TVR &(VR::VRType)(TagToType<
Group, Element >::VRType))) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1
>::GDCM_STATIC_ASSERT " gdcm::Attribute< Group, Element, TVR, VM::VM1
>::GDCM_STATIC_ASSERT(((VM::VMType) VM::VM1 &(VM::VMType)(TagToType<
Group, Element >::VMType))) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1
>::GDCM_STATIC_ASSERT " gdcm::Attribute< Group, Element, TVR, VM::VM1
>::GDCM_STATIC_ASSERT(((((VR::VRType) TVR &VR::VR_VM1)&&((VM::VMType)
VM::VM1==VM::VM1))||!((VR::VRType) TVR &VR::VR_VM1))) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1
>::GetAsDataElement " DataElement gdcm::Attribute< Group, Element,
TVR, VM::VM1 >::GetAsDataElement() const ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1
>::GetNumberOfValues " unsigned int gdcm::Attribute< Group, Element,
TVR, VM::VM1 >::GetNumberOfValues() const ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1
>::GetValue " ArrayType& gdcm::Attribute< Group, Element, TVR, VM::VM1
>::GetValue() ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1
>::GetValue " ArrayType const& gdcm::Attribute< Group, Element, TVR,
VM::VM1 >::GetValue() const ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1
>::GetValues " const ArrayType* gdcm::Attribute< Group, Element, TVR,
VM::VM1 >::GetValues() const ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1
>::Print " void gdcm::Attribute< Group, Element, TVR, VM::VM1
>::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1
>::Set " void gdcm::Attribute< Group, Element, TVR, VM::VM1
>::Set(DataSet const &ds) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1
>::SetFromDataElement " void gdcm::Attribute< Group, Element, TVR,
VM::VM1 >::SetFromDataElement(DataElement const &de) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1
>::SetFromDataSet " void gdcm::Attribute< Group, Element, TVR, VM::VM1
>::SetFromDataSet(DataSet const &ds) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1
>::SetValue " void gdcm::Attribute< Group, Element, TVR, VM::VM1
>::SetValue(ArrayType v) ";


// File: classgdcm_1_1Attribute_3_01Group_00_01Element_00_01TVR_00_01VM_1_1VM1__3_01_4.xml
%feature("docstring") gdcm::Attribute< Group, Element, TVR, VM::VM1_3
> " C++ includes: gdcmAttribute.h ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_3
>::GetVM " VM gdcm::Attribute< Group, Element, TVR, VM::VM1_3
>::GetVM() const ";


// File: classgdcm_1_1Attribute_3_01Group_00_01Element_00_01TVR_00_01VM_1_1VM1__8_01_4.xml
%feature("docstring") gdcm::Attribute< Group, Element, TVR, VM::VM1_8
> " C++ includes: gdcmAttribute.h ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_8
>::GetVM " VM gdcm::Attribute< Group, Element, TVR, VM::VM1_8
>::GetVM() const ";


// File: classgdcm_1_1Attribute_3_01Group_00_01Element_00_01TVR_00_01VM_1_1VM1__n_01_4.xml
%feature("docstring") gdcm::Attribute< Group, Element, TVR, VM::VM1_n
> " C++ includes: gdcmAttribute.h ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::Attribute " gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::Attribute() ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::~Attribute " gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::~Attribute() ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::GDCM_STATIC_ASSERT " gdcm::Attribute< Group, Element, TVR,
VM::VM1_n >::GDCM_STATIC_ASSERT(((VR::VRType) TVR
&(VR::VRType)(TagToType< Group, Element >::VRType))) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::GDCM_STATIC_ASSERT " gdcm::Attribute< Group, Element, TVR,
VM::VM1_n >::GDCM_STATIC_ASSERT((VM::VM1_n &(VM::VMType)(TagToType<
Group, Element >::VMType))) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::GDCM_STATIC_ASSERT " gdcm::Attribute< Group, Element, TVR,
VM::VM1_n >::GDCM_STATIC_ASSERT(((((VR::VRType) TVR
&VR::VR_VM1)&&((VM::VMType) TagToType< Group, Element
>::VMType==VM::VM1))||!((VR::VRType) TVR &VR::VR_VM1))) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::GetAsDataElement " DataElement gdcm::Attribute< Group, Element,
TVR, VM::VM1_n >::GetAsDataElement() const ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::GetNumberOfValues " unsigned int gdcm::Attribute< Group, Element,
TVR, VM::VM1_n >::GetNumberOfValues() const ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::GetValue " ArrayType& gdcm::Attribute< Group, Element, TVR,
VM::VM1_n >::GetValue(unsigned int idx=0) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::GetValue " ArrayType const& gdcm::Attribute< Group, Element, TVR,
VM::VM1_n >::GetValue(unsigned int idx=0) const ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::GetValues " const ArrayType* gdcm::Attribute< Group, Element, TVR,
VM::VM1_n >::GetValues() const ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::Print " void gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::Set " void gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::Set(DataSet const &ds) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::SetFromDataElement " void gdcm::Attribute< Group, Element, TVR,
VM::VM1_n >::SetFromDataElement(DataElement const &de) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::SetFromDataSet " void gdcm::Attribute< Group, Element, TVR,
VM::VM1_n >::SetFromDataSet(DataSet const &ds) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::SetNumberOfValues " void gdcm::Attribute< Group, Element, TVR,
VM::VM1_n >::SetNumberOfValues(unsigned int numel) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::SetValue " void gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::SetValue(unsigned int idx, ArrayType v) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::SetValue " void gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::SetValue(ArrayType v) ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::SetValues " void gdcm::Attribute< Group, Element, TVR, VM::VM1_n
>::SetValues(const ArrayType *array, unsigned int numel, bool
own=false) ";


// File: classgdcm_1_1Attribute_3_01Group_00_01Element_00_01TVR_00_01VM_1_1VM2__2n_01_4.xml
%feature("docstring") gdcm::Attribute< Group, Element, TVR, VM::VM2_2n
> " C++ includes: gdcmAttribute.h ";


// File: classgdcm_1_1Attribute_3_01Group_00_01Element_00_01TVR_00_01VM_1_1VM2__n_01_4.xml
%feature("docstring") gdcm::Attribute< Group, Element, TVR, VM::VM2_n
> " C++ includes: gdcmAttribute.h ";

%feature("docstring")  gdcm::Attribute< Group, Element, TVR, VM::VM2_n
>::GetVM " VM gdcm::Attribute< Group, Element, TVR, VM::VM2_n
>::GetVM() const ";


// File: classgdcm_1_1Attribute_3_01Group_00_01Element_00_01TVR_00_01VM_1_1VM3__3n_01_4.xml
%feature("docstring") gdcm::Attribute< Group, Element, TVR, VM::VM3_3n
> " C++ includes: gdcmAttribute.h ";


// File: classgdcm_1_1Attribute_3_01Group_00_01Element_00_01TVR_00_01VM_1_1VM3__n_01_4.xml
%feature("docstring") gdcm::Attribute< Group, Element, TVR, VM::VM3_n
> " C++ includes: gdcmAttribute.h ";


// File: classgdcm_1_1AudioCodec.xml
%feature("docstring") gdcm::AudioCodec "

AudioCodec.

C++ includes: gdcmAudioCodec.h ";

%feature("docstring")  gdcm::AudioCodec::AudioCodec "gdcm::AudioCodec::AudioCodec() ";

%feature("docstring")  gdcm::AudioCodec::~AudioCodec "gdcm::AudioCodec::~AudioCodec() ";

%feature("docstring")  gdcm::AudioCodec::CanCode "bool
gdcm::AudioCodec::CanCode(TransferSyntax const &) const

Return whether this coder support this transfer syntax (can code it)
";

%feature("docstring")  gdcm::AudioCodec::CanDecode "bool
gdcm::AudioCodec::CanDecode(TransferSyntax const &) const

Return whether this decoder support this transfer syntax (can decode
it) ";

%feature("docstring")  gdcm::AudioCodec::Decode "bool
gdcm::AudioCodec::Decode(DataElement const &is, DataElement &os)

Decode. ";


// File: classstd_1_1auto__ptr.xml
%feature("docstring") std::auto_ptr "

STL class. ";


// File: classstd_1_1bad__alloc.xml
%feature("docstring") std::bad_alloc "

STL class. ";


// File: classstd_1_1bad__cast.xml
%feature("docstring") std::bad_cast "

STL class. ";


// File: classstd_1_1bad__exception.xml
%feature("docstring") std::bad_exception "

STL class. ";


// File: classstd_1_1bad__typeid.xml
%feature("docstring") std::bad_typeid "

STL class. ";


// File: classgdcm_1_1Base64.xml
%feature("docstring") gdcm::Base64 "

Class for Base64.

C++ includes: gdcmBase64.h ";


// File: classgdcm_1_1network_1_1BaseCompositeMessage.xml
%feature("docstring") gdcm::network::BaseCompositeMessage "

BaseCompositeMessage The Composite events described in section
3.7-2009 of the DICOM standard all use their own messages. These
messages are constructed using Presentation Data Values, from section
3.8-2009 of the standard, and then fill in appropriate values in their
datasets.

So, for the five composites: C-ECHO

C-FIND

C-MOVE

C-GET

C-STORE there are a series of messages. However, all of these messages
are obtained as part of a PDataPDU, and all have to be placed there.
Therefore, since they all have shared functionality and construction
tropes, that will be put into a base class. Further, the base class
will be then returned by the factory class, gdcmCompositePDUFactory.
This is an abstract class. It cannot be instantiated on its own.

C++ includes: gdcmBaseCompositeMessage.h ";

%feature("docstring")
gdcm::network::BaseCompositeMessage::~BaseCompositeMessage "virtual
gdcm::network::BaseCompositeMessage::~BaseCompositeMessage() ";

%feature("docstring")
gdcm::network::BaseCompositeMessage::ConstructPDV "virtual
std::vector<PresentationDataValue>
gdcm::network::BaseCompositeMessage::ConstructPDV(const ULConnection
&inConnection, const BaseRootQuery *inRootQuery)=0 ";


// File: classgdcm_1_1network_1_1BasePDU.xml
%feature("docstring") gdcm::network::BasePDU "

BasePDU base class for PDUs.

all PDUs start with the first ten bytes as specified: 01 PDU type 02
reserved 3-6 PDU Length (unsigned) 7-10 variable

on some, 7-10 are split (7-8 as protocol version in Associate-RQ, for
instance, while associate-rj splits those four bytes differently).

Also common to all the PDUs is their ability to read and write to a
stream.

So, let's just get them all bunched together into one (abstract)
class, shall we?

Why? 1) so that the ULEvent can have the PDU stored in it, since the
event takes PDUs and not other class structures (other class
structures get converted into PDUs) 2) to make reading PDUs in the
event loop cleaner

C++ includes: gdcmBasePDU.h ";

%feature("docstring")  gdcm::network::BasePDU::~BasePDU "virtual
gdcm::network::BasePDU::~BasePDU() ";

%feature("docstring")  gdcm::network::BasePDU::IsLastFragment "virtual bool gdcm::network::BasePDU::IsLastFragment() const =0 ";

%feature("docstring")  gdcm::network::BasePDU::Print "virtual void
gdcm::network::BasePDU::Print(std::ostream &os) const =0 ";

%feature("docstring")  gdcm::network::BasePDU::Read "virtual
std::istream& gdcm::network::BasePDU::Read(std::istream &is)=0 ";

%feature("docstring")  gdcm::network::BasePDU::Size "virtual size_t
gdcm::network::BasePDU::Size() const =0 ";

%feature("docstring")  gdcm::network::BasePDU::Write "virtual const
std::ostream& gdcm::network::BasePDU::Write(std::ostream &os) const =0
";


// File: classgdcm_1_1BaseRootQuery.xml
%feature("docstring") gdcm::BaseRootQuery "

BaseRootQuery contains: a baseclass which will produce a dataset for
c-find and c-move with patient/study root.

This class contains the functionality used in patient c-find and
c-move queries. PatientRootQuery and StudyRootQuery derive from this
class.

Namely: 1) list all tags associated with a particular query type 2)
produce a query dataset via tag association

Eventually, it can be used to validate a particular dataset type.

The dataset held by this object (or, really, one of its derivates)
should be passed to a c-find or c-move query.

C++ includes: gdcmBaseRootQuery.h ";

%feature("docstring")  gdcm::BaseRootQuery::~BaseRootQuery "virtual
gdcm::BaseRootQuery::~BaseRootQuery() ";

%feature("docstring")  gdcm::BaseRootQuery::AddQueryDataSet "void
gdcm::BaseRootQuery::AddQueryDataSet(const DataSet &ds) ";

%feature("docstring")  gdcm::BaseRootQuery::GetAbstractSyntaxUID "virtual UIDs::TSName gdcm::BaseRootQuery::GetAbstractSyntaxUID() const
=0 ";

%feature("docstring")  gdcm::BaseRootQuery::GetQueryDataSet "DataSet
const& gdcm::BaseRootQuery::GetQueryDataSet() const

Set/Get the internal representation of the query as a DataSet. ";

%feature("docstring")  gdcm::BaseRootQuery::GetQueryDataSet "DataSet&
gdcm::BaseRootQuery::GetQueryDataSet() ";

%feature("docstring")  gdcm::BaseRootQuery::GetQueryLevelFromQueryRoot
"EQueryLevel
gdcm::BaseRootQuery::GetQueryLevelFromQueryRoot(ERootType roottype) ";

%feature("docstring")  gdcm::BaseRootQuery::GetTagListByLevel "virtual std::vector<Tag> gdcm::BaseRootQuery::GetTagListByLevel(const
EQueryLevel &inQueryLevel)=0

this function will return all tags at a given query level, so that
they maybe selected for searching. The boolean forFind is true if the
query is a find query, or false for a move query. ";

%feature("docstring")  gdcm::BaseRootQuery::InitializeDataSet "virtual void gdcm::BaseRootQuery::InitializeDataSet(const EQueryLevel
&inQueryLevel)=0

this function sets tag 8,52 to the appropriate value based on query
level also fills in the right unique tags, as per the standard's
requirements should allow for connection with dcmtk ";

%feature("docstring")  gdcm::BaseRootQuery::Print "void
gdcm::BaseRootQuery::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::BaseRootQuery::SetSearchParameter "void
gdcm::BaseRootQuery::SetSearchParameter(const Tag &inTag, const
std::string &inValue) ";

%feature("docstring")  gdcm::BaseRootQuery::SetSearchParameter "void
gdcm::BaseRootQuery::SetSearchParameter(const std::string &inKeyword,
const std::string &inValue) ";

%feature("docstring")  gdcm::BaseRootQuery::ValidateQuery "virtual
bool gdcm::BaseRootQuery::ValidateQuery(bool inStrict=true) const =0

have to be able to ensure that 0x8,0x52 is set (which will be true if
InitializeDataSet is called...) that the level is appropriate (ie, not
setting PATIENT for a study query that the tags in the query match the
right level (either required, unique, optional) by default, this
function checks to see if the query is for finding, which is more
permissive than for moving. For moving, only the unique tags are
allowed. 10 Jan 2011: adding in the 'strict' mode. according to the
standard (at least, how I've read it), only tags for a particular
level should be allowed in a particular query (ie, just series level
tags in a series level query). However, it seems that dcm4chee doesn't
share that interpretation. So, if 'inStrict' is false, then tags from
the current level and all higher levels are now considered valid. So,
if you're doing a non-strict series-level query, tags from the patient
and study level can be passed along as well. ";

%feature("docstring")  gdcm::BaseRootQuery::WriteHelpFile "const
std::ostream& gdcm::BaseRootQuery::WriteHelpFile(std::ostream &os) ";

%feature("docstring")  gdcm::BaseRootQuery::WriteQuery "bool
gdcm::BaseRootQuery::WriteQuery(const std::string &inFileName) ";


// File: classstd_1_1basic__fstream.xml
%feature("docstring") std::basic_fstream "

STL class. ";


// File: classstd_1_1basic__ifstream.xml
%feature("docstring") std::basic_ifstream "

STL class. ";


// File: classstd_1_1basic__ios.xml
%feature("docstring") std::basic_ios "

STL class. ";


// File: classstd_1_1basic__iostream.xml
%feature("docstring") std::basic_iostream "

STL class. ";


// File: classstd_1_1basic__istream.xml
%feature("docstring") std::basic_istream "

STL class. ";


// File: classstd_1_1basic__istringstream.xml
%feature("docstring") std::basic_istringstream "

STL class. ";


// File: classstd_1_1basic__ofstream.xml
%feature("docstring") std::basic_ofstream "

STL class. ";


// File: classstd_1_1basic__ostream.xml
%feature("docstring") std::basic_ostream "

STL class. ";


// File: classstd_1_1basic__ostringstream.xml
%feature("docstring") std::basic_ostringstream "

STL class. ";


// File: classstd_1_1basic__string.xml
%feature("docstring") std::basic_string "

STL class. ";


// File: classstd_1_1basic__stringstream.xml
%feature("docstring") std::basic_stringstream "

STL class. ";


// File: structgdcm_1_1SegmentHelper_1_1BasicCodedEntry.xml
%feature("docstring") gdcm::SegmentHelper::BasicCodedEntry "

This structure defines a basic coded entry with all of its attributes.

See:  PS 3.3 section 8.8.

C++ includes: gdcmSegmentHelper.h ";

%feature("docstring")
gdcm::SegmentHelper::BasicCodedEntry::BasicCodedEntry "gdcm::SegmentHelper::BasicCodedEntry::BasicCodedEntry()

Constructor. ";

%feature("docstring")
gdcm::SegmentHelper::BasicCodedEntry::BasicCodedEntry "gdcm::SegmentHelper::BasicCodedEntry::BasicCodedEntry(const char
*a_CV, const char *a_CSD, const char *a_CM)

constructor which defines type 1 attributes. ";

%feature("docstring")
gdcm::SegmentHelper::BasicCodedEntry::BasicCodedEntry "gdcm::SegmentHelper::BasicCodedEntry::BasicCodedEntry(const char
*a_CV, const char *a_CSD, const char *a_CSV, const char *a_CM)

constructor which defines attributes. ";

%feature("docstring")  gdcm::SegmentHelper::BasicCodedEntry::IsEmpty "bool gdcm::SegmentHelper::BasicCodedEntry::IsEmpty(const bool
checkOptionalAttributes=false) const

Check if each attibutes of the basic coded entry is defined.

Parameters:
-----------

checkOptionalAttributes:  Check also type 1C attributes. ";


// File: classgdcm_1_1BasicOffsetTable.xml
%feature("docstring") gdcm::BasicOffsetTable "

Class to represent a BasicOffsetTable.

C++ includes: gdcmBasicOffsetTable.h ";

%feature("docstring")  gdcm::BasicOffsetTable::BasicOffsetTable "gdcm::BasicOffsetTable::BasicOffsetTable() ";

%feature("docstring")  gdcm::BasicOffsetTable::Read "std::istream&
gdcm::BasicOffsetTable::Read(std::istream &is) ";


// File: classgdcm_1_1Bitmap.xml
%feature("docstring") gdcm::Bitmap "

Bitmap class A bitmap based image. Used as parent for both IconImage
and the main Pixel Data Image It does not contains any World Space
information (IPP, IOP)

C++ includes: gdcmBitmap.h ";

%feature("docstring")  gdcm::Bitmap::Bitmap "gdcm::Bitmap::Bitmap()
";

%feature("docstring")  gdcm::Bitmap::~Bitmap "gdcm::Bitmap::~Bitmap()
";

%feature("docstring")  gdcm::Bitmap::AreOverlaysInPixelData "virtual
bool gdcm::Bitmap::AreOverlaysInPixelData() const ";

%feature("docstring")  gdcm::Bitmap::Clear "void
gdcm::Bitmap::Clear() ";

%feature("docstring")  gdcm::Bitmap::GetBuffer "bool
gdcm::Bitmap::GetBuffer(char *buffer) const

Acces the raw data. ";

%feature("docstring")  gdcm::Bitmap::GetBufferLength "unsigned long
gdcm::Bitmap::GetBufferLength() const

Return the length of the image after decompression WARNING for palette
color: It will NOT take into account the Palette Color thus you need
to multiply this length by 3 if computing the size of equivalent RGB
image ";

%feature("docstring")  gdcm::Bitmap::GetColumns "unsigned int
gdcm::Bitmap::GetColumns() const ";

%feature("docstring")  gdcm::Bitmap::GetDataElement "const
DataElement& gdcm::Bitmap::GetDataElement() const ";

%feature("docstring")  gdcm::Bitmap::GetDataElement "DataElement&
gdcm::Bitmap::GetDataElement() ";

%feature("docstring")  gdcm::Bitmap::GetDimension "unsigned int
gdcm::Bitmap::GetDimension(unsigned int idx) const ";

%feature("docstring")  gdcm::Bitmap::GetDimensions "const unsigned
int* gdcm::Bitmap::GetDimensions() const

Return the dimension of the pixel data, first dimension (x), then 2nd
(y), then 3rd (z)... ";

%feature("docstring")  gdcm::Bitmap::GetLUT "const LookupTable&
gdcm::Bitmap::GetLUT() const ";

%feature("docstring")  gdcm::Bitmap::GetLUT "LookupTable&
gdcm::Bitmap::GetLUT() ";

%feature("docstring")  gdcm::Bitmap::GetNeedByteSwap "bool
gdcm::Bitmap::GetNeedByteSwap() const ";

%feature("docstring")  gdcm::Bitmap::GetNumberOfDimensions "unsigned
int gdcm::Bitmap::GetNumberOfDimensions() const

Return the number of dimension of the pixel data bytes; for example 2
for a 2D matrices of values. ";

%feature("docstring")  gdcm::Bitmap::GetPhotometricInterpretation "const PhotometricInterpretation&
gdcm::Bitmap::GetPhotometricInterpretation() const

return the photometric interpretation ";

%feature("docstring")  gdcm::Bitmap::GetPixelFormat "const
PixelFormat& gdcm::Bitmap::GetPixelFormat() const

Get/Set PixelFormat. ";

%feature("docstring")  gdcm::Bitmap::GetPixelFormat "PixelFormat&
gdcm::Bitmap::GetPixelFormat() ";

%feature("docstring")  gdcm::Bitmap::GetPlanarConfiguration "unsigned
int gdcm::Bitmap::GetPlanarConfiguration() const

return the planar configuration ";

%feature("docstring")  gdcm::Bitmap::GetRows "unsigned int
gdcm::Bitmap::GetRows() const ";

%feature("docstring")  gdcm::Bitmap::GetTransferSyntax "const
TransferSyntax& gdcm::Bitmap::GetTransferSyntax() const ";

%feature("docstring")  gdcm::Bitmap::IsEmpty "bool
gdcm::Bitmap::IsEmpty() const ";

%feature("docstring")  gdcm::Bitmap::IsLossy "bool
gdcm::Bitmap::IsLossy() const

Return whether or not the image was compressed using a lossy
compressor or not. ";

%feature("docstring")  gdcm::Bitmap::IsTransferSyntaxCompatible "bool
gdcm::Bitmap::IsTransferSyntaxCompatible(TransferSyntax const &ts)
const ";

%feature("docstring")  gdcm::Bitmap::Print "void
gdcm::Bitmap::Print(std::ostream &) const ";

%feature("docstring")  gdcm::Bitmap::SetColumns "void
gdcm::Bitmap::SetColumns(unsigned int col) ";

%feature("docstring")  gdcm::Bitmap::SetDataElement "void
gdcm::Bitmap::SetDataElement(DataElement const &de) ";

%feature("docstring")  gdcm::Bitmap::SetDimension "void
gdcm::Bitmap::SetDimension(unsigned int idx, unsigned int dim) ";

%feature("docstring")  gdcm::Bitmap::SetDimensions "void
gdcm::Bitmap::SetDimensions(const unsigned int dims[3]) ";

%feature("docstring")  gdcm::Bitmap::SetLossyFlag "void
gdcm::Bitmap::SetLossyFlag(bool f)

Specifically set that the image was compressed using a lossy
compression mechanism. ";

%feature("docstring")  gdcm::Bitmap::SetLUT "void
gdcm::Bitmap::SetLUT(LookupTable const &lut)

Set/Get LUT. ";

%feature("docstring")  gdcm::Bitmap::SetNeedByteSwap "void
gdcm::Bitmap::SetNeedByteSwap(bool b) ";

%feature("docstring")  gdcm::Bitmap::SetNumberOfDimensions "void
gdcm::Bitmap::SetNumberOfDimensions(unsigned int dim) ";

%feature("docstring")  gdcm::Bitmap::SetPhotometricInterpretation "void
gdcm::Bitmap::SetPhotometricInterpretation(PhotometricInterpretation
const &pi) ";

%feature("docstring")  gdcm::Bitmap::SetPixelFormat "void
gdcm::Bitmap::SetPixelFormat(PixelFormat const &pf) ";

%feature("docstring")  gdcm::Bitmap::SetPlanarConfiguration "void
gdcm::Bitmap::SetPlanarConfiguration(unsigned int pc)

WARNING:  you need to call SetPixelFormat first (before
SetPlanarConfiguration) for consistency checking ";

%feature("docstring")  gdcm::Bitmap::SetRows "void
gdcm::Bitmap::SetRows(unsigned int rows) ";

%feature("docstring")  gdcm::Bitmap::SetTransferSyntax "void
gdcm::Bitmap::SetTransferSyntax(TransferSyntax const &ts)

Transfer syntax. ";


// File: classgdcm_1_1BitmapToBitmapFilter.xml
%feature("docstring") gdcm::BitmapToBitmapFilter "

BitmapToBitmapFilter class Super class for all filter taking an image
and producing an output image.

C++ includes: gdcmBitmapToBitmapFilter.h ";

%feature("docstring")
gdcm::BitmapToBitmapFilter::BitmapToBitmapFilter "gdcm::BitmapToBitmapFilter::BitmapToBitmapFilter() ";

%feature("docstring")
gdcm::BitmapToBitmapFilter::~BitmapToBitmapFilter "gdcm::BitmapToBitmapFilter::~BitmapToBitmapFilter() ";

%feature("docstring")  gdcm::BitmapToBitmapFilter::GetOutput "const
Bitmap& gdcm::BitmapToBitmapFilter::GetOutput() const

Get Output image. ";

%feature("docstring")  gdcm::BitmapToBitmapFilter::GetOutputAsBitmap "const Bitmap& gdcm::BitmapToBitmapFilter::GetOutputAsBitmap() const ";

%feature("docstring")  gdcm::BitmapToBitmapFilter::SetInput "void
gdcm::BitmapToBitmapFilter::SetInput(const Bitmap &image)

Set input image. ";


// File: classstd_1_1bitset.xml
%feature("docstring") std::bitset "

STL class. ";


// File: classgdcm_1_1BoxRegion.xml
%feature("docstring") gdcm::BoxRegion "

Class for manipulation box region This is a very simple implementation
of the Region class. It only support 3D box type region. It assumes
the 3D Box does not have a tilt Origin is as (0,0,0)

C++ includes: gdcmBoxRegion.h ";

%feature("docstring")  gdcm::BoxRegion::BoxRegion "gdcm::BoxRegion::BoxRegion() ";

%feature("docstring")  gdcm::BoxRegion::BoxRegion "gdcm::BoxRegion::BoxRegion(const BoxRegion &)

copy/cstor and al. ";

%feature("docstring")  gdcm::BoxRegion::~BoxRegion "gdcm::BoxRegion::~BoxRegion() ";

%feature("docstring")  gdcm::BoxRegion::Area "size_t
gdcm::BoxRegion::Area() const

compute the area ";

%feature("docstring")  gdcm::BoxRegion::Clone "Region*
gdcm::BoxRegion::Clone() const ";

%feature("docstring")  gdcm::BoxRegion::ComputeBoundingBox "BoxRegion
gdcm::BoxRegion::ComputeBoundingBox()

Return the Axis-Aligned minimum bounding box for all regions. ";

%feature("docstring")  gdcm::BoxRegion::Empty "bool
gdcm::BoxRegion::Empty() const

return whether this domain is empty: ";

%feature("docstring")  gdcm::BoxRegion::GetXMax "unsigned int
gdcm::BoxRegion::GetXMax() const ";

%feature("docstring")  gdcm::BoxRegion::GetXMin "unsigned int
gdcm::BoxRegion::GetXMin() const

Get domain. ";

%feature("docstring")  gdcm::BoxRegion::GetYMax "unsigned int
gdcm::BoxRegion::GetYMax() const ";

%feature("docstring")  gdcm::BoxRegion::GetYMin "unsigned int
gdcm::BoxRegion::GetYMin() const ";

%feature("docstring")  gdcm::BoxRegion::GetZMax "unsigned int
gdcm::BoxRegion::GetZMax() const ";

%feature("docstring")  gdcm::BoxRegion::GetZMin "unsigned int
gdcm::BoxRegion::GetZMin() const ";

%feature("docstring")  gdcm::BoxRegion::IsValid "bool
gdcm::BoxRegion::IsValid() const

return whether this is valid domain ";

%feature("docstring")  gdcm::BoxRegion::Print "void
gdcm::BoxRegion::Print(std::ostream &os=std::cout) const

Print. ";

%feature("docstring")  gdcm::BoxRegion::SetDomain "void
gdcm::BoxRegion::SetDomain(unsigned int xmin, unsigned int xmax,
unsigned int ymin, unsigned int ymax, unsigned int zmin, unsigned int
zmax)

Set domain. ";


// File: classgdcm_1_1ByteBuffer.xml
%feature("docstring") gdcm::ByteBuffer "

ByteBuffer.

Detailled description here looks like a std::streambuf or std::filebuf
class with the get and peek pointer

C++ includes: gdcmByteBuffer.h ";

%feature("docstring")  gdcm::ByteBuffer::ByteBuffer "gdcm::ByteBuffer::ByteBuffer() ";

%feature("docstring")  gdcm::ByteBuffer::Get "char*
gdcm::ByteBuffer::Get(int len) ";

%feature("docstring")  gdcm::ByteBuffer::GetStart "const char*
gdcm::ByteBuffer::GetStart() const ";

%feature("docstring")  gdcm::ByteBuffer::ShiftEnd "void
gdcm::ByteBuffer::ShiftEnd(int len) ";

%feature("docstring")  gdcm::ByteBuffer::UpdatePosition "void
gdcm::ByteBuffer::UpdatePosition() ";


// File: classgdcm_1_1ByteSwap.xml
%feature("docstring") gdcm::ByteSwap "

ByteSwap.

Perform machine dependent byte swaping (Little Endian, Big Endian, Bad
Little Endian, Bad Big Endian). TODO: bswap_32 / bswap_64 ...

C++ includes: gdcmByteSwap.h ";


// File: classgdcm_1_1ByteSwapFilter.xml
%feature("docstring") gdcm::ByteSwapFilter "

ByteSwapFilter In place byte-swapping of a dataset FIXME: FL status ??

C++ includes: gdcmByteSwapFilter.h ";

%feature("docstring")  gdcm::ByteSwapFilter::ByteSwapFilter "gdcm::ByteSwapFilter::ByteSwapFilter(DataSet &ds) ";

%feature("docstring")  gdcm::ByteSwapFilter::~ByteSwapFilter "gdcm::ByteSwapFilter::~ByteSwapFilter() ";

%feature("docstring")  gdcm::ByteSwapFilter::ByteSwap "bool
gdcm::ByteSwapFilter::ByteSwap() ";

%feature("docstring")  gdcm::ByteSwapFilter::SetByteSwapTag "void
gdcm::ByteSwapFilter::SetByteSwapTag(bool b) ";


// File: classgdcm_1_1ByteValue.xml
%feature("docstring") gdcm::ByteValue "

Class to represent binary value (array of bytes)

C++ includes: gdcmByteValue.h ";

%feature("docstring")  gdcm::ByteValue::ByteValue "gdcm::ByteValue::ByteValue(const char *array=0, VL const &vl=0) ";

%feature("docstring")  gdcm::ByteValue::ByteValue "gdcm::ByteValue::ByteValue(std::vector< char > &v)

WARNING:  casting to uint32_t ";

%feature("docstring")  gdcm::ByteValue::~ByteValue "gdcm::ByteValue::~ByteValue() ";

%feature("docstring")  gdcm::ByteValue::Append "void
gdcm::ByteValue::Append(ByteValue const &bv) ";

%feature("docstring")  gdcm::ByteValue::Clear "void
gdcm::ByteValue::Clear() ";

%feature("docstring")  gdcm::ByteValue::ComputeLength "VL
gdcm::ByteValue::ComputeLength() const ";

%feature("docstring")  gdcm::ByteValue::Fill "void
gdcm::ByteValue::Fill(char c) ";

%feature("docstring")  gdcm::ByteValue::GetBuffer "bool
gdcm::ByteValue::GetBuffer(char *buffer, unsigned long length) const
";

%feature("docstring")  gdcm::ByteValue::GetLength "VL
gdcm::ByteValue::GetLength() const ";

%feature("docstring")  gdcm::ByteValue::GetPointer "const char*
gdcm::ByteValue::GetPointer() const ";

%feature("docstring")  gdcm::ByteValue::IsEmpty "bool
gdcm::ByteValue::IsEmpty() const ";

%feature("docstring")  gdcm::ByteValue::IsPrintable "bool
gdcm::ByteValue::IsPrintable(VL length) const

Checks whether a ' ByteValue' is printable or not (in order to avoid
corrupting the terminal of invocation when printing) I don't think
this function is working since it does not handle UNICODE or character
set... ";

%feature("docstring")  gdcm::ByteValue::PrintASCII "void
gdcm::ByteValue::PrintASCII(std::ostream &os, VL maxlength) const ";

%feature("docstring")  gdcm::ByteValue::PrintASCIIXML "void
gdcm::ByteValue::PrintASCIIXML(std::ostream &os) const ";

%feature("docstring")  gdcm::ByteValue::PrintGroupLength "void
gdcm::ByteValue::PrintGroupLength(std::ostream &os) ";

%feature("docstring")  gdcm::ByteValue::PrintHex "void
gdcm::ByteValue::PrintHex(std::ostream &os, VL maxlength) const ";

%feature("docstring")  gdcm::ByteValue::PrintHexXML "void
gdcm::ByteValue::PrintHexXML(std::ostream &os) const ";

%feature("docstring")  gdcm::ByteValue::PrintPNXML "void
gdcm::ByteValue::PrintPNXML(std::ostream &os) const

To Print Values in Native DICOM format ";

%feature("docstring")  gdcm::ByteValue::Read "std::istream&
gdcm::ByteValue::Read(std::istream &is, bool readvalues=true) ";

%feature("docstring")  gdcm::ByteValue::Read "std::istream&
gdcm::ByteValue::Read(std::istream &is) ";

%feature("docstring")  gdcm::ByteValue::SetLength "void
gdcm::ByteValue::SetLength(VL vl) ";

%feature("docstring")  gdcm::ByteValue::Write "std::ostream const&
gdcm::ByteValue::Write(std::ostream &os) const ";

%feature("docstring")  gdcm::ByteValue::Write "std::ostream const&
gdcm::ByteValue::Write(std::ostream &os) const ";

%feature("docstring")  gdcm::ByteValue::WriteBuffer "bool
gdcm::ByteValue::WriteBuffer(std::ostream &os) const ";


// File: classgdcm_1_1CAPICryptoFactory.xml
%feature("docstring") gdcm::CAPICryptoFactory "C++ includes:
gdcmCAPICryptoFactory.h ";

%feature("docstring")  gdcm::CAPICryptoFactory::CAPICryptoFactory "gdcm::CAPICryptoFactory::CAPICryptoFactory(CryptoLib id) ";

%feature("docstring")  gdcm::CAPICryptoFactory::CreateCMSProvider "CryptographicMessageSyntax*
gdcm::CAPICryptoFactory::CreateCMSProvider() ";


// File: classgdcm_1_1CAPICryptographicMessageSyntax.xml
%feature("docstring") gdcm::CAPICryptographicMessageSyntax "C++
includes: gdcmCAPICryptographicMessageSyntax.h ";

%feature("docstring")
gdcm::CAPICryptographicMessageSyntax::CAPICryptographicMessageSyntax "gdcm::CAPICryptographicMessageSyntax::CAPICryptographicMessageSyntax()
";

%feature("docstring")
gdcm::CAPICryptographicMessageSyntax::~CAPICryptographicMessageSyntax
"gdcm::CAPICryptographicMessageSyntax::~CAPICryptographicMessageSyntax()
";

%feature("docstring")  gdcm::CAPICryptographicMessageSyntax::Decrypt "bool gdcm::CAPICryptographicMessageSyntax::Decrypt(char *output,
size_t &outlen, const char *array, size_t len) const

decrypt content from a CMS envelopedData structure ";

%feature("docstring")  gdcm::CAPICryptographicMessageSyntax::Encrypt "bool gdcm::CAPICryptographicMessageSyntax::Encrypt(char *output,
size_t &outlen, const char *array, size_t len) const

create a CMS envelopedData structure ";

%feature("docstring")
gdcm::CAPICryptographicMessageSyntax::GetCipherType "CipherTypes
gdcm::CAPICryptographicMessageSyntax::GetCipherType() const ";

%feature("docstring")
gdcm::CAPICryptographicMessageSyntax::GetInitialized "bool
gdcm::CAPICryptographicMessageSyntax::GetInitialized() const ";

%feature("docstring")
gdcm::CAPICryptographicMessageSyntax::ParseCertificateFile "bool
gdcm::CAPICryptographicMessageSyntax::ParseCertificateFile(const char
*filename) ";

%feature("docstring")
gdcm::CAPICryptographicMessageSyntax::ParseKeyFile "bool
gdcm::CAPICryptographicMessageSyntax::ParseKeyFile(const char
*filename) ";

%feature("docstring")
gdcm::CAPICryptographicMessageSyntax::SetCipherType "void
gdcm::CAPICryptographicMessageSyntax::SetCipherType(CipherTypes type)
";

%feature("docstring")
gdcm::CAPICryptographicMessageSyntax::SetPassword "bool
gdcm::CAPICryptographicMessageSyntax::SetPassword(const char *pass,
size_t passLen) ";


// File: classgdcm_1_1network_1_1CEchoRQ.xml
%feature("docstring") gdcm::network::CEchoRQ "

CEchoRQ this file defines the messages for the cecho action.

C++ includes: gdcmCEchoMessages.h ";

%feature("docstring")  gdcm::network::CEchoRQ::ConstructPDV "std::vector<PresentationDataValue>
gdcm::network::CEchoRQ::ConstructPDV(const ULConnection &inConnection,
const BaseRootQuery *inRootQuery) ";


// File: classgdcm_1_1network_1_1CEchoRSP.xml
%feature("docstring") gdcm::network::CEchoRSP "

CEchoRSP this file defines the messages for the cecho action.

C++ includes: gdcmCEchoMessages.h ";

%feature("docstring")  gdcm::network::CEchoRSP::ConstructPDVByDataSet
"std::vector<PresentationDataValue>
gdcm::network::CEchoRSP::ConstructPDVByDataSet(const DataSet
*inDataSet) ";


// File: classgdcm_1_1network_1_1CFind.xml
%feature("docstring") gdcm::network::CFind "

PS 3.4 - 2009 Table B.2-1 C-STORE STATUS

C++ includes: gdcmDIMSE.h ";


// File: classgdcm_1_1network_1_1CFindCancelRQ.xml
%feature("docstring") gdcm::network::CFindCancelRQ "

CFindCancelRQ this file defines the messages for the cfind action.

C++ includes: gdcmCFindMessages.h ";

%feature("docstring")
gdcm::network::CFindCancelRQ::ConstructPDVByDataSet "std::vector<PresentationDataValue>
gdcm::network::CFindCancelRQ::ConstructPDVByDataSet(const DataSet
*inDataSet) ";


// File: classgdcm_1_1network_1_1CFindRQ.xml
%feature("docstring") gdcm::network::CFindRQ "

CFindRQ this file defines the messages for the cfind action.

C++ includes: gdcmCFindMessages.h ";

%feature("docstring")  gdcm::network::CFindRQ::ConstructPDV "std::vector<PresentationDataValue>
gdcm::network::CFindRQ::ConstructPDV(const ULConnection &inConnection,
const BaseRootQuery *inRootQuery) ";


// File: classgdcm_1_1network_1_1CFindRSP.xml
%feature("docstring") gdcm::network::CFindRSP "

CFindRSP this file defines the messages for the cfind action.

C++ includes: gdcmCFindMessages.h ";

%feature("docstring")  gdcm::network::CFindRSP::ConstructPDVByDataSet
"std::vector<PresentationDataValue>
gdcm::network::CFindRSP::ConstructPDVByDataSet(const DataSet
*inDataSet) ";


// File: classgdcm_1_1network_1_1CMoveCancelRq.xml
%feature("docstring") gdcm::network::CMoveCancelRq "C++ includes:
gdcmCMoveMessages.h ";

%feature("docstring")
gdcm::network::CMoveCancelRq::ConstructPDVByDataSet "std::vector<PresentationDataValue>
gdcm::network::CMoveCancelRq::ConstructPDVByDataSet(const DataSet
*inDataSet) ";


// File: classgdcm_1_1network_1_1CMoveRQ.xml
%feature("docstring") gdcm::network::CMoveRQ "

CMoveRQ this file defines the messages for the cmove action.

C++ includes: gdcmCMoveMessages.h ";

%feature("docstring")  gdcm::network::CMoveRQ::ConstructPDV "std::vector<PresentationDataValue>
gdcm::network::CMoveRQ::ConstructPDV(const ULConnection &inConnection,
const BaseRootQuery *inRootQuery) ";


// File: classgdcm_1_1network_1_1CMoveRSP.xml
%feature("docstring") gdcm::network::CMoveRSP "

CMoveRSP this file defines the messages for the cmove action.

C++ includes: gdcmCMoveMessages.h ";

%feature("docstring")  gdcm::network::CMoveRSP::ConstructPDVByDataSet
"std::vector<PresentationDataValue>
gdcm::network::CMoveRSP::ConstructPDVByDataSet(const DataSet
*inDataSet) ";


// File: classgdcm_1_1Codec.xml
%feature("docstring") gdcm::Codec "

Codec class.

C++ includes: gdcmCodec.h ";


// File: classgdcm_1_1Coder.xml
%feature("docstring") gdcm::Coder "

Coder.

C++ includes: gdcmCoder.h ";

%feature("docstring")  gdcm::Coder::~Coder "virtual
gdcm::Coder::~Coder() ";

%feature("docstring")  gdcm::Coder::CanCode "virtual bool
gdcm::Coder::CanCode(TransferSyntax const &) const =0

Return whether this coder support this transfer syntax (can code it)
";

%feature("docstring")  gdcm::Coder::Code "virtual bool
gdcm::Coder::Code(DataElement const &in_, DataElement &out_)

Code. ";


// File: classgdcm_1_1CodeString.xml
%feature("docstring") gdcm::CodeString "

CodeString This is an implementation of DICOM VR: CS The cstor will
properly Trim so that operator== is correct.

the cstor of CodeString will Trim the string on the fly so as to
remove the extra leading and ending spaces. However it will not
perform validation on the fly ( CodeString obj can contains invalid
char such as lower cases). This design was chosen to be a little
tolerant to broken DICOM implementation, and thus allow user to
compare lower case CS from there input file without the need to first
rewrite them to get rid of invalid character (validation is a
different operation from searching, querying).

WARNING:  when writing out DICOM file it is highly recommended to
perform the IsValid() call, at least to check that the length of the
string match the definition in the standard.

C++ includes: gdcmCodeString.h ";

%feature("docstring")  gdcm::CodeString::CodeString "gdcm::CodeString::CodeString()

CodeString constructors. ";

%feature("docstring")  gdcm::CodeString::CodeString "gdcm::CodeString::CodeString(const value_type *s) ";

%feature("docstring")  gdcm::CodeString::CodeString "gdcm::CodeString::CodeString(const value_type *s, size_type n) ";

%feature("docstring")  gdcm::CodeString::CodeString "gdcm::CodeString::CodeString(const InternalClass &s, size_type pos=0,
size_type n=InternalClass::npos) ";

%feature("docstring")  gdcm::CodeString::GetAsString "std::string
gdcm::CodeString::GetAsString() const

Return the full code string as std::string. ";

%feature("docstring")  gdcm::CodeString::IsValid "bool
gdcm::CodeString::IsValid() const

Check if CodeString obj is correct.. ";

%feature("docstring")  gdcm::CodeString::Size "size_type
gdcm::CodeString::Size() const

Return the size of the string. ";


// File: classgdcm_1_1Command.xml
%feature("docstring") gdcm::Command "

Command superclass for callback/observer methods.

See:   Subject

C++ includes: gdcmCommand.h ";

%feature("docstring")  gdcm::Command::Execute "virtual void
gdcm::Command::Execute(Subject *caller, const Event &event)=0

Abstract method that defines the action to be taken by the command. ";

%feature("docstring")  gdcm::Command::Execute "virtual void
gdcm::Command::Execute(const Subject *caller, const Event &event)=0

Abstract method that defines the action to be taken by the command.
This variant is expected to be used when requests comes from a const
Object ";


// File: classgdcm_1_1CommandDataSet.xml
%feature("docstring") gdcm::CommandDataSet "

Class to represent a Command DataSet.

See:   DataSet

C++ includes: gdcmCommandDataSet.h ";

%feature("docstring")  gdcm::CommandDataSet::CommandDataSet "gdcm::CommandDataSet::CommandDataSet() ";

%feature("docstring")  gdcm::CommandDataSet::~CommandDataSet "gdcm::CommandDataSet::~CommandDataSet() ";

%feature("docstring")  gdcm::CommandDataSet::Insert "void
gdcm::CommandDataSet::Insert(const DataElement &de)

Insert a DataElement in the DataSet. WARNING:  : Tag need to be >= 0x8
to be considered valid data element ";

%feature("docstring")  gdcm::CommandDataSet::Read "std::istream&
gdcm::CommandDataSet::Read(std::istream &is)

Read. ";

%feature("docstring")  gdcm::CommandDataSet::Replace "void
gdcm::CommandDataSet::Replace(const DataElement &de)

Replace a dataelement with another one. ";

%feature("docstring")  gdcm::CommandDataSet::Write "std::ostream&
gdcm::CommandDataSet::Write(std::ostream &os) const

Write. ";


// File: classstd_1_1complex.xml
%feature("docstring") std::complex "

STL class. ";


// File: classgdcm_1_1network_1_1CompositeMessageFactory.xml
%feature("docstring") gdcm::network::CompositeMessageFactory "

CompositeMessageFactory This class constructs PDataPDUs, but that have
been specifically constructed for the composite DICOM services
(C-Echo, C-Find, C-Get, C-Move, and C-Store). It will also handle
parsing the incoming data to determine which of the CompositePDUs the
incoming data is, and so therefore allowing the scu to determine what
to do with incoming data (if acting as a storescp server, for
instance).

C++ includes: gdcmCompositeMessageFactory.h ";


// File: classgdcm_1_1CompositeNetworkFunctions.xml
%feature("docstring") gdcm::CompositeNetworkFunctions "

Composite Network Functions These functions provide a generic API to
the DICOM functions implemented in GDCM. Advanced users can use this
code as a template for building their own versions of these functions
(for instance, to provide progress bars or some other way of handling
returned query information), but for most users, these functions
should be sufficient to interface with a PACS to a local machine. Note
that these functions are not contained within a static class or some
other class-style interface, because multiple connections can be
instantiated in the same program. The DICOM standard is much more
function oriented rather than class oriented in this instance, so the
design of this API reflects that functional approach. These functions
implements the following SCU operations:

C-ECHO SCU

C-FIND SCU

C-STORE SCU

C-MOVE SCU (+internal C-STORE SCP)

C++ includes: gdcmCompositeNetworkFunctions.h ";


// File: classstd_1_1list_1_1const__iterator.xml
%feature("docstring") std::list::const_iterator "

STL iterator class. ";


// File: classstd_1_1forward__list_1_1const__iterator.xml
%feature("docstring") std::forward_list::const_iterator "

STL iterator class. ";


// File: classstd_1_1map_1_1const__iterator.xml
%feature("docstring") std::map::const_iterator "

STL iterator class. ";


// File: classstd_1_1unordered__map_1_1const__iterator.xml
%feature("docstring") std::unordered_map::const_iterator "

STL iterator class. ";


// File: classstd_1_1basic__string_1_1const__iterator.xml
%feature("docstring") std::basic_string::const_iterator "

STL iterator class. ";


// File: classstd_1_1multimap_1_1const__iterator.xml
%feature("docstring") std::multimap::const_iterator "

STL iterator class. ";


// File: classstd_1_1unordered__multimap_1_1const__iterator.xml
%feature("docstring") std::unordered_multimap::const_iterator "

STL iterator class. ";


// File: classstd_1_1set_1_1const__iterator.xml
%feature("docstring") std::set::const_iterator "

STL iterator class. ";


// File: classstd_1_1string_1_1const__iterator.xml
%feature("docstring") std::string::const_iterator "

STL iterator class. ";


// File: classstd_1_1unordered__set_1_1const__iterator.xml
%feature("docstring") std::unordered_set::const_iterator "

STL iterator class. ";


// File: classstd_1_1multiset_1_1const__iterator.xml
%feature("docstring") std::multiset::const_iterator "

STL iterator class. ";


// File: classstd_1_1wstring_1_1const__iterator.xml
%feature("docstring") std::wstring::const_iterator "

STL iterator class. ";


// File: classstd_1_1unordered__multiset_1_1const__iterator.xml
%feature("docstring") std::unordered_multiset::const_iterator "

STL iterator class. ";


// File: classstd_1_1vector_1_1const__iterator.xml
%feature("docstring") std::vector::const_iterator "

STL iterator class. ";


// File: classstd_1_1deque_1_1const__iterator.xml
%feature("docstring") std::deque::const_iterator "

STL iterator class. ";


// File: classstd_1_1list_1_1const__reverse__iterator.xml
%feature("docstring") std::list::const_reverse_iterator "

STL iterator class. ";


// File: classstd_1_1forward__list_1_1const__reverse__iterator.xml
%feature("docstring") std::forward_list::const_reverse_iterator "

STL iterator class. ";


// File: classstd_1_1map_1_1const__reverse__iterator.xml
%feature("docstring") std::map::const_reverse_iterator "

STL iterator class. ";


// File: classstd_1_1unordered__map_1_1const__reverse__iterator.xml
%feature("docstring") std::unordered_map::const_reverse_iterator "

STL iterator class. ";


// File: classstd_1_1multimap_1_1const__reverse__iterator.xml
%feature("docstring") std::multimap::const_reverse_iterator "

STL iterator class. ";


// File: classstd_1_1basic__string_1_1const__reverse__iterator.xml
%feature("docstring") std::basic_string::const_reverse_iterator "

STL iterator class. ";


// File: classstd_1_1unordered__multimap_1_1const__reverse__iterator.xml
%feature("docstring") std::unordered_multimap::const_reverse_iterator
"

STL iterator class. ";


// File: classstd_1_1set_1_1const__reverse__iterator.xml
%feature("docstring") std::set::const_reverse_iterator "

STL iterator class. ";


// File: classstd_1_1string_1_1const__reverse__iterator.xml
%feature("docstring") std::string::const_reverse_iterator "

STL iterator class. ";


// File: classstd_1_1unordered__set_1_1const__reverse__iterator.xml
%feature("docstring") std::unordered_set::const_reverse_iterator "

STL iterator class. ";


// File: classstd_1_1multiset_1_1const__reverse__iterator.xml
%feature("docstring") std::multiset::const_reverse_iterator "

STL iterator class. ";


// File: classstd_1_1wstring_1_1const__reverse__iterator.xml
%feature("docstring") std::wstring::const_reverse_iterator "

STL iterator class. ";


// File: classstd_1_1unordered__multiset_1_1const__reverse__iterator.xml
%feature("docstring") std::unordered_multiset::const_reverse_iterator
"

STL iterator class. ";


// File: classstd_1_1vector_1_1const__reverse__iterator.xml
%feature("docstring") std::vector::const_reverse_iterator "

STL iterator class. ";


// File: classstd_1_1deque_1_1const__reverse__iterator.xml
%feature("docstring") std::deque::const_reverse_iterator "

STL iterator class. ";


// File: classgdcm_1_1ConstCharWrapper.xml
%feature("docstring") gdcm::ConstCharWrapper "

Do not use me.

C++ includes: gdcmConstCharWrapper.h ";

%feature("docstring")  gdcm::ConstCharWrapper::ConstCharWrapper "gdcm::ConstCharWrapper::ConstCharWrapper(const char *i=0) ";


// File: classgdcm_1_1CP246ExplicitDataElement.xml
%feature("docstring") gdcm::CP246ExplicitDataElement "

Class to read/write a DataElement as CP246Explicit Data Element.

Some system are producing SQ, declare them as UN, but encode the SQ as
'Explicit' instead of Implicit

C++ includes: gdcmCP246ExplicitDataElement.h ";

%feature("docstring")  gdcm::CP246ExplicitDataElement::GetLength "VL
gdcm::CP246ExplicitDataElement::GetLength() const ";

%feature("docstring")  gdcm::CP246ExplicitDataElement::Read "std::istream& gdcm::CP246ExplicitDataElement::Read(std::istream &is)
";

%feature("docstring")  gdcm::CP246ExplicitDataElement::ReadPreValue "std::istream&
gdcm::CP246ExplicitDataElement::ReadPreValue(std::istream &is) ";

%feature("docstring")  gdcm::CP246ExplicitDataElement::ReadValue "std::istream& gdcm::CP246ExplicitDataElement::ReadValue(std::istream
&is, bool readvalues=true) ";

%feature("docstring")  gdcm::CP246ExplicitDataElement::ReadWithLength
"std::istream&
gdcm::CP246ExplicitDataElement::ReadWithLength(std::istream &is, VL
&length) ";


// File: classgdcm_1_1CryptoFactory.xml
%feature("docstring") gdcm::CryptoFactory "

Class to do handle the crypto factory.

GDCM needs to access in a platform independant way the user specified
crypto engine. It can be: CAPI (windows only)

OPENSSL (portable)

OPENSSLP7 (portable) By default the factory will try: CAPI if on
windows OPENSSL if possible OPENSSLP7 when older OpenSSL is used.

C++ includes: gdcmCryptoFactory.h ";

%feature("docstring")  gdcm::CryptoFactory::CreateCMSProvider "virtual CryptographicMessageSyntax*
gdcm::CryptoFactory::CreateCMSProvider()=0 ";


// File: classgdcm_1_1CryptographicMessageSyntax.xml
%feature("docstring") gdcm::CryptographicMessageSyntax "C++ includes:
gdcmCryptographicMessageSyntax.h ";

%feature("docstring")
gdcm::CryptographicMessageSyntax::CryptographicMessageSyntax "gdcm::CryptographicMessageSyntax::CryptographicMessageSyntax() ";

%feature("docstring")
gdcm::CryptographicMessageSyntax::~CryptographicMessageSyntax "virtual
gdcm::CryptographicMessageSyntax::~CryptographicMessageSyntax() ";

%feature("docstring")  gdcm::CryptographicMessageSyntax::Decrypt "virtual bool gdcm::CryptographicMessageSyntax::Decrypt(char *output,
size_t &outlen, const char *array, size_t len) const =0

decrypt content from a CMS envelopedData structure ";

%feature("docstring")  gdcm::CryptographicMessageSyntax::Encrypt "virtual bool gdcm::CryptographicMessageSyntax::Encrypt(char *output,
size_t &outlen, const char *array, size_t len) const =0

create a CMS envelopedData structure ";

%feature("docstring")  gdcm::CryptographicMessageSyntax::GetCipherType
"virtual CipherTypes
gdcm::CryptographicMessageSyntax::GetCipherType() const =0 ";

%feature("docstring")
gdcm::CryptographicMessageSyntax::ParseCertificateFile "virtual bool
gdcm::CryptographicMessageSyntax::ParseCertificateFile(const char
*filename)=0 ";

%feature("docstring")  gdcm::CryptographicMessageSyntax::ParseKeyFile
"virtual bool gdcm::CryptographicMessageSyntax::ParseKeyFile(const
char *filename)=0 ";

%feature("docstring")  gdcm::CryptographicMessageSyntax::SetCipherType
"virtual void
gdcm::CryptographicMessageSyntax::SetCipherType(CipherTypes type)=0 ";

%feature("docstring")  gdcm::CryptographicMessageSyntax::SetPassword "virtual bool gdcm::CryptographicMessageSyntax::SetPassword(const char
*pass, size_t passLen)=0 ";


// File: classgdcm_1_1CSAElement.xml
%feature("docstring") gdcm::CSAElement "

Class to represent a CSA Element.

See:   CSAHeader

C++ includes: gdcmCSAElement.h ";

%feature("docstring")  gdcm::CSAElement::CSAElement "gdcm::CSAElement::CSAElement(unsigned int kf=0) ";

%feature("docstring")  gdcm::CSAElement::CSAElement "gdcm::CSAElement::CSAElement(const CSAElement &_val) ";

%feature("docstring")  gdcm::CSAElement::GetByteValue "const
ByteValue* gdcm::CSAElement::GetByteValue() const

Return the Value of CSAElement as a ByteValue (if possible) WARNING:
: You need to check for NULL return value ";

%feature("docstring")  gdcm::CSAElement::GetKey "unsigned int
gdcm::CSAElement::GetKey() const

Set/Get Key. ";

%feature("docstring")  gdcm::CSAElement::GetName "const char*
gdcm::CSAElement::GetName() const

Set/Get Name. ";

%feature("docstring")  gdcm::CSAElement::GetNoOfItems "unsigned int
gdcm::CSAElement::GetNoOfItems() const

Set/Get NoOfItems. ";

%feature("docstring")  gdcm::CSAElement::GetSyngoDT "unsigned int
gdcm::CSAElement::GetSyngoDT() const

Set/Get SyngoDT. ";

%feature("docstring")  gdcm::CSAElement::GetValue "Value const&
gdcm::CSAElement::GetValue() const

Set/Get Value (bytes array, SQ of items, SQ of fragments): ";

%feature("docstring")  gdcm::CSAElement::GetValue "Value&
gdcm::CSAElement::GetValue() ";

%feature("docstring")  gdcm::CSAElement::GetVM "const VM&
gdcm::CSAElement::GetVM() const

Set/Get VM. ";

%feature("docstring")  gdcm::CSAElement::GetVR "VR const&
gdcm::CSAElement::GetVR() const

Set/Get VR. ";

%feature("docstring")  gdcm::CSAElement::IsEmpty "bool
gdcm::CSAElement::IsEmpty() const

Check if CSA Element is empty. ";

%feature("docstring")  gdcm::CSAElement::SetByteValue "void
gdcm::CSAElement::SetByteValue(const char *array, VL length)

Set. ";

%feature("docstring")  gdcm::CSAElement::SetKey "void
gdcm::CSAElement::SetKey(unsigned int key) ";

%feature("docstring")  gdcm::CSAElement::SetName "void
gdcm::CSAElement::SetName(const char *name) ";

%feature("docstring")  gdcm::CSAElement::SetNoOfItems "void
gdcm::CSAElement::SetNoOfItems(unsigned int items) ";

%feature("docstring")  gdcm::CSAElement::SetSyngoDT "void
gdcm::CSAElement::SetSyngoDT(unsigned int syngodt) ";

%feature("docstring")  gdcm::CSAElement::SetValue "void
gdcm::CSAElement::SetValue(Value const &vl) ";

%feature("docstring")  gdcm::CSAElement::SetVM "void
gdcm::CSAElement::SetVM(const VM &vm) ";

%feature("docstring")  gdcm::CSAElement::SetVR "void
gdcm::CSAElement::SetVR(VR const &vr) ";


// File: classgdcm_1_1CSAHeader.xml
%feature("docstring") gdcm::CSAHeader "

Class for CSAHeader.

SIEMENS store private information in tag (0x0029,0x10,\"SIEMENS CSA
HEADER\") this class is meant for user wishing to access values stored
within this private attribute. There are basically two main 'format'
for this attribute : SV10/NOMAGIC and DATASET_FORMAT SV10 and NOMAGIC
are from a user prospective identical, see CSAHeader.xml for possible
name / value stored in this format. DATASET_FORMAT is in fact simply
just another DICOM dataset (implicit) with -currently unknown- value.
This can be only be printed for now.

WARNING:  Everything you do with this code is at your own risk, since
decoding process was not written from specification documents.

the API of this class might change. Todo MrEvaProtocol in 29,1020
contains ^M that would be nice to get rid of on UNIX system...

See:   PDBHeader  External references: 5.1.3.2.4.1 MEDCOM History
Information and 5.1.4.3 CSA Non-Image Module
inhttp://tamsinfo.toshiba.com/docrequest/pdf/E.Soft_v2.0.pdf

C++ includes: gdcmCSAHeader.h ";

%feature("docstring")  gdcm::CSAHeader::CSAHeader "gdcm::CSAHeader::CSAHeader() ";

%feature("docstring")  gdcm::CSAHeader::~CSAHeader "gdcm::CSAHeader::~CSAHeader() ";

%feature("docstring")  gdcm::CSAHeader::FindCSAElementByName "bool
gdcm::CSAHeader::FindCSAElementByName(const char *name)

Return true if the CSA element matching 'name' is found or not
WARNING:  Case Sensitive ";

%feature("docstring")  gdcm::CSAHeader::GetCSAElementByName "const
CSAElement& gdcm::CSAHeader::GetCSAElementByName(const char *name)

Return the CSAElement corresponding to name 'name' WARNING:  Case
Sensitive ";

%feature("docstring")  gdcm::CSAHeader::GetDataSet "const DataSet&
gdcm::CSAHeader::GetDataSet() const

Return the DataSet output (use only if Format == DATASET_FORMAT ) ";

%feature("docstring")  gdcm::CSAHeader::GetFormat "CSAHeaderType
gdcm::CSAHeader::GetFormat() const

return the format of the CSAHeader SV10 and NOMAGIC are equivalent. ";

%feature("docstring")  gdcm::CSAHeader::GetInterfile "const char*
gdcm::CSAHeader::GetInterfile() const

Return the string output (use only if Format == Interfile) ";

%feature("docstring")  gdcm::CSAHeader::LoadFromDataElement "bool
gdcm::CSAHeader::LoadFromDataElement(DataElement const &de)

Decode the CSAHeader from element 'de'. ";

%feature("docstring")  gdcm::CSAHeader::Print "void
gdcm::CSAHeader::Print(std::ostream &os) const

Print the CSAHeader (use only if Format == SV10 or NOMAGIC) ";

%feature("docstring")  gdcm::CSAHeader::Read "std::istream&
gdcm::CSAHeader::Read(std::istream &is) ";

%feature("docstring")  gdcm::CSAHeader::Write "const std::ostream&
gdcm::CSAHeader::Write(std::ostream &os) const ";


// File: classgdcm_1_1CSAHeaderDict.xml
%feature("docstring") gdcm::CSAHeaderDict "

Class to represent a map of CSAHeaderDictEntry.

C++ includes: gdcmCSAHeaderDict.h ";

%feature("docstring")  gdcm::CSAHeaderDict::CSAHeaderDict "gdcm::CSAHeaderDict::CSAHeaderDict() ";

%feature("docstring")  gdcm::CSAHeaderDict::AddCSAHeaderDictEntry "void gdcm::CSAHeaderDict::AddCSAHeaderDictEntry(const
CSAHeaderDictEntry &de) ";

%feature("docstring")  gdcm::CSAHeaderDict::Begin "ConstIterator
gdcm::CSAHeaderDict::Begin() const ";

%feature("docstring")  gdcm::CSAHeaderDict::End "ConstIterator
gdcm::CSAHeaderDict::End() const ";

%feature("docstring")  gdcm::CSAHeaderDict::GetCSAHeaderDictEntry "const CSAHeaderDictEntry&
gdcm::CSAHeaderDict::GetCSAHeaderDictEntry(const char *name) const ";

%feature("docstring")  gdcm::CSAHeaderDict::IsEmpty "bool
gdcm::CSAHeaderDict::IsEmpty() const ";


// File: classgdcm_1_1CSAHeaderDictEntry.xml
%feature("docstring") gdcm::CSAHeaderDictEntry "

Class to represent an Entry in the Dict Does not really exist within
the DICOM definition, just a way to minimize storage and have a
mapping from gdcm::Tag to the needed information.

bla TODO FIXME: Need a PublicCSAHeaderDictEntry...indeed
CSAHeaderDictEntry has a notion of retired which does not exist in
PrivateCSAHeaderDictEntry...

See:   gdcm::Dict

C++ includes: gdcmCSAHeaderDictEntry.h ";

%feature("docstring")  gdcm::CSAHeaderDictEntry::CSAHeaderDictEntry "gdcm::CSAHeaderDictEntry::CSAHeaderDictEntry(const char *name=\"\", VR
const &vr=VR::INVALID, VM const &vm=VM::VM0, const char *desc=\"\") ";

%feature("docstring")  gdcm::CSAHeaderDictEntry::GetDescription "const char* gdcm::CSAHeaderDictEntry::GetDescription() const

Set/Get Description. ";

%feature("docstring")  gdcm::CSAHeaderDictEntry::GetName "const char*
gdcm::CSAHeaderDictEntry::GetName() const

Set/Get Name. ";

%feature("docstring")  gdcm::CSAHeaderDictEntry::GetVM "const VM&
gdcm::CSAHeaderDictEntry::GetVM() const

Set/Get VM. ";

%feature("docstring")  gdcm::CSAHeaderDictEntry::GetVR "const VR&
gdcm::CSAHeaderDictEntry::GetVR() const

Set/Get VR. ";

%feature("docstring")  gdcm::CSAHeaderDictEntry::SetDescription "void
gdcm::CSAHeaderDictEntry::SetDescription(const char *desc) ";

%feature("docstring")  gdcm::CSAHeaderDictEntry::SetName "void
gdcm::CSAHeaderDictEntry::SetName(const char *name) ";

%feature("docstring")  gdcm::CSAHeaderDictEntry::SetVM "void
gdcm::CSAHeaderDictEntry::SetVM(VM const &vm) ";

%feature("docstring")  gdcm::CSAHeaderDictEntry::SetVR "void
gdcm::CSAHeaderDictEntry::SetVR(const VR &vr) ";


// File: classgdcm_1_1CSAHeaderDictException.xml
%feature("docstring") gdcm::CSAHeaderDictException "C++ includes:
gdcmCSAHeaderDict.h ";


// File: classgdcm_1_1network_1_1CStoreRQ.xml
%feature("docstring") gdcm::network::CStoreRQ "

CStoreRQ this file defines the messages for the cecho action.

C++ includes: gdcmCStoreMessages.h ";

%feature("docstring")  gdcm::network::CStoreRQ::ConstructPDV "std::vector<PresentationDataValue>
gdcm::network::CStoreRQ::ConstructPDV(const ULConnection
&inConnection, const File &file) ";


// File: classgdcm_1_1network_1_1CStoreRSP.xml
%feature("docstring") gdcm::network::CStoreRSP "

CStoreRSP this file defines the messages for the cecho action.

C++ includes: gdcmCStoreMessages.h ";

%feature("docstring")  gdcm::network::CStoreRSP::ConstructPDV "std::vector<PresentationDataValue>
gdcm::network::CStoreRSP::ConstructPDV(const DataSet *inDataSet, const
BasePDU *inPC) ";


// File: classgdcm_1_1Curve.xml
%feature("docstring") gdcm::Curve "

Curve class to handle element 50xx,3000 Curve Data WARNING: This is
deprecated and lastly defined in PS 3.3 - 2004.

Examples: GE_DLX-8-MONO2-Multiframe-Jpeg_Lossless.dcm

GE_DLX-8-MONO2-Multiframe.dcm

gdcmSampleData/Philips_Medical_Images/integris_HV_5000/xa_integris.dcm

TOSHIBA-CurveData[1-3].dcm

C++ includes: gdcmCurve.h ";

%feature("docstring")  gdcm::Curve::Curve "gdcm::Curve::Curve() ";

%feature("docstring")  gdcm::Curve::Curve "gdcm::Curve::Curve(Curve
const &ov) ";

%feature("docstring")  gdcm::Curve::~Curve "gdcm::Curve::~Curve() ";

%feature("docstring")  gdcm::Curve::Decode "void
gdcm::Curve::Decode(std::istream &is, std::ostream &os) ";

%feature("docstring")  gdcm::Curve::GetAsPoints "void
gdcm::Curve::GetAsPoints(float *array) const ";

%feature("docstring")  gdcm::Curve::GetCurveDataDescriptor "std::vector<unsigned short> const&
gdcm::Curve::GetCurveDataDescriptor() const ";

%feature("docstring")  gdcm::Curve::GetDataValueRepresentation "unsigned short gdcm::Curve::GetDataValueRepresentation() const ";

%feature("docstring")  gdcm::Curve::GetDimensions "unsigned short
gdcm::Curve::GetDimensions() const ";

%feature("docstring")  gdcm::Curve::GetGroup "unsigned short
gdcm::Curve::GetGroup() const ";

%feature("docstring")  gdcm::Curve::GetNumberOfPoints "unsigned short
gdcm::Curve::GetNumberOfPoints() const ";

%feature("docstring")  gdcm::Curve::GetTypeOfData "const char*
gdcm::Curve::GetTypeOfData() const ";

%feature("docstring")  gdcm::Curve::GetTypeOfDataDescription "const
char* gdcm::Curve::GetTypeOfDataDescription() const ";

%feature("docstring")  gdcm::Curve::IsEmpty "bool
gdcm::Curve::IsEmpty() const ";

%feature("docstring")  gdcm::Curve::Print "void
gdcm::Curve::Print(std::ostream &) const ";

%feature("docstring")  gdcm::Curve::SetCoordinateStartValue "void
gdcm::Curve::SetCoordinateStartValue(unsigned short v) ";

%feature("docstring")  gdcm::Curve::SetCoordinateStepValue "void
gdcm::Curve::SetCoordinateStepValue(unsigned short v) ";

%feature("docstring")  gdcm::Curve::SetCurve "void
gdcm::Curve::SetCurve(const char *array, unsigned int length) ";

%feature("docstring")  gdcm::Curve::SetCurveDataDescriptor "void
gdcm::Curve::SetCurveDataDescriptor(const uint16_t *values, size_t
num) ";

%feature("docstring")  gdcm::Curve::SetCurveDescription "void
gdcm::Curve::SetCurveDescription(const char *curvedescription) ";

%feature("docstring")  gdcm::Curve::SetDataValueRepresentation "void
gdcm::Curve::SetDataValueRepresentation(unsigned short
datavaluerepresentation) ";

%feature("docstring")  gdcm::Curve::SetDimensions "void
gdcm::Curve::SetDimensions(unsigned short dimensions) ";

%feature("docstring")  gdcm::Curve::SetGroup "void
gdcm::Curve::SetGroup(unsigned short group) ";

%feature("docstring")  gdcm::Curve::SetNumberOfPoints "void
gdcm::Curve::SetNumberOfPoints(unsigned short numberofpoints) ";

%feature("docstring")  gdcm::Curve::SetTypeOfData "void
gdcm::Curve::SetTypeOfData(const char *typeofdata) ";

%feature("docstring")  gdcm::Curve::Update "void
gdcm::Curve::Update(const DataElement &de) ";


// File: classgdcm_1_1DataElement.xml
%feature("docstring") gdcm::DataElement "

Class to represent a Data Element either Implicit or Explicit.

DATA ELEMENT: A unit of information as defined by a single entry in
the data dictionary. An encoded Information Object Definition ( IOD)
Attribute that is composed of, at a minimum, three fields: a Data
Element Tag, a Value Length, and a Value Field. For some specific
Transfer Syntaxes, a Data Element also contains a VR Field where the
Value Representation of that Data Element is specified explicitly.

Design: A DataElement in GDCM always store VL ( Value Length) on a 32
bits integer even when VL is 16 bits

A DataElement always store the VR even for Implicit TS, in which case
VR is defaulted to VR::INVALID

For Item start/end (See 0xfffe tags), Value is NULL

See:   ExplicitDataElement ImplicitDataElement

C++ includes: gdcmDataElement.h ";

%feature("docstring")  gdcm::DataElement::DataElement "gdcm::DataElement::DataElement(const Tag &t=Tag(0), const VL &vl=0,
const VR &vr=VR::INVALID) ";

%feature("docstring")  gdcm::DataElement::DataElement "gdcm::DataElement::DataElement(const DataElement &_val) ";

%feature("docstring")  gdcm::DataElement::Clear "void
gdcm::DataElement::Clear()

Clear Data Element (make Value empty and invalidate Tag & VR) ";

%feature("docstring")  gdcm::DataElement::Empty "void
gdcm::DataElement::Empty()

Make Data Element empty (no Value) ";

%feature("docstring")  gdcm::DataElement::GetByteValue "const
ByteValue* gdcm::DataElement::GetByteValue() const

Return the Value of DataElement as a ByteValue (if possible) WARNING:
: You need to check for NULL return value ";

%feature("docstring")  gdcm::DataElement::GetLength "VL
gdcm::DataElement::GetLength() const ";

%feature("docstring")  gdcm::DataElement::GetSequenceOfFragments "const SequenceOfFragments* gdcm::DataElement::GetSequenceOfFragments()
const

Return the Value of DataElement as a Sequence Of Fragments (if
possible) WARNING:  : You need to check for NULL return value ";

%feature("docstring")  gdcm::DataElement::GetSequenceOfFragments "SequenceOfFragments* gdcm::DataElement::GetSequenceOfFragments() ";

%feature("docstring")  gdcm::DataElement::GetTag "const Tag&
gdcm::DataElement::GetTag() const

Get Tag. ";

%feature("docstring")  gdcm::DataElement::GetTag "Tag&
gdcm::DataElement::GetTag() ";

%feature("docstring")  gdcm::DataElement::GetValue "Value const&
gdcm::DataElement::GetValue() const

Set/Get Value (bytes array, SQ of items, SQ of fragments): ";

%feature("docstring")  gdcm::DataElement::GetValue "Value&
gdcm::DataElement::GetValue() ";

%feature("docstring")  gdcm::DataElement::GetValueAsSQ "SmartPointer<SequenceOfItems> gdcm::DataElement::GetValueAsSQ() const

Interpret the Value stored in the DataElement. This is more robust
(but also more expensive) to call this function rather than the
simpliest form: GetSequenceOfItems() It also return NULL when the
Value is NOT of type SequenceOfItems WARNING:  in case
GetSequenceOfItems() succeed the function return this value, otherwise
it creates a new SequenceOfItems, you should handle that in your case,
for instance: SmartPointer<SequenceOfItems> sqi = de.GetValueAsSQ();
";

%feature("docstring")  gdcm::DataElement::GetVL "const VL&
gdcm::DataElement::GetVL() const

Get VL. ";

%feature("docstring")  gdcm::DataElement::GetVL "VL&
gdcm::DataElement::GetVL() ";

%feature("docstring")  gdcm::DataElement::GetVR "VR const&
gdcm::DataElement::GetVR() const

Get VR do not set VR::SQ on bytevalue data element ";

%feature("docstring")  gdcm::DataElement::IsEmpty "bool
gdcm::DataElement::IsEmpty() const

Check if Data Element is empty. ";

%feature("docstring")  gdcm::DataElement::IsUndefinedLength "bool
gdcm::DataElement::IsUndefinedLength() const

return if Value Length if of undefined length ";

%feature("docstring")  gdcm::DataElement::Read "std::istream&
gdcm::DataElement::Read(std::istream &is) ";

%feature("docstring")  gdcm::DataElement::ReadOrSkip "std::istream&
gdcm::DataElement::ReadOrSkip(std::istream &is, std::set< Tag > const
&skiptags) ";

%feature("docstring")  gdcm::DataElement::ReadPreValue "std::istream&
gdcm::DataElement::ReadPreValue(std::istream &is, std::set< Tag >
const &skiptags) ";

%feature("docstring")  gdcm::DataElement::ReadValue "std::istream&
gdcm::DataElement::ReadValue(std::istream &is, std::set< Tag > const
&skiptags) ";

%feature("docstring")  gdcm::DataElement::ReadValueWithLength "std::istream& gdcm::DataElement::ReadValueWithLength(std::istream &is,
VL &length, std::set< Tag > const &skiptags) ";

%feature("docstring")  gdcm::DataElement::ReadWithLength "std::istream& gdcm::DataElement::ReadWithLength(std::istream &is, VL
&length) ";

%feature("docstring")  gdcm::DataElement::SetByteValue "void
gdcm::DataElement::SetByteValue(const char *array, VL length)

Set the byte value WARNING:  user need to read DICOM standard for an
understanding of: even padding

\\\\0 vs space padding By default even padding is achieved using \\\\0
regardless of the of VR ";

%feature("docstring")  gdcm::DataElement::SetTag "void
gdcm::DataElement::SetTag(const Tag &t)

Set Tag Use with cautious (need to match Part 6) ";

%feature("docstring")  gdcm::DataElement::SetValue "void
gdcm::DataElement::SetValue(Value const &vl)

WARNING:  you need to set the ValueLengthField explicitely ";

%feature("docstring")  gdcm::DataElement::SetVL "void
gdcm::DataElement::SetVL(const VL &vl)

Set VL Use with cautious (need to match Part 6), advanced user only
See:   SetByteValue ";

%feature("docstring")  gdcm::DataElement::SetVLToUndefined "void
gdcm::DataElement::SetVLToUndefined() ";

%feature("docstring")  gdcm::DataElement::SetVR "void
gdcm::DataElement::SetVR(VR const &vr)

Set VR Use with cautious (need to match Part 6), advanced user only vr
is a VR::VRALL (not a dual one such as OB_OW) ";

%feature("docstring")  gdcm::DataElement::Write "const std::ostream&
gdcm::DataElement::Write(std::ostream &os) const ";


// File: classgdcm_1_1DataElementException.xml
%feature("docstring") gdcm::DataElementException "C++ includes:
gdcmDataSet.h ";


// File: classgdcm_1_1DataEvent.xml
%feature("docstring") gdcm::DataEvent "

DataEvent.

C++ includes: gdcmDataEvent.h ";

%feature("docstring")  gdcm::DataEvent::DataEvent "gdcm::DataEvent::DataEvent(const char *bytes=0, size_t len=0) ";

%feature("docstring")  gdcm::DataEvent::DataEvent "gdcm::DataEvent::DataEvent(const Self &s) ";

%feature("docstring")  gdcm::DataEvent::~DataEvent "virtual
gdcm::DataEvent::~DataEvent() ";

%feature("docstring")  gdcm::DataEvent::CheckEvent "virtual bool
gdcm::DataEvent::CheckEvent(const ::gdcm::Event *e) const ";

%feature("docstring")  gdcm::DataEvent::GetData "const char*
gdcm::DataEvent::GetData() const ";

%feature("docstring")  gdcm::DataEvent::GetDataLength "size_t
gdcm::DataEvent::GetDataLength() const ";

%feature("docstring")  gdcm::DataEvent::GetEventName "virtual const
char* gdcm::DataEvent::GetEventName() const

Return the StringName associated with the event. ";

%feature("docstring")  gdcm::DataEvent::MakeObject "virtual
::gdcm::Event* gdcm::DataEvent::MakeObject() const

Create an Event of this type This method work as a Factory for
creating events of each particular type. ";

%feature("docstring")  gdcm::DataEvent::SetData "void
gdcm::DataEvent::SetData(const char *bytes, size_t len) ";


// File: classgdcm_1_1DataSet.xml
%feature("docstring") gdcm::DataSet "

Class to represent a Data Set (which contains Data Elements) A Data
Set represents an instance of a real world Information Object.

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
dataset with defined length.

WARNING:  a DataSet does not have a Transfer Syntax type, only a File
does.

C++ includes: gdcmDataSet.h ";

%feature("docstring")  gdcm::DataSet::Begin "ConstIterator
gdcm::DataSet::Begin() const ";

%feature("docstring")  gdcm::DataSet::Begin "Iterator
gdcm::DataSet::Begin() ";

%feature("docstring")  gdcm::DataSet::Clear "void
gdcm::DataSet::Clear() ";

%feature("docstring")  gdcm::DataSet::ComputeGroupLength "unsigned
int gdcm::DataSet::ComputeGroupLength(Tag const &tag) const ";

%feature("docstring")  gdcm::DataSet::End "ConstIterator
gdcm::DataSet::End() const ";

%feature("docstring")  gdcm::DataSet::End "Iterator
gdcm::DataSet::End() ";

%feature("docstring")  gdcm::DataSet::FindDataElement "bool
gdcm::DataSet::FindDataElement(const PrivateTag &t) const

Look up if private tag 't' is present in the dataset: ";

%feature("docstring")  gdcm::DataSet::FindDataElement "bool
gdcm::DataSet::FindDataElement(const Tag &t) const ";

%feature("docstring")  gdcm::DataSet::FindNextDataElement "const
DataElement& gdcm::DataSet::FindNextDataElement(const Tag &t) const ";

%feature("docstring")  gdcm::DataSet::GetDataElement "const
DataElement& gdcm::DataSet::GetDataElement(const Tag &t) const

Return the DataElement with Tag 't' WARNING:  : This only search at
the 'root level' of the DataSet ";

%feature("docstring")  gdcm::DataSet::GetDataElement "const
DataElement& gdcm::DataSet::GetDataElement(const PrivateTag &t) const

Return the dataelement. ";

%feature("docstring")  gdcm::DataSet::GetDES "const DataElementSet&
gdcm::DataSet::GetDES() const ";

%feature("docstring")  gdcm::DataSet::GetDES "DataElementSet&
gdcm::DataSet::GetDES() ";

%feature("docstring")  gdcm::DataSet::GetLength "VL
gdcm::DataSet::GetLength() const ";

%feature("docstring")  gdcm::DataSet::GetMediaStorage "MediaStorage
gdcm::DataSet::GetMediaStorage() const ";

%feature("docstring")  gdcm::DataSet::GetPrivateCreator "std::string
gdcm::DataSet::GetPrivateCreator(const Tag &t) const

Return the private creator of the private tag 't': ";

%feature("docstring")  gdcm::DataSet::Insert "void
gdcm::DataSet::Insert(const DataElement &de)

Insert a DataElement in the DataSet. WARNING:  : Tag need to be >= 0x8
to be considered valid data element ";

%feature("docstring")  gdcm::DataSet::IsEmpty "bool
gdcm::DataSet::IsEmpty() const

Returns if the dataset is empty. ";

%feature("docstring")  gdcm::DataSet::Print "void
gdcm::DataSet::Print(std::ostream &os, std::string const &indent=\"\")
const ";

%feature("docstring")  gdcm::DataSet::Read "std::istream&
gdcm::DataSet::Read(std::istream &is) ";

%feature("docstring")  gdcm::DataSet::ReadNested "std::istream&
gdcm::DataSet::ReadNested(std::istream &is) ";

%feature("docstring")  gdcm::DataSet::ReadSelectedPrivateTags "std::istream& gdcm::DataSet::ReadSelectedPrivateTags(std::istream &is,
const std::set< PrivateTag > &tags, bool readvalues=true) ";

%feature("docstring")
gdcm::DataSet::ReadSelectedPrivateTagsWithLength "std::istream&
gdcm::DataSet::ReadSelectedPrivateTagsWithLength(std::istream &is,
const std::set< PrivateTag > &tags, VL &length, bool readvalues=true)
";

%feature("docstring")  gdcm::DataSet::ReadSelectedTags "std::istream&
gdcm::DataSet::ReadSelectedTags(std::istream &is, const std::set< Tag
> &tags, bool readvalues=true) ";

%feature("docstring")  gdcm::DataSet::ReadSelectedTagsWithLength "std::istream& gdcm::DataSet::ReadSelectedTagsWithLength(std::istream
&is, const std::set< Tag > &tags, VL &length, bool readvalues=true) ";

%feature("docstring")  gdcm::DataSet::ReadUpToTag "std::istream&
gdcm::DataSet::ReadUpToTag(std::istream &is, const Tag &t, std::set<
Tag > const &skiptags) ";

%feature("docstring")  gdcm::DataSet::ReadUpToTagWithLength "std::istream& gdcm::DataSet::ReadUpToTagWithLength(std::istream &is,
const Tag &t, std::set< Tag > const &skiptags, VL &length) ";

%feature("docstring")  gdcm::DataSet::ReadWithLength "std::istream&
gdcm::DataSet::ReadWithLength(std::istream &is, VL &length) ";

%feature("docstring")  gdcm::DataSet::Remove "SizeType
gdcm::DataSet::Remove(const Tag &tag)

Completely remove a dataelement from the dataset. ";

%feature("docstring")  gdcm::DataSet::Replace "void
gdcm::DataSet::Replace(const DataElement &de)

Replace a dataelement with another one. ";

%feature("docstring")  gdcm::DataSet::ReplaceEmpty "void
gdcm::DataSet::ReplaceEmpty(const DataElement &de)

Only replace a DICOM attribute when it is missing or empty. ";

%feature("docstring")  gdcm::DataSet::Size "SizeType
gdcm::DataSet::Size() const ";

%feature("docstring")  gdcm::DataSet::Write "std::ostream const&
gdcm::DataSet::Write(std::ostream &os) const ";


// File: classgdcm_1_1DataSetEvent.xml
%feature("docstring") gdcm::DataSetEvent "

DataSetEvent Special type of event triggered during the DataSet
store/move process.

See:

C++ includes: gdcmDataSetEvent.h ";

%feature("docstring")  gdcm::DataSetEvent::DataSetEvent "gdcm::DataSetEvent::DataSetEvent(DataSet const *ds=NULL) ";

%feature("docstring")  gdcm::DataSetEvent::DataSetEvent "gdcm::DataSetEvent::DataSetEvent(const Self &s) ";

%feature("docstring")  gdcm::DataSetEvent::~DataSetEvent "virtual
gdcm::DataSetEvent::~DataSetEvent() ";

%feature("docstring")  gdcm::DataSetEvent::CheckEvent "virtual bool
gdcm::DataSetEvent::CheckEvent(const ::gdcm::Event *e) const ";

%feature("docstring")  gdcm::DataSetEvent::GetDataSet "DataSet const&
gdcm::DataSetEvent::GetDataSet() const ";

%feature("docstring")  gdcm::DataSetEvent::GetEventName "virtual
const char* gdcm::DataSetEvent::GetEventName() const

Return the StringName associated with the event. ";

%feature("docstring")  gdcm::DataSetEvent::MakeObject "virtual
::gdcm::Event* gdcm::DataSetEvent::MakeObject() const

Create an Event of this type This method work as a Factory for
creating events of each particular type. ";


// File: classgdcm_1_1DataSetHelper.xml
%feature("docstring") gdcm::DataSetHelper "

DataSetHelper (internal class, not intended for user level)

C++ includes: gdcmDataSetHelper.h ";


// File: classgdcm_1_1Decoder.xml
%feature("docstring") gdcm::Decoder "

Decoder.

C++ includes: gdcmDecoder.h ";

%feature("docstring")  gdcm::Decoder::~Decoder "virtual
gdcm::Decoder::~Decoder() ";

%feature("docstring")  gdcm::Decoder::CanDecode "virtual bool
gdcm::Decoder::CanDecode(TransferSyntax const &) const =0

Return whether this decoder support this transfer syntax (can decode
it) ";

%feature("docstring")  gdcm::Decoder::Decode "virtual bool
gdcm::Decoder::Decode(DataElement const &, DataElement &)

Decode. ";


// File: classgdcm_1_1DefinedTerms.xml
%feature("docstring") gdcm::DefinedTerms "

Defined Terms are used when the specified explicit Values may be
extended by implementors to include additional new Values. These new
Values shall be specified in the Conformance Statement (see PS 3.2)
and shall not have the same meaning as currently defined Values in
this standard. A Data Element with Defined Terms that does not contain
a Value equivalent to one of the Values currently specified in this
standard shall not be considered to have an invalid value. Note:
Interpretation Type ID (4008,0210) is an example of a Data Element
having Defined Terms. It is defined to have a Value that may be one of
the set of standard Values; REPORT or AMENDMENT (see PS 3.3). Because
this Data Element has Defined Terms other Interpretation Type IDs may
be defined by the implementor.

C++ includes: gdcmDefinedTerms.h ";

%feature("docstring")  gdcm::DefinedTerms::DefinedTerms "gdcm::DefinedTerms::DefinedTerms() ";


// File: classgdcm_1_1Defs.xml
%feature("docstring") gdcm::Defs "

FIXME I do not like the name ' Defs'.

bla

C++ includes: gdcmDefs.h ";

%feature("docstring")  gdcm::Defs::Defs "gdcm::Defs::Defs() ";

%feature("docstring")  gdcm::Defs::~Defs "gdcm::Defs::~Defs() ";

%feature("docstring")  gdcm::Defs::GetIODFromFile "const IOD&
gdcm::Defs::GetIODFromFile(const File &file) const ";

%feature("docstring")  gdcm::Defs::GetIODs "const IODs&
gdcm::Defs::GetIODs() const ";

%feature("docstring")  gdcm::Defs::GetIODs "IODs&
gdcm::Defs::GetIODs() ";

%feature("docstring")  gdcm::Defs::GetMacros "const Macros&
gdcm::Defs::GetMacros() const

Users should not directly use Macro. Macro are simply a way for DICOM
WG to re-use Tables. Macros are conviently wraped within Modules. See
gdcm::Module API directly ";

%feature("docstring")  gdcm::Defs::GetMacros "Macros&
gdcm::Defs::GetMacros() ";

%feature("docstring")  gdcm::Defs::GetModules "const Modules&
gdcm::Defs::GetModules() const ";

%feature("docstring")  gdcm::Defs::GetModules "Modules&
gdcm::Defs::GetModules() ";

%feature("docstring")  gdcm::Defs::GetTypeFromTag "Type
gdcm::Defs::GetTypeFromTag(const File &file, const Tag &tag) const ";

%feature("docstring")  gdcm::Defs::IsEmpty "bool
gdcm::Defs::IsEmpty() const ";

%feature("docstring")  gdcm::Defs::Verify "bool
gdcm::Defs::Verify(const File &file) const ";

%feature("docstring")  gdcm::Defs::Verify "bool
gdcm::Defs::Verify(const DataSet &ds) const ";


// File: classgdcm_1_1DeltaEncodingCodec.xml
%feature("docstring") gdcm::DeltaEncodingCodec "

DeltaEncodingCodec compression used by some private vendor.

C++ includes: gdcmDeltaEncodingCodec.h ";

%feature("docstring")  gdcm::DeltaEncodingCodec::DeltaEncodingCodec "gdcm::DeltaEncodingCodec::DeltaEncodingCodec() ";

%feature("docstring")  gdcm::DeltaEncodingCodec::~DeltaEncodingCodec "gdcm::DeltaEncodingCodec::~DeltaEncodingCodec() ";

%feature("docstring")  gdcm::DeltaEncodingCodec::CanDecode "bool
gdcm::DeltaEncodingCodec::CanDecode(TransferSyntax const &ts) ";

%feature("docstring")  gdcm::DeltaEncodingCodec::Decode "bool
gdcm::DeltaEncodingCodec::Decode(DataElement const &is, DataElement
&os)

Decode. ";


// File: classstd_1_1deque.xml
%feature("docstring") std::deque "

STL class. ";


// File: classgdcm_1_1DICOMDIR.xml
%feature("docstring") gdcm::DICOMDIR "

DICOMDIR class.

Structured for handling DICOMDIR

C++ includes: gdcmDICOMDIR.h ";

%feature("docstring")  gdcm::DICOMDIR::DICOMDIR "gdcm::DICOMDIR::DICOMDIR() ";

%feature("docstring")  gdcm::DICOMDIR::DICOMDIR "gdcm::DICOMDIR::DICOMDIR(const FileSet &fs) ";


// File: classgdcm_1_1DICOMDIRGenerator.xml
%feature("docstring") gdcm::DICOMDIRGenerator "

DICOMDIRGenerator class This is a STD-GEN-CD DICOMDIR generator. ref:
PS 3.11-2008 Annex D (Normative) - General Purpose CD-R and DVD
Interchange Profiles.

PS 3.11 - 2008 / D.3.2 Physical Medium And Medium Format The STD-GEN-
CD and STD-GEN-SEC-CD application profiles require the 120 mm CD-R
physical medium with the ISO/IEC 9660 Media Format, as defined in
PS3.12. See also PS 3.12 - 2008 / Annex F 120mm CD-R Medium
(Normative) and PS 3.10 - 2008 / 8 DICOM File Service / 8.1 FILE-SET

WARNING:  : PS 3.11 - 2008 / D.3.1 SOP Classes and Transfer Syntaxes
Composite Image & Stand-alone Storage are required to be stored as
Explicit VR Little Endian Uncompressed (1.2.840.10008.1.2.1). When a
DICOM file is found using another Transfer Syntax the generator will
simply stops.

Input files should be Explicit VR Little Endian

filenames should be valid VR::CS value (16 bytes, upper case ...)

Bug : There is a current limitation of not handling Referenced SOP
Class UID / Referenced SOP Instance UID simply because the Scanner
does not allow us See PS 3.11 / Table D.3-2 STD-GEN Additional
DICOMDIR Keys

C++ includes: gdcmDICOMDIRGenerator.h ";

%feature("docstring")  gdcm::DICOMDIRGenerator::DICOMDIRGenerator "gdcm::DICOMDIRGenerator::DICOMDIRGenerator() ";

%feature("docstring")  gdcm::DICOMDIRGenerator::~DICOMDIRGenerator "gdcm::DICOMDIRGenerator::~DICOMDIRGenerator() ";

%feature("docstring")  gdcm::DICOMDIRGenerator::Generate "bool
gdcm::DICOMDIRGenerator::Generate()

Main function to generate the DICOMDIR. ";

%feature("docstring")  gdcm::DICOMDIRGenerator::GetFile "File&
gdcm::DICOMDIRGenerator::GetFile() ";

%feature("docstring")  gdcm::DICOMDIRGenerator::SetDescriptor "void
gdcm::DICOMDIRGenerator::SetDescriptor(const char *d)

Set the File Set ID. WARNING:  this need to be a valid VR::CS value ";

%feature("docstring")  gdcm::DICOMDIRGenerator::SetFile "void
gdcm::DICOMDIRGenerator::SetFile(const File &f)

Set/Get file. The DICOMDIR file will be valid once a call to Generate
has been done. ";

%feature("docstring")  gdcm::DICOMDIRGenerator::SetFilenames "void
gdcm::DICOMDIRGenerator::SetFilenames(FilenamesType const &fns)

Set the list of filenames from which the DICOMDIR should be generated
from. ";

%feature("docstring")  gdcm::DICOMDIRGenerator::SetRootDirectory "void gdcm::DICOMDIRGenerator::SetRootDirectory(FilenameType const
&root)

Set the root directory from which the filenames should be considered.
";


// File: classgdcm_1_1Dict.xml
%feature("docstring") gdcm::Dict "

Class to represent a map of DictEntry.

bla TODO FIXME: For Element == 0x0 need to return Name = Group Length
ValueRepresentation = UL ValueMultiplicity = 1

C++ includes: gdcmDict.h ";

%feature("docstring")  gdcm::Dict::Dict "gdcm::Dict::Dict() ";

%feature("docstring")  gdcm::Dict::AddDictEntry "void
gdcm::Dict::AddDictEntry(const Tag &tag, const DictEntry &de) ";

%feature("docstring")  gdcm::Dict::Begin "ConstIterator
gdcm::Dict::Begin() const ";

%feature("docstring")  gdcm::Dict::End "ConstIterator
gdcm::Dict::End() const ";

%feature("docstring")  gdcm::Dict::GetDictEntry "const DictEntry&
gdcm::Dict::GetDictEntry(const Tag &tag) const ";

%feature("docstring")  gdcm::Dict::GetDictEntryByKeyword "const
DictEntry& gdcm::Dict::GetDictEntryByKeyword(const char *keyword, Tag
&tag) const

Lookup DictEntry by keyword. Even if DICOM standard defines keyword as
being unique. The lookup table is built on Tag. Therefore looking up a
DictEntry by Keyword is more inefficient than looking up by Tag. ";

%feature("docstring")  gdcm::Dict::GetDictEntryByName "const
DictEntry& gdcm::Dict::GetDictEntryByName(const char *name, Tag &tag)
const

Inefficient way of looking up tag by name. Technically DICOM does not
garantee uniqueness (and Curve / Overlay are there to prove it). But
most of the time name is in fact uniq and can be uniquely link to a
tag ";

%feature("docstring")  gdcm::Dict::GetKeywordFromTag "const char*
gdcm::Dict::GetKeywordFromTag(Tag const &tag) const

Function to return the Keyword from a Tag. ";

%feature("docstring")  gdcm::Dict::IsEmpty "bool
gdcm::Dict::IsEmpty() const ";


// File: classgdcm_1_1DictConverter.xml
%feature("docstring") gdcm::DictConverter "

Class to convert a .dic file into something else:

CXX code : embeded dict into shared lib (DICT_DEFAULT)

Debug mode (DICT_DEBUG)

XML dict (DICT_XML)

C++ includes: gdcmDictConverter.h ";

%feature("docstring")  gdcm::DictConverter::DictConverter "gdcm::DictConverter::DictConverter() ";

%feature("docstring")  gdcm::DictConverter::~DictConverter "gdcm::DictConverter::~DictConverter() ";

%feature("docstring")  gdcm::DictConverter::Convert "void
gdcm::DictConverter::Convert() ";

%feature("docstring")  gdcm::DictConverter::GetDictName "const
std::string& gdcm::DictConverter::GetDictName() const ";

%feature("docstring")  gdcm::DictConverter::GetInputFilename "const
std::string& gdcm::DictConverter::GetInputFilename() const ";

%feature("docstring")  gdcm::DictConverter::GetOutputFilename "const
std::string& gdcm::DictConverter::GetOutputFilename() const ";

%feature("docstring")  gdcm::DictConverter::GetOutputType "int
gdcm::DictConverter::GetOutputType() const ";

%feature("docstring")  gdcm::DictConverter::SetDictName "void
gdcm::DictConverter::SetDictName(const char *name) ";

%feature("docstring")  gdcm::DictConverter::SetInputFileName "void
gdcm::DictConverter::SetInputFileName(const char *filename) ";

%feature("docstring")  gdcm::DictConverter::SetOutputFileName "void
gdcm::DictConverter::SetOutputFileName(const char *filename) ";

%feature("docstring")  gdcm::DictConverter::SetOutputType "void
gdcm::DictConverter::SetOutputType(int type) ";


// File: classgdcm_1_1DictEntry.xml
%feature("docstring") gdcm::DictEntry "

Class to represent an Entry in the Dict Does not really exist within
the DICOM definition, just a way to minimize storage and have a
mapping from gdcm::Tag to the needed information.

bla TODO FIXME: Need a PublicDictEntry...indeed DictEntry has a notion
of retired which does not exist in PrivateDictEntry...

See:   gdcm::Dict

C++ includes: gdcmDictEntry.h ";

%feature("docstring")  gdcm::DictEntry::DictEntry "gdcm::DictEntry::DictEntry(const char *name=\"\", const char
*keyword=\"\", VR const &vr=VR::INVALID, VM const &vm=VM::VM0, bool
ret=false) ";

%feature("docstring")  gdcm::DictEntry::GetKeyword "const char*
gdcm::DictEntry::GetKeyword() const

same as GetName but without spaces... ";

%feature("docstring")  gdcm::DictEntry::GetName "const char*
gdcm::DictEntry::GetName() const

Set/Get Name. ";

%feature("docstring")  gdcm::DictEntry::GetRetired "bool
gdcm::DictEntry::GetRetired() const

Set/Get Retired flag. ";

%feature("docstring")  gdcm::DictEntry::GetVM "const VM&
gdcm::DictEntry::GetVM() const

Set/Get VM. ";

%feature("docstring")  gdcm::DictEntry::GetVR "const VR&
gdcm::DictEntry::GetVR() const

Set/Get VR. ";

%feature("docstring")  gdcm::DictEntry::IsUnique "bool
gdcm::DictEntry::IsUnique() const

Return whether the name of the DataElement can be considered to be
unique. As of 2008 all elements name were unique (except the
expclitely 'XX' ones) ";

%feature("docstring")  gdcm::DictEntry::SetElementXX "void
gdcm::DictEntry::SetElementXX(bool v)

Set whether element is shared in multiple elements (Source Image IDs
typically) ";

%feature("docstring")  gdcm::DictEntry::SetGroupXX "void
gdcm::DictEntry::SetGroupXX(bool v)

Set whether element is shared in multiple groups (Curve/Overlay
typically) ";

%feature("docstring")  gdcm::DictEntry::SetKeyword "void
gdcm::DictEntry::SetKeyword(const char *keyword) ";

%feature("docstring")  gdcm::DictEntry::SetName "void
gdcm::DictEntry::SetName(const char *name) ";

%feature("docstring")  gdcm::DictEntry::SetRetired "void
gdcm::DictEntry::SetRetired(bool retired) ";

%feature("docstring")  gdcm::DictEntry::SetVM "void
gdcm::DictEntry::SetVM(VM const &vm) ";

%feature("docstring")  gdcm::DictEntry::SetVR "void
gdcm::DictEntry::SetVR(const VR &vr) ";


// File: classgdcm_1_1DictPrinter.xml
%feature("docstring") gdcm::DictPrinter "

DictPrinter class.

C++ includes: gdcmDictPrinter.h ";

%feature("docstring")  gdcm::DictPrinter::DictPrinter "gdcm::DictPrinter::DictPrinter() ";

%feature("docstring")  gdcm::DictPrinter::~DictPrinter "gdcm::DictPrinter::~DictPrinter() ";

%feature("docstring")  gdcm::DictPrinter::Print "void
gdcm::DictPrinter::Print(std::ostream &os)

Print. ";


// File: classgdcm_1_1Dicts.xml
%feature("docstring") gdcm::Dicts "

Class to manipulate the sum of knowledge (all the dict user load)

bla

C++ includes: gdcmDicts.h ";

%feature("docstring")  gdcm::Dicts::Dicts "gdcm::Dicts::Dicts() ";

%feature("docstring")  gdcm::Dicts::~Dicts "gdcm::Dicts::~Dicts() ";

%feature("docstring")  gdcm::Dicts::GetCSAHeaderDict "const
CSAHeaderDict& gdcm::Dicts::GetCSAHeaderDict() const ";

%feature("docstring")  gdcm::Dicts::GetDictEntry "const DictEntry&
gdcm::Dicts::GetDictEntry(const Tag &tag, const char *owner=NULL)
const

NOT THREAD SAFE.

works for both public and private dicts: owner is null for public dict
WARNING:  owner need to be set to appropriate owner for call to work.
see ";

%feature("docstring")  gdcm::Dicts::GetDictEntry "const DictEntry&
gdcm::Dicts::GetDictEntry(const PrivateTag &tag) const ";

%feature("docstring")  gdcm::Dicts::GetPrivateDict "const
PrivateDict& gdcm::Dicts::GetPrivateDict() const ";

%feature("docstring")  gdcm::Dicts::GetPrivateDict "PrivateDict&
gdcm::Dicts::GetPrivateDict() ";

%feature("docstring")  gdcm::Dicts::GetPublicDict "const Dict&
gdcm::Dicts::GetPublicDict() const ";

%feature("docstring")  gdcm::Dicts::IsEmpty "bool
gdcm::Dicts::IsEmpty() const ";


// File: classgdcm_1_1network_1_1DIMSE.xml
%feature("docstring") gdcm::network::DIMSE "

DIMSE PS 3.7 - 2009 Annex E Command Dictionary (Normative) E.1
REGISTRY OF DICOM COMMAND ELEMENTS Table E.1-1 COMMAND FIELDS (PART 1)

C++ includes: gdcmDIMSE.h ";


// File: classgdcm_1_1DirectionCosines.xml
%feature("docstring") gdcm::DirectionCosines "

class to handle DirectionCosines

C++ includes: gdcmDirectionCosines.h ";

%feature("docstring")  gdcm::DirectionCosines::DirectionCosines "gdcm::DirectionCosines::DirectionCosines() ";

%feature("docstring")  gdcm::DirectionCosines::DirectionCosines "gdcm::DirectionCosines::DirectionCosines(const double dircos[6]) ";

%feature("docstring")  gdcm::DirectionCosines::~DirectionCosines "gdcm::DirectionCosines::~DirectionCosines() ";

%feature("docstring")  gdcm::DirectionCosines::ComputeDistAlongNormal
"double gdcm::DirectionCosines::ComputeDistAlongNormal(const double
ipp[3]) const

Compute the distance along the normal. ";

%feature("docstring")  gdcm::DirectionCosines::Cross "void
gdcm::DirectionCosines::Cross(double z[3]) const

Compute Cross product. ";

%feature("docstring")  gdcm::DirectionCosines::CrossDot "double
gdcm::DirectionCosines::CrossDot(DirectionCosines const &dc) const

Compute the Dot product of the two cross vector of both
DirectionCosines object. ";

%feature("docstring")  gdcm::DirectionCosines::Dot "double
gdcm::DirectionCosines::Dot() const

Compute Dot. ";

%feature("docstring")  gdcm::DirectionCosines::IsValid "bool
gdcm::DirectionCosines::IsValid() const

Return whether or not this is a valid direction cosines. ";

%feature("docstring")  gdcm::DirectionCosines::Normalize "void
gdcm::DirectionCosines::Normalize()

Normalize in-place. ";

%feature("docstring")  gdcm::DirectionCosines::Print "void
gdcm::DirectionCosines::Print(std::ostream &) const

Print. ";

%feature("docstring")  gdcm::DirectionCosines::SetFromString "bool
gdcm::DirectionCosines::SetFromString(const char *str)

Initialize from string str. It requires 6 floating point separated by
a backslash character. ";


// File: classgdcm_1_1Directory.xml
%feature("docstring") gdcm::Directory "

Class for manipulation directories.

This implementation provide a cross platform implementation for
manipulating directores: basically traversing directories and
harvesting files

will not take into account unix type hidden file recursive option will
not look into UNIX type hidden directory (those starting with a '.')

Since python or C# provide there own equivalent implementation, in
which case gdcm::Directory does not make much sense.

C++ includes: gdcmDirectory.h ";

%feature("docstring")  gdcm::Directory::Directory "gdcm::Directory::Directory() ";

%feature("docstring")  gdcm::Directory::~Directory "gdcm::Directory::~Directory() ";

%feature("docstring")  gdcm::Directory::GetDirectories "FilenamesType
const& gdcm::Directory::GetDirectories() const

Return the Directories traversed. ";

%feature("docstring")  gdcm::Directory::GetFilenames "FilenamesType
const& gdcm::Directory::GetFilenames() const

Set/Get the file names within the directory. ";

%feature("docstring")  gdcm::Directory::GetToplevel "FilenameType
const& gdcm::Directory::GetToplevel() const

Get the name of the toplevel directory. ";

%feature("docstring")  gdcm::Directory::Load "unsigned int
gdcm::Directory::Load(FilenameType const &name, bool recursive=false)

construct a list of filenames and subdirectory beneath directory: name
WARNING:  : hidden file and hidden directory are not loaded. ";

%feature("docstring")  gdcm::Directory::Print "void
gdcm::Directory::Print(std::ostream &os=std::cout) const

Print. ";


// File: classgdcm_1_1DirectoryHelper.xml
%feature("docstring") gdcm::DirectoryHelper "

DirectoryHelper this class is designed to help mitigate some of the
commonly performed operations on directories. namely: 1) the ability
to determine the number of series in a directory by what type of
series is present 2) the ability to find all ct series in a directory
3) the ability to find all mr series in a directory 4) to load a set
of DataSets from a series that's already been sorted by the IPP sorter
5) For rtstruct stuff, you need to know the sopinstanceuid of each z
plane, so there's a retrieval function for that 6) then a few other
functions for rtstruct writeouts.

C++ includes: gdcmDirectoryHelper.h ";


// File: classstd_1_1domain__error.xml
%feature("docstring") std::domain_error "

STL class. ";


// File: classgdcm_1_1DummyValueGenerator.xml
%feature("docstring") gdcm::DummyValueGenerator "

Class for generating dummy value.

See:   Anonymizer

C++ includes: gdcmDummyValueGenerator.h ";


// File: classgdcm_1_1Dumper.xml
%feature("docstring") gdcm::Dumper "

Codec class.

Use it to simply dump value read from the file. No interpretation is
done. But it is real fast ! Almost no overhead

C++ includes: gdcmDumper.h ";

%feature("docstring")  gdcm::Dumper::Dumper "gdcm::Dumper::Dumper()
";

%feature("docstring")  gdcm::Dumper::~Dumper "gdcm::Dumper::~Dumper()
";


// File: classgdcm_1_1Element.xml
%feature("docstring") gdcm::Element "

Element class.

TODO

C++ includes: gdcmElement.h ";

%feature("docstring")  gdcm::Element::GetAsDataElement "DataElement
gdcm::Element< TVR, TVM >::GetAsDataElement() const ";

%feature("docstring")  gdcm::Element::GetLength "unsigned long
gdcm::Element< TVR, TVM >::GetLength() const ";

%feature("docstring")  gdcm::Element::GetValue "const
VRToType<TVR>::Type& gdcm::Element< TVR, TVM >::GetValue(unsigned int
idx=0) const ";

%feature("docstring")  gdcm::Element::GetValue "VRToType<TVR>::Type&
gdcm::Element< TVR, TVM >::GetValue(unsigned int idx=0) ";

%feature("docstring")  gdcm::Element::GetValues "const
VRToType<TVR>::Type* gdcm::Element< TVR, TVM >::GetValues() const ";

%feature("docstring")  gdcm::Element::Print "void gdcm::Element< TVR,
TVM >::Print(std::ostream &_os) const ";

%feature("docstring")  gdcm::Element::Read "void gdcm::Element< TVR,
TVM >::Read(std::istream &_is) ";

%feature("docstring")  gdcm::Element::Set "void gdcm::Element< TVR,
TVM >::Set(Value const &v) ";

%feature("docstring")  gdcm::Element::SetFromDataElement "void
gdcm::Element< TVR, TVM >::SetFromDataElement(DataElement const &de)
";

%feature("docstring")  gdcm::Element::SetValue "void gdcm::Element<
TVR, TVM >::SetValue(typename VRToType< TVR >::Type v, unsigned int
idx=0) ";

%feature("docstring")  gdcm::Element::Write "void gdcm::Element< TVR,
TVM >::Write(std::ostream &_os) const ";


// File: classgdcm_1_1Element_3_01TVR_00_01VM_1_1VM1__2_01_4.xml
%feature("docstring") gdcm::Element< TVR, VM::VM1_2 > " C++ includes:
gdcmElement.h ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_2 >::SetLength "
void gdcm::Element< TVR, VM::VM1_2 >::SetLength(int len) ";


// File: classgdcm_1_1Element_3_01TVR_00_01VM_1_1VM1__n_01_4.xml
%feature("docstring") gdcm::Element< TVR, VM::VM1_n > " C++ includes:
gdcmElement.h ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_n >::Element "
gdcm::Element< TVR, VM::VM1_n >::Element() ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_n >::Element "
gdcm::Element< TVR, VM::VM1_n >::Element(const Element &_val) ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_n >::~Element "
gdcm::Element< TVR, VM::VM1_n >::~Element() ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_n
>::GetAsDataElement " DataElement gdcm::Element< TVR, VM::VM1_n
>::GetAsDataElement() const ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_n >::GetLength "
unsigned long gdcm::Element< TVR, VM::VM1_n >::GetLength() const ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_n >::GetValue "
const VRToType<TVR>::Type& gdcm::Element< TVR, VM::VM1_n
>::GetValue(unsigned int idx=0) const ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_n >::GetValue "
VRToType<TVR>::Type& gdcm::Element< TVR, VM::VM1_n
>::GetValue(unsigned int idx=0) ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_n >::Print " void
gdcm::Element< TVR, VM::VM1_n >::Print(std::ostream &_os) const ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_n >::Read " void
gdcm::Element< TVR, VM::VM1_n >::Read(std::istream &_is) ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_n >::Set " void
gdcm::Element< TVR, VM::VM1_n >::Set(Value const &v) ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_n >::SetArray "
void gdcm::Element< TVR, VM::VM1_n >::SetArray(const Type *array,
unsigned long len, bool save=false) ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_n
>::SetFromDataElement " void gdcm::Element< TVR, VM::VM1_n
>::SetFromDataElement(DataElement const &de) ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_n >::SetLength "
void gdcm::Element< TVR, VM::VM1_n >::SetLength(unsigned long len) ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_n >::SetValue "
void gdcm::Element< TVR, VM::VM1_n >::SetValue(typename VRToType< TVR
>::Type v, unsigned int idx=0) ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_n >::Write " void
gdcm::Element< TVR, VM::VM1_n >::Write(std::ostream &_os) const ";

%feature("docstring")  gdcm::Element< TVR, VM::VM1_n >::WriteASCII "
void gdcm::Element< TVR, VM::VM1_n >::WriteASCII(std::ostream &os)
const ";


// File: classgdcm_1_1Element_3_01TVR_00_01VM_1_1VM2__2n_01_4.xml
%feature("docstring") gdcm::Element< TVR, VM::VM2_2n > " C++ includes:
gdcmElement.h ";

%feature("docstring")  gdcm::Element< TVR, VM::VM2_2n >::SetLength "
void gdcm::Element< TVR, VM::VM2_2n >::SetLength(int len) ";


// File: classgdcm_1_1Element_3_01TVR_00_01VM_1_1VM2__n_01_4.xml
%feature("docstring") gdcm::Element< TVR, VM::VM2_n > " C++ includes:
gdcmElement.h ";

%feature("docstring")  gdcm::Element< TVR, VM::VM2_n >::SetLength "
void gdcm::Element< TVR, VM::VM2_n >::SetLength(int len) ";


// File: classgdcm_1_1Element_3_01TVR_00_01VM_1_1VM3__3n_01_4.xml
%feature("docstring") gdcm::Element< TVR, VM::VM3_3n > " C++ includes:
gdcmElement.h ";

%feature("docstring")  gdcm::Element< TVR, VM::VM3_3n >::SetLength "
void gdcm::Element< TVR, VM::VM3_3n >::SetLength(int len) ";


// File: classgdcm_1_1Element_3_01TVR_00_01VM_1_1VM3__n_01_4.xml
%feature("docstring") gdcm::Element< TVR, VM::VM3_n > " C++ includes:
gdcmElement.h ";

%feature("docstring")  gdcm::Element< TVR, VM::VM3_n >::SetLength "
void gdcm::Element< TVR, VM::VM3_n >::SetLength(int len) ";


// File: classgdcm_1_1Element_3_01VR_1_1AS_00_01VM_1_1VM5_01_4.xml
%feature("docstring") gdcm::Element< VR::AS, VM::VM5 > " C++ includes:
gdcmElement.h ";

%feature("docstring")  gdcm::Element< VR::AS, VM::VM5 >::GetLength "
unsigned long gdcm::Element< VR::AS, VM::VM5 >::GetLength() const ";

%feature("docstring")  gdcm::Element< VR::AS, VM::VM5 >::Print " void
gdcm::Element< VR::AS, VM::VM5 >::Print(std::ostream &_os) const ";


// File: classgdcm_1_1Element_3_01VR_1_1OB_00_01VM_1_1VM1_01_4.xml
%feature("docstring") gdcm::Element< VR::OB, VM::VM1 > " C++ includes:
gdcmElement.h ";


// File: classgdcm_1_1Element_3_01VR_1_1OW_00_01VM_1_1VM1_01_4.xml
%feature("docstring") gdcm::Element< VR::OW, VM::VM1 > " C++ includes:
gdcmElement.h ";


// File: classgdcm_1_1ElementDisableCombinations.xml
%feature("docstring") gdcm::ElementDisableCombinations "

A class which is used to produce compile errors for an invalid
combination of template parameters.

Invalid combinations have specialized declarations with no definition.

C++ includes: gdcmElement.h ";


// File: classgdcm_1_1ElementDisableCombinations_3_01VR_1_1OB_00_01VM_1_1VM1__n_01_4.xml
%feature("docstring") gdcm::ElementDisableCombinations< VR::OB,
VM::VM1_n > " C++ includes: gdcmElement.h ";


// File: classgdcm_1_1ElementDisableCombinations_3_01VR_1_1OW_00_01VM_1_1VM1__n_01_4.xml
%feature("docstring") gdcm::ElementDisableCombinations< VR::OW,
VM::VM1_n > " C++ includes: gdcmElement.h ";


// File: classgdcm_1_1EncapsulatedDocument.xml
%feature("docstring") gdcm::EncapsulatedDocument "

EncapsulatedDocument.

C++ includes: gdcmEncapsulatedDocument.h ";

%feature("docstring")
gdcm::EncapsulatedDocument::EncapsulatedDocument "gdcm::EncapsulatedDocument::EncapsulatedDocument() ";


// File: classgdcm_1_1EncodingImplementation_3_01VR_1_1VRASCII_01_4.xml
%feature("docstring") gdcm::EncodingImplementation< VR::VRASCII > "
C++ includes: gdcmElement.h ";

%feature("docstring")  gdcm::EncodingImplementation< VR::VRASCII
>::Write " void gdcm::EncodingImplementation< VR::VRASCII
>::Write(const float *data, unsigned long length, std::ostream &_os)
";

%feature("docstring")  gdcm::EncodingImplementation< VR::VRASCII
>::Write " void gdcm::EncodingImplementation< VR::VRASCII
>::Write(const double *data, unsigned long length, std::ostream &_os)
";


// File: classgdcm_1_1EncodingImplementation_3_01VR_1_1VRBINARY_01_4.xml
%feature("docstring") gdcm::EncodingImplementation< VR::VRBINARY > "
C++ includes: gdcmElement.h ";


// File: classgdcm_1_1EndEvent.xml
%feature("docstring") gdcm::EndEvent "C++ includes: gdcmEvent.h ";


// File: classgdcm_1_1EnumeratedValues.xml
%feature("docstring") gdcm::EnumeratedValues "

Element. A Data Element with Enumerated Values that does not have a
Value equivalent to one of the Values specified in this standard has
an invalid value within the scope of a specific Information Object/SOP
Class definition. Note:

Patient Sex (0010, 0040) is an example of a Data Element having
Enumerated Values. It is defined to have a Value that is either \"M\",
\"F\", or \"O\" (see PS 3.3). No other Value shall be given to this
Data Element.

Future modifications of this standard may add to the set of allowed
values for Data Elements with Enumerated Values. Such additions by
themselves may or may not require a change in SOP Class UIDs,
depending on the semantics of the Data Element.

C++ includes: gdcmEnumeratedValues.h ";

%feature("docstring")  gdcm::EnumeratedValues::EnumeratedValues "gdcm::EnumeratedValues::EnumeratedValues() ";


// File: classstd_1_1error__category.xml
%feature("docstring") std::error_category "

STL class. ";


// File: classstd_1_1error__code.xml
%feature("docstring") std::error_code "

STL class. ";


// File: classstd_1_1error__condition.xml
%feature("docstring") std::error_condition "

STL class. ";


// File: classgdcm_1_1Event.xml
%feature("docstring") gdcm::Event "

superclass for callback/observer methods

See:   Command Subject

C++ includes: gdcmEvent.h ";

%feature("docstring")  gdcm::Event::Event "gdcm::Event::Event() ";

%feature("docstring")  gdcm::Event::Event "gdcm::Event::Event(const
Event &) ";

%feature("docstring")  gdcm::Event::~Event "virtual
gdcm::Event::~Event() ";

%feature("docstring")  gdcm::Event::CheckEvent "virtual bool
gdcm::Event::CheckEvent(const Event *) const =0

Check if given event matches or derives from this event. ";

%feature("docstring")  gdcm::Event::GetEventName "virtual const char*
gdcm::Event::GetEventName(void) const =0

Return the StringName associated with the event. ";

%feature("docstring")  gdcm::Event::MakeObject "virtual Event*
gdcm::Event::MakeObject() const =0

Create an Event of this type This method work as a Factory for
creating events of each particular type. ";

%feature("docstring")  gdcm::Event::Print "virtual void
gdcm::Event::Print(std::ostream &os) const

Print Event information. This method can be overridden by specific
Event subtypes. The default is to print out the type of the event. ";


// File: classgdcm_1_1Exception.xml
%feature("docstring") gdcm::Exception "

Exception.

Standard exception handling object. Its copy-constructor and
assignment operator are generated by the compiler.

C++ includes: gdcmException.h ";

%feature("docstring")  gdcm::Exception::Exception "gdcm::Exception::Exception(const char *desc=\"None\", const char
*file=__FILE__, unsigned int lineNumber=__LINE__, const char
*func=\"\")

Explicit constructor, initializing the description and the text
returned by what(). The last parameter is ignored for the time being.
It may be used to specify the function where the exception was thrown.
";

%feature("docstring")  gdcm::Exception::~Exception "virtual
gdcm::Exception::~Exception()  throw ()";

%feature("docstring")  gdcm::Exception::GetDescription "const char*
gdcm::Exception::GetDescription() const

Return the Description. ";

%feature("docstring")  gdcm::Exception::what "const char*
gdcm::Exception::what() const  throw () what implementation ";


// File: classstd_1_1exception.xml
%feature("docstring") std::exception "

STL class. ";


// File: classgdcm_1_1ExitEvent.xml
%feature("docstring") gdcm::ExitEvent "C++ includes: gdcmEvent.h ";


// File: classgdcm_1_1ExplicitDataElement.xml
%feature("docstring") gdcm::ExplicitDataElement "

Class to read/write a DataElement as Explicit Data Element.

bla

C++ includes: gdcmExplicitDataElement.h ";

%feature("docstring")  gdcm::ExplicitDataElement::GetLength "VL
gdcm::ExplicitDataElement::GetLength() const ";

%feature("docstring")  gdcm::ExplicitDataElement::Read "std::istream&
gdcm::ExplicitDataElement::Read(std::istream &is) ";

%feature("docstring")  gdcm::ExplicitDataElement::ReadPreValue "std::istream& gdcm::ExplicitDataElement::ReadPreValue(std::istream
&is) ";

%feature("docstring")  gdcm::ExplicitDataElement::ReadValue "std::istream& gdcm::ExplicitDataElement::ReadValue(std::istream &is,
bool readvalues=true) ";

%feature("docstring")  gdcm::ExplicitDataElement::ReadWithLength "std::istream& gdcm::ExplicitDataElement::ReadWithLength(std::istream
&is, VL &length) ";

%feature("docstring")  gdcm::ExplicitDataElement::Write "const
std::ostream& gdcm::ExplicitDataElement::Write(std::ostream &os) const
";


// File: classgdcm_1_1ExplicitImplicitDataElement.xml
%feature("docstring") gdcm::ExplicitImplicitDataElement "

Class to read/write a DataElement as ExplicitImplicit Data Element.

This only happen for some Philips images Should I derive from
ExplicitDataElement instead ? This is the class that is the closest
the GDCM1.x parser. At each element we try first to read it as
explicit, if this fails, then we try again as an implicit element.

C++ includes: gdcmExplicitImplicitDataElement.h ";

%feature("docstring")  gdcm::ExplicitImplicitDataElement::GetLength "VL gdcm::ExplicitImplicitDataElement::GetLength() const ";

%feature("docstring")  gdcm::ExplicitImplicitDataElement::Read "std::istream& gdcm::ExplicitImplicitDataElement::Read(std::istream
&is) ";

%feature("docstring")  gdcm::ExplicitImplicitDataElement::ReadPreValue
"std::istream&
gdcm::ExplicitImplicitDataElement::ReadPreValue(std::istream &is) ";

%feature("docstring")  gdcm::ExplicitImplicitDataElement::ReadValue "std::istream&
gdcm::ExplicitImplicitDataElement::ReadValue(std::istream &is, bool
readvalues=true) ";

%feature("docstring")
gdcm::ExplicitImplicitDataElement::ReadWithLength "std::istream&
gdcm::ExplicitImplicitDataElement::ReadWithLength(std::istream &is, VL
&length) ";


// File: classstd_1_1ios__base_1_1failure.xml
%feature("docstring") std::ios_base::failure "

STL class. ";


// File: classgdcm_1_1Fiducials.xml
%feature("docstring") gdcm::Fiducials "

Fiducials.

C++ includes: gdcmFiducials.h ";

%feature("docstring")  gdcm::Fiducials::Fiducials "gdcm::Fiducials::Fiducials() ";


// File: classgdcm_1_1File.xml
%feature("docstring") gdcm::File "

a DICOM File See PS 3.10 File: A File is an ordered string of zero or
more bytes, where the first byte is at the beginning of the file and
the last byte at the end of the File. Files are identified by a unique
File ID and may by written, read and/or deleted.

See:   Reader Writer

C++ includes: gdcmFile.h ";

%feature("docstring")  gdcm::File::File "gdcm::File::File() ";

%feature("docstring")  gdcm::File::GetDataSet "const DataSet&
gdcm::File::GetDataSet() const

Get Data Set. ";

%feature("docstring")  gdcm::File::GetDataSet "DataSet&
gdcm::File::GetDataSet()

Get Data Set. ";

%feature("docstring")  gdcm::File::GetHeader "const
FileMetaInformation& gdcm::File::GetHeader() const

Get File Meta Information. ";

%feature("docstring")  gdcm::File::GetHeader "FileMetaInformation&
gdcm::File::GetHeader()

Get File Meta Information. ";

%feature("docstring")  gdcm::File::Read "std::istream&
gdcm::File::Read(std::istream &is)

Read. ";

%feature("docstring")  gdcm::File::SetDataSet "void
gdcm::File::SetDataSet(const DataSet &ds)

Set Data Set. ";

%feature("docstring")  gdcm::File::SetHeader "void
gdcm::File::SetHeader(const FileMetaInformation &fmi)

Set File Meta Information. ";

%feature("docstring")  gdcm::File::Write "std::ostream const&
gdcm::File::Write(std::ostream &os) const

Write. ";


// File: classgdcm_1_1FileAnonymizer.xml
%feature("docstring") gdcm::FileAnonymizer "

FileAnonymizer.

This Anonymizer is a file-based Anonymizer. It requires a valid DICOM
file and will use the Value Length to skip over any information.

It will not load the DICOM dataset taken from SetInputFileName() into
memory and should consume much less memory than Anonymizer.

WARNING:  : Each time you call Replace() with a value. This value will
copied, and stored in memory. The behavior is not ideal for extremely
large data (larger than memory size). This class is really meant to
take a large DICOM input file and then only changed some small
attribute.  caveats: This class will NOT work with unordered
attributes in a DICOM File,

This class does neither recompute nor update the Group Length element,

This class currently does not update the File Meta Information header.

Only strict inplace Replace operation is supported when input and
output file are the same.

C++ includes: gdcmFileAnonymizer.h ";

%feature("docstring")  gdcm::FileAnonymizer::FileAnonymizer "gdcm::FileAnonymizer::FileAnonymizer() ";

%feature("docstring")  gdcm::FileAnonymizer::~FileAnonymizer "gdcm::FileAnonymizer::~FileAnonymizer() ";

%feature("docstring")  gdcm::FileAnonymizer::Empty "void
gdcm::FileAnonymizer::Empty(Tag const &t)

Make Tag t empty Warning: does not handle SQ element ";

%feature("docstring")  gdcm::FileAnonymizer::Remove "void
gdcm::FileAnonymizer::Remove(Tag const &t)

remove a tag (even a SQ can be removed) ";

%feature("docstring")  gdcm::FileAnonymizer::Replace "void
gdcm::FileAnonymizer::Replace(Tag const &t, const char *value_str)

Replace tag with another value, if tag is not found it will be
created: WARNING: this function can only execute if tag is a VRASCII
WARNING: Do not ever try to write a value in a SQ Data Element ! ";

%feature("docstring")  gdcm::FileAnonymizer::Replace "void
gdcm::FileAnonymizer::Replace(Tag const &t, const char *value_data, VL
const &vl)

when the value contains \\\\0, it is a good idea to specify the
length. This function is required when dealing with VRBINARY tag ";

%feature("docstring")  gdcm::FileAnonymizer::SetInputFileName "void
gdcm::FileAnonymizer::SetInputFileName(const char *filename_native)

Set input filename. ";

%feature("docstring")  gdcm::FileAnonymizer::SetOutputFileName "void
gdcm::FileAnonymizer::SetOutputFileName(const char *filename_native)

Set output filename. ";

%feature("docstring")  gdcm::FileAnonymizer::Write "bool
gdcm::FileAnonymizer::Write()

Write the output file. ";


// File: classgdcm_1_1FileChangeTransferSyntax.xml
%feature("docstring") gdcm::FileChangeTransferSyntax "

FileChangeTransferSyntax.

This class is a file-based (limited) replacement of the in-memory
ImageChangeTransferSyntax.

This class provide a file-based compression-only mecanism. It will
take in an uncompressed DICOM image file (Pixel Data element). Then
produced as output a compressed DICOM file (Transfer Syntax will be
updated).

Currently it supports the following transfer syntax:
JPEGLosslessProcess14_1

C++ includes: gdcmFileChangeTransferSyntax.h ";

%feature("docstring")
gdcm::FileChangeTransferSyntax::FileChangeTransferSyntax "gdcm::FileChangeTransferSyntax::FileChangeTransferSyntax() ";

%feature("docstring")
gdcm::FileChangeTransferSyntax::~FileChangeTransferSyntax "gdcm::FileChangeTransferSyntax::~FileChangeTransferSyntax() ";

%feature("docstring")  gdcm::FileChangeTransferSyntax::Change "bool
gdcm::FileChangeTransferSyntax::Change()

Change the transfer syntax. ";

%feature("docstring")  gdcm::FileChangeTransferSyntax::GetCodec "ImageCodec* gdcm::FileChangeTransferSyntax::GetCodec()

Retrieve the actual codec (valid after calling SetTransferSyntax) Only
advanced users should call this function. ";

%feature("docstring")
gdcm::FileChangeTransferSyntax::SetInputFileName "void
gdcm::FileChangeTransferSyntax::SetInputFileName(const char
*filename_native)

Set input filename (raw DICOM) ";

%feature("docstring")
gdcm::FileChangeTransferSyntax::SetOutputFileName "void
gdcm::FileChangeTransferSyntax::SetOutputFileName(const char
*filename_native)

Set output filename (target compressed DICOM) ";

%feature("docstring")
gdcm::FileChangeTransferSyntax::SetTransferSyntax "void
gdcm::FileChangeTransferSyntax::SetTransferSyntax(TransferSyntax const
&ts)

Specify the Target Transfer Syntax. ";


// File: classgdcm_1_1FileDerivation.xml
%feature("docstring") gdcm::FileDerivation "

FileDerivation class See PS 3.16 - 2008 For the list of Code Value
that can be used for in Derivation Code Sequence.

URL:http://medical.nema.org/medical/dicom/2008/08_16pu.pdf

DICOM Part 16 has two Context Groups CID 7202 and CID 7203 which
contain a set of codes defining reason for a source image reference
(ie. reason code for referenced image sequence) and a coded
description of the deriation applied to the new image data from the
original. Both these context groups are extensible.

File Derivation is compulsary when creating a lossy derived image.

C++ includes: gdcmFileDerivation.h ";

%feature("docstring")  gdcm::FileDerivation::FileDerivation "gdcm::FileDerivation::FileDerivation() ";

%feature("docstring")  gdcm::FileDerivation::~FileDerivation "gdcm::FileDerivation::~FileDerivation() ";

%feature("docstring")  gdcm::FileDerivation::AddReference "bool
gdcm::FileDerivation::AddReference(const char *referencedsopclassuid,
const char *referencedsopinstanceuid)

Create the proper reference. Need to pass the original SOP Class UID
and the original SOP Instance UID, so that those value can be used as
Reference. WARNING:  referencedsopclassuid and
referencedsopinstanceuid needs to be \\\\0 padded. This is not
compatible with how ByteValue->GetPointer works. ";

%feature("docstring")  gdcm::FileDerivation::Derive "bool
gdcm::FileDerivation::Derive()

Change. ";

%feature("docstring")  gdcm::FileDerivation::GetFile "File&
gdcm::FileDerivation::GetFile() ";

%feature("docstring")  gdcm::FileDerivation::GetFile "const File&
gdcm::FileDerivation::GetFile() const ";

%feature("docstring")
gdcm::FileDerivation::SetDerivationCodeSequenceCodeValue "void
gdcm::FileDerivation::SetDerivationCodeSequenceCodeValue(unsigned int
codevalue)

Specify the Derivation Code Sequence Code Value. Eg 113040. ";

%feature("docstring")  gdcm::FileDerivation::SetDerivationDescription
"void gdcm::FileDerivation::SetDerivationDescription(const char *dd)

Specify the Derivation Description. Eg \"lossy conversion\". ";

%feature("docstring")  gdcm::FileDerivation::SetFile "void
gdcm::FileDerivation::SetFile(const File &f)

Set/Get File. ";

%feature("docstring")
gdcm::FileDerivation::SetPurposeOfReferenceCodeSequenceCodeValue "void
gdcm::FileDerivation::SetPurposeOfReferenceCodeSequenceCodeValue(unsigned
int codevalue)

Specify the Purpose Of Reference Code Value. Eg. 121320. ";


// File: classgdcm_1_1FileExplicitFilter.xml
%feature("docstring") gdcm::FileExplicitFilter "

FileExplicitFilter class After changing a file from Implicit to
Explicit representation (see ImageChangeTransferSyntax) one operation
is to make sure the VR of each DICOM attribute are accurate and do
match the one from PS 3.6. Indeed when a file is written in Implicit
reprensentation, the VR is not stored directly in the file.

WARNING:  changing an implicit dataset to an explicit dataset is NOT a
trivial task of simply changing the VR to the dict one: One has to
make sure SQ is properly set

One has to recompute the explicit length SQ

One has to make sure that VR is valid for the encoding

One has to make sure that VR 16bits can store the original value
length

C++ includes: gdcmFileExplicitFilter.h ";

%feature("docstring")  gdcm::FileExplicitFilter::FileExplicitFilter "gdcm::FileExplicitFilter::FileExplicitFilter() ";

%feature("docstring")  gdcm::FileExplicitFilter::~FileExplicitFilter "gdcm::FileExplicitFilter::~FileExplicitFilter() ";

%feature("docstring")  gdcm::FileExplicitFilter::Change "bool
gdcm::FileExplicitFilter::Change()

Set FMI Transfer Syntax.

Change ";

%feature("docstring")  gdcm::FileExplicitFilter::GetFile "File&
gdcm::FileExplicitFilter::GetFile() ";

%feature("docstring")  gdcm::FileExplicitFilter::SetChangePrivateTags
"void gdcm::FileExplicitFilter::SetChangePrivateTags(bool b)

Decide whether or not to VR'ify private tags. ";

%feature("docstring")  gdcm::FileExplicitFilter::SetFile "void
gdcm::FileExplicitFilter::SetFile(const File &f)

Set/Get File. ";

%feature("docstring")
gdcm::FileExplicitFilter::SetRecomputeItemLength "void
gdcm::FileExplicitFilter::SetRecomputeItemLength(bool b)

By default set Sequence & Item length to Undefined to avoid
recomputing length: ";

%feature("docstring")
gdcm::FileExplicitFilter::SetRecomputeSequenceLength "void
gdcm::FileExplicitFilter::SetRecomputeSequenceLength(bool b) ";

%feature("docstring")  gdcm::FileExplicitFilter::SetUseVRUN "void
gdcm::FileExplicitFilter::SetUseVRUN(bool b)

When VR=16bits in explicit but Implicit has a 32bits length, use
VR=UN. ";


// File: classgdcm_1_1FileMetaInformation.xml
%feature("docstring") gdcm::FileMetaInformation "

Class to represent a File Meta Information.

FileMetaInformation is a Explicit Structured Set. Whenever the file
contains an ImplicitDataElement DataSet, a conversion will take place.

Definition: The File Meta Information includes identifying information
on the encapsulated Data Set. This header consists of a 128 byte File
Preamble, followed by a 4 byte DICOM prefix, followed by the File Meta
Elements shown in Table 7.1-1. This header shall be present in every
DICOM file.

See:   Writer Reader

C++ includes: gdcmFileMetaInformation.h ";

%feature("docstring")  gdcm::FileMetaInformation::FileMetaInformation
"gdcm::FileMetaInformation::FileMetaInformation() ";

%feature("docstring")  gdcm::FileMetaInformation::FileMetaInformation
"gdcm::FileMetaInformation::FileMetaInformation(FileMetaInformation
const &fmi) ";

%feature("docstring")  gdcm::FileMetaInformation::FillFromDataSet "void gdcm::FileMetaInformation::FillFromDataSet(DataSet const &ds)

Construct a FileMetaInformation from an already existing DataSet: ";

%feature("docstring")
gdcm::FileMetaInformation::GetDataSetTransferSyntax "const
TransferSyntax& gdcm::FileMetaInformation::GetDataSetTransferSyntax()
const ";

%feature("docstring")  gdcm::FileMetaInformation::GetFullLength "VL
gdcm::FileMetaInformation::GetFullLength() const ";

%feature("docstring")  gdcm::FileMetaInformation::GetMediaStorage "MediaStorage gdcm::FileMetaInformation::GetMediaStorage() const ";

%feature("docstring")
gdcm::FileMetaInformation::GetMediaStorageAsString "std::string
gdcm::FileMetaInformation::GetMediaStorageAsString() const ";

%feature("docstring")  gdcm::FileMetaInformation::GetMetaInformationTS
"TransferSyntax::NegociatedType
gdcm::FileMetaInformation::GetMetaInformationTS() const ";

%feature("docstring")  gdcm::FileMetaInformation::GetPreamble "const
Preamble& gdcm::FileMetaInformation::GetPreamble() const

Get Preamble. ";

%feature("docstring")  gdcm::FileMetaInformation::GetPreamble "Preamble& gdcm::FileMetaInformation::GetPreamble() ";

%feature("docstring")  gdcm::FileMetaInformation::Insert "void
gdcm::FileMetaInformation::Insert(const DataElement &de)

Insert a DataElement in the DataSet. WARNING:  : Tag need to be >= 0x8
to be considered valid data element ";

%feature("docstring")  gdcm::FileMetaInformation::IsValid "bool
gdcm::FileMetaInformation::IsValid() const ";

%feature("docstring")  gdcm::FileMetaInformation::Read "std::istream&
gdcm::FileMetaInformation::Read(std::istream &is)

Read. ";

%feature("docstring")  gdcm::FileMetaInformation::ReadCompat "std::istream& gdcm::FileMetaInformation::ReadCompat(std::istream &is)
";

%feature("docstring")  gdcm::FileMetaInformation::Replace "void
gdcm::FileMetaInformation::Replace(const DataElement &de)

Replace a dataelement with another one. ";

%feature("docstring")
gdcm::FileMetaInformation::SetDataSetTransferSyntax "void
gdcm::FileMetaInformation::SetDataSetTransferSyntax(const
TransferSyntax &ts) ";

%feature("docstring")  gdcm::FileMetaInformation::SetPreamble "void
gdcm::FileMetaInformation::SetPreamble(const Preamble &p) ";

%feature("docstring")  gdcm::FileMetaInformation::Write "std::ostream& gdcm::FileMetaInformation::Write(std::ostream &os) const

Write. ";


// File: classgdcm_1_1Filename.xml
%feature("docstring") gdcm::Filename "

Class to manipulate file name's.

OS independant representation of a filename (to query path, name and
extension from a filename)

C++ includes: gdcmFilename.h ";

%feature("docstring")  gdcm::Filename::Filename "gdcm::Filename::Filename(const char *filename=\"\") ";

%feature("docstring")  gdcm::Filename::EndWith "bool
gdcm::Filename::EndWith(const char ending[]) const

Does the filename ends with a particular string ? ";

%feature("docstring")  gdcm::Filename::GetExtension "const char*
gdcm::Filename::GetExtension()

return only the extension part of a filename ";

%feature("docstring")  gdcm::Filename::GetFileName "const char*
gdcm::Filename::GetFileName() const

Return the full filename. ";

%feature("docstring")  gdcm::Filename::GetName "const char*
gdcm::Filename::GetName()

return only the name part of a filename ";

%feature("docstring")  gdcm::Filename::GetPath "const char*
gdcm::Filename::GetPath()

Return only the path component of a filename. ";

%feature("docstring")  gdcm::Filename::IsEmpty "bool
gdcm::Filename::IsEmpty() const

return whether the filename is empty ";

%feature("docstring")  gdcm::Filename::IsIdentical "bool
gdcm::Filename::IsIdentical(Filename const &fn) const ";

%feature("docstring")  gdcm::Filename::ToUnixSlashes "const char*
gdcm::Filename::ToUnixSlashes()

Convert backslash (windows style) to UNIX style slash. ";

%feature("docstring")  gdcm::Filename::ToWindowsSlashes "const char*
gdcm::Filename::ToWindowsSlashes()

Convert foward slash (UNIX style) to windows style slash. ";


// File: classgdcm_1_1FileNameEvent.xml
%feature("docstring") gdcm::FileNameEvent "

FileNameEvent Special type of event triggered during processing of
FileSet.

See:   AnyEvent

C++ includes: gdcmFileNameEvent.h ";

%feature("docstring")  gdcm::FileNameEvent::FileNameEvent "gdcm::FileNameEvent::FileNameEvent(const char *s=\"\") ";

%feature("docstring")  gdcm::FileNameEvent::FileNameEvent "gdcm::FileNameEvent::FileNameEvent(const Self &s) ";

%feature("docstring")  gdcm::FileNameEvent::~FileNameEvent "virtual
gdcm::FileNameEvent::~FileNameEvent() ";

%feature("docstring")  gdcm::FileNameEvent::CheckEvent "virtual bool
gdcm::FileNameEvent::CheckEvent(const ::gdcm::Event *e) const ";

%feature("docstring")  gdcm::FileNameEvent::GetEventName "virtual
const char* gdcm::FileNameEvent::GetEventName() const

Return the StringName associated with the event. ";

%feature("docstring")  gdcm::FileNameEvent::GetFileName "const char*
gdcm::FileNameEvent::GetFileName() const ";

%feature("docstring")  gdcm::FileNameEvent::MakeObject "virtual
::gdcm::Event* gdcm::FileNameEvent::MakeObject() const

Create an Event of this type This method work as a Factory for
creating events of each particular type. ";

%feature("docstring")  gdcm::FileNameEvent::SetFileName "void
gdcm::FileNameEvent::SetFileName(const char *f) ";


// File: classgdcm_1_1FilenameGenerator.xml
%feature("docstring") gdcm::FilenameGenerator "

FilenameGenerator.

class to generate filenames based on a pattern (C-style)

Output will be:

for i = 0, number of filenames: outfilename[i] = prefix + (pattern %
i)

where pattern % i means C-style snprintf of Pattern using value 'i'

C++ includes: gdcmFilenameGenerator.h ";

%feature("docstring")  gdcm::FilenameGenerator::FilenameGenerator "gdcm::FilenameGenerator::FilenameGenerator() ";

%feature("docstring")  gdcm::FilenameGenerator::~FilenameGenerator "gdcm::FilenameGenerator::~FilenameGenerator() ";

%feature("docstring")  gdcm::FilenameGenerator::Generate "bool
gdcm::FilenameGenerator::Generate()

Generate (return success) ";

%feature("docstring")  gdcm::FilenameGenerator::GetFilename "const
char* gdcm::FilenameGenerator::GetFilename(SizeType n) const

Get a particular filename (call after Generate) ";

%feature("docstring")  gdcm::FilenameGenerator::GetFilenames "FilenamesType const& gdcm::FilenameGenerator::GetFilenames() const

Return all filenames. ";

%feature("docstring")  gdcm::FilenameGenerator::GetNumberOfFilenames "SizeType gdcm::FilenameGenerator::GetNumberOfFilenames() const ";

%feature("docstring")  gdcm::FilenameGenerator::GetPattern "const
char* gdcm::FilenameGenerator::GetPattern() const ";

%feature("docstring")  gdcm::FilenameGenerator::GetPrefix "const
char* gdcm::FilenameGenerator::GetPrefix() const ";

%feature("docstring")  gdcm::FilenameGenerator::SetNumberOfFilenames "void gdcm::FilenameGenerator::SetNumberOfFilenames(SizeType nfiles)

Set/Get the number of filenames to generate. ";

%feature("docstring")  gdcm::FilenameGenerator::SetPattern "void
gdcm::FilenameGenerator::SetPattern(const char *pattern)

Set/Get pattern. ";

%feature("docstring")  gdcm::FilenameGenerator::SetPrefix "void
gdcm::FilenameGenerator::SetPrefix(const char *prefix)

Set/Get prefix. ";


// File: classgdcm_1_1FileSet.xml
%feature("docstring") gdcm::FileSet "

File-set: A File-set is a collection of DICOM Files (and possibly non-
DICOM Files) that share a common naming space within which File IDs
are unique.

C++ includes: gdcmFileSet.h ";

%feature("docstring")  gdcm::FileSet::FileSet "gdcm::FileSet::FileSet() ";

%feature("docstring")  gdcm::FileSet::AddFile "void
gdcm::FileSet::AddFile(File const &)

Deprecated . Does nothing ";

%feature("docstring")  gdcm::FileSet::AddFile "bool
gdcm::FileSet::AddFile(const char *filename)

Add a file 'filename' to the list of files. Return true on success,
false in case filename could not be found on system. ";

%feature("docstring")  gdcm::FileSet::GetFiles "FilesType const&
gdcm::FileSet::GetFiles() const ";

%feature("docstring")  gdcm::FileSet::SetFiles "void
gdcm::FileSet::SetFiles(FilesType const &files) ";


// File: classgdcm_1_1FileStreamer.xml
%feature("docstring") gdcm::FileStreamer "

FileStreamer This class let a user create a massive DICOM DataSet from
a template DICOM file, by appending chunks of data.

This class support two mode of operation: Creating a single
DataElement by appending chunk after chunk of data.

Creating a set of DataElement within the same group, using a private
creator for start. New DataElement are added any time the user defined
maximum size for data element is reached.

WARNING:  any existing DataElement is removed, pick carefully which
DataElement to add.

C++ includes: gdcmFileStreamer.h ";

%feature("docstring")  gdcm::FileStreamer::FileStreamer "gdcm::FileStreamer::FileStreamer() ";

%feature("docstring")  gdcm::FileStreamer::~FileStreamer "gdcm::FileStreamer::~FileStreamer() ";

%feature("docstring")  gdcm::FileStreamer::AppendToDataElement "bool
gdcm::FileStreamer::AppendToDataElement(const Tag &t, const char
*array, size_t len)

Append to previously started Tag t. ";

%feature("docstring")  gdcm::FileStreamer::AppendToGroupDataElement "bool gdcm::FileStreamer::AppendToGroupDataElement(const PrivateTag
&pt, const char *array, size_t len)

Append to previously started private creator. ";

%feature("docstring")  gdcm::FileStreamer::CheckDataElement "bool
gdcm::FileStreamer::CheckDataElement(const Tag &t)

Decide to check the Data Element to be written (default: off) The
implementation has default strategy for checking validity of
DataElement. Currently it only support checking for the following
tags: (7fe0,0010) Pixel Data ";

%feature("docstring")  gdcm::FileStreamer::CheckTemplateFileName "void gdcm::FileStreamer::CheckTemplateFileName(bool check)

Instead of simply blindly copying the input DICOM Template file, GDCM
will be used to check the input file, and correct any issues
recognized within the file. Only use if you do not have control over
the input template file. ";

%feature("docstring")  gdcm::FileStreamer::ReserveDataElement "bool
gdcm::FileStreamer::ReserveDataElement(size_t len)

Add a hint on the final size of the dataelement. When optimally
chosen, this reduce the number of file in-place copying. Should be
called before StartDataElement ";

%feature("docstring")  gdcm::FileStreamer::ReserveGroupDataElement "bool gdcm::FileStreamer::ReserveGroupDataElement(unsigned short
ndataelement)

Optimisation: pre-allocate the number of dataelement within the
private group (ndataelement <= 256). Should be called before
StartGroupDataElement ";

%feature("docstring")  gdcm::FileStreamer::SetOutputFileName "void
gdcm::FileStreamer::SetOutputFileName(const char *filename_native)

Set output filename (target file) ";

%feature("docstring")  gdcm::FileStreamer::SetTemplateFileName "void
gdcm::FileStreamer::SetTemplateFileName(const char *filename_native)

Set input DICOM template filename. ";

%feature("docstring")  gdcm::FileStreamer::StartDataElement "bool
gdcm::FileStreamer::StartDataElement(const Tag &t)

Start Single Data Element Operation This will delete any existing Tag
t. Need to call it only once. ";

%feature("docstring")  gdcm::FileStreamer::StartGroupDataElement "bool gdcm::FileStreamer::StartGroupDataElement(const PrivateTag &pt,
size_t maxsizede=0, uint8_t startoffset=0)

Start Private Group (multiple DataElement) Operation. Each newly added
DataElement will have a length lower than

Parameters:
-----------

maxsizede:  . When not specified, maxsizede is set to maximum size
allowed by DICOM (= 2^32). startoffset can be used to specify the very
first element you want to start with (instead of the first possible).
Value should be in [0x0, 0xff] This will find the first available
private creator. ";

%feature("docstring")  gdcm::FileStreamer::StopDataElement "bool
gdcm::FileStreamer::StopDataElement(const Tag &t)

Stop appending to tag t. This will compute the proper attribute
length. ";

%feature("docstring")  gdcm::FileStreamer::StopGroupDataElement "bool
gdcm::FileStreamer::StopGroupDataElement(const PrivateTag &pt)

Stop appending to private creator. ";


// File: classgdcm_1_1FileWithName.xml
%feature("docstring") gdcm::FileWithName "

FileWithName.

Backward only class do not use in newer code

C++ includes: gdcmSerieHelper.h ";

%feature("docstring")  gdcm::FileWithName::FileWithName "gdcm::FileWithName::FileWithName(File &f) ";


// File: classgdcm_1_1FindPatientRootQuery.xml
%feature("docstring") gdcm::FindPatientRootQuery "

PatientRootQuery contains: the class which will produce a dataset for
c-find with patient root.

C++ includes: gdcmFindPatientRootQuery.h ";

%feature("docstring")
gdcm::FindPatientRootQuery::FindPatientRootQuery "gdcm::FindPatientRootQuery::FindPatientRootQuery() ";

%feature("docstring")
gdcm::FindPatientRootQuery::GetAbstractSyntaxUID "UIDs::TSName
gdcm::FindPatientRootQuery::GetAbstractSyntaxUID() const ";

%feature("docstring")  gdcm::FindPatientRootQuery::GetTagListByLevel "std::vector<Tag> gdcm::FindPatientRootQuery::GetTagListByLevel(const
EQueryLevel &inQueryLevel)

this function will return all tags at a given query level, so that
they maybe selected for searching. The boolean forFind is true if the
query is a find query, or false for a move query. ";

%feature("docstring")  gdcm::FindPatientRootQuery::InitializeDataSet "void gdcm::FindPatientRootQuery::InitializeDataSet(const EQueryLevel
&inQueryLevel)

this function sets tag 8,52 to the appropriate value based on query
level also fills in the right unique tags, as per the standard's
requirements should allow for connection with dcmtk ";

%feature("docstring")  gdcm::FindPatientRootQuery::ValidateQuery "bool gdcm::FindPatientRootQuery::ValidateQuery(bool inStrict=true)
const

have to be able to ensure that 0x8,0x52 is set (which will be true if
InitializeDataSet is called...) that the level is appropriate (ie, not
setting PATIENT for a study query that the tags in the query match the
right level (either required, unique, optional) by default, this
function checks to see if the query is for finding, which is more
permissive than for moving. For moving, only the unique tags are
allowed. 10 Jan 2011: adding in the 'strict' mode. according to the
standard (at least, how I've read it), only tags for a particular
level should be allowed in a particular query (ie, just series level
tags in a series level query). However, it seems that dcm4chee doesn't
share that interpretation. So, if 'inStrict' is false, then tags from
the current level and all higher levels are now considered valid. So,
if you're doing a non-strict series-level query, tags from the patient
and study level can be passed along as well. ";


// File: classgdcm_1_1FindStudyRootQuery.xml
%feature("docstring") gdcm::FindStudyRootQuery "

FindStudyRootQuery contains: the class which will produce a dataset
for C-FIND with study root.

C++ includes: gdcmFindStudyRootQuery.h ";

%feature("docstring")  gdcm::FindStudyRootQuery::FindStudyRootQuery "gdcm::FindStudyRootQuery::FindStudyRootQuery() ";

%feature("docstring")  gdcm::FindStudyRootQuery::GetAbstractSyntaxUID
"UIDs::TSName gdcm::FindStudyRootQuery::GetAbstractSyntaxUID() const
";

%feature("docstring")  gdcm::FindStudyRootQuery::GetTagListByLevel "std::vector<Tag> gdcm::FindStudyRootQuery::GetTagListByLevel(const
EQueryLevel &inQueryLevel)

this function will return all tags at a given query level, so that
they maybe selected for searching. The boolean forFind is true if the
query is a find query, or false for a move query. ";

%feature("docstring")  gdcm::FindStudyRootQuery::InitializeDataSet "void gdcm::FindStudyRootQuery::InitializeDataSet(const EQueryLevel
&inQueryLevel)

this function sets tag 8,52 to the appropriate value based on query
level also fills in the right unique tags, as per the standard's
requirements should allow for connection with dcmtk ";

%feature("docstring")  gdcm::FindStudyRootQuery::ValidateQuery "bool
gdcm::FindStudyRootQuery::ValidateQuery(bool inStrict=true) const

have to be able to ensure that (0008,0052) is set that the level is
appropriate (ie, not setting PATIENT for a study query that the tags
in the query match the right level (either required, unique, optional)
";


// File: classstd_1_1forward__list.xml
%feature("docstring") std::forward_list "

STL class. ";


// File: classgdcm_1_1Fragment.xml
%feature("docstring") gdcm::Fragment "

Class to represent a Fragment.

C++ includes: gdcmFragment.h ";

%feature("docstring")  gdcm::Fragment::Fragment "gdcm::Fragment::Fragment() ";

%feature("docstring")  gdcm::Fragment::ComputeLength "VL
gdcm::Fragment::ComputeLength() const ";

%feature("docstring")  gdcm::Fragment::GetLength "VL
gdcm::Fragment::GetLength() const ";

%feature("docstring")  gdcm::Fragment::Read "std::istream&
gdcm::Fragment::Read(std::istream &is) ";

%feature("docstring")  gdcm::Fragment::ReadBacktrack "std::istream&
gdcm::Fragment::ReadBacktrack(std::istream &is) ";

%feature("docstring")  gdcm::Fragment::ReadPreValue "std::istream&
gdcm::Fragment::ReadPreValue(std::istream &is) ";

%feature("docstring")  gdcm::Fragment::ReadValue "std::istream&
gdcm::Fragment::ReadValue(std::istream &is) ";

%feature("docstring")  gdcm::Fragment::Write "std::ostream&
gdcm::Fragment::Write(std::ostream &os) const ";


// File: classstd_1_1fstream.xml
%feature("docstring") std::fstream "

STL class. ";


// File: classgdcm_1_1Global.xml
%feature("docstring") gdcm::Global "

Global.

Global should be included in any translation unit that will use Dict
or that implements the singleton pattern. It makes sure that the Dict
singleton is created before and destroyed after all other singletons
in GDCM.

C++ includes: gdcmGlobal.h ";

%feature("docstring")  gdcm::Global::Global "gdcm::Global::Global()
";

%feature("docstring")  gdcm::Global::~Global "gdcm::Global::~Global()
";

%feature("docstring")  gdcm::Global::Append "bool
gdcm::Global::Append(const char *path)

Append path at the end of the path list WARNING:  not thread safe ! ";

%feature("docstring")  gdcm::Global::GetDefs "Defs const&
gdcm::Global::GetDefs() const

retrieve the default/internal (Part 3) You need to explicitely call
LoadResourcesFiles before ";

%feature("docstring")  gdcm::Global::GetDicts "Dicts const&
gdcm::Global::GetDicts() const

retrieve the default/internal dicts (Part 6) This dict is filled up at
load time ";

%feature("docstring")  gdcm::Global::GetDicts "Dicts&
gdcm::Global::GetDicts() ";

%feature("docstring")  gdcm::Global::LoadResourcesFiles "bool
gdcm::Global::LoadResourcesFiles()

Load all internal XML files, resource path need to have been set
before calling this member function (see Append/Prepend members func)
WARNING:  not thread safe ! ";

%feature("docstring")  gdcm::Global::Prepend "bool
gdcm::Global::Prepend(const char *path)

Prepend path at the beginning of the path list WARNING:  not thread
safe ! ";


// File: classgdcm_1_1GroupDict.xml
%feature("docstring") gdcm::GroupDict "

Class to represent the mapping from group number to its abbreviation
and name.

Should I rewrite this class to use a std::map instead of std::vector
for problem of memory consumption ?

C++ includes: gdcmGroupDict.h ";

%feature("docstring")  gdcm::GroupDict::GroupDict "gdcm::GroupDict::GroupDict() ";

%feature("docstring")  gdcm::GroupDict::~GroupDict "gdcm::GroupDict::~GroupDict() ";

%feature("docstring")  gdcm::GroupDict::GetAbbreviation "std::string
const& gdcm::GroupDict::GetAbbreviation(uint16_t num) const ";

%feature("docstring")  gdcm::GroupDict::GetName "std::string const&
gdcm::GroupDict::GetName(uint16_t num) const ";

%feature("docstring")  gdcm::GroupDict::Size "size_t
gdcm::GroupDict::Size() const ";


// File: classgdcm_1_1IconImageFilter.xml
%feature("docstring") gdcm::IconImageFilter "

IconImageFilter This filter will extract icons from a File This filter
will loop over all known sequence (public and private) that may
contains an IconImage and retrieve them. The filter will fails with a
value of false if no icon can be found Since it handle both public and
private icon type, one should not assume the icon is in uncompress
form, some private vendor store private icon in JPEG8/JPEG12.

Implementation details: This filter supports the following Icons:
(0088,0200) Icon Image Sequence

(0009,10,GEIIS) GE IIS Thumbnail Sequence

(6003,10,GEMS_Ultrasound_ImageGroup_001) GEMS Image Thumbnail Sequence

(0055,30,VEPRO VIF 3.0 DATA) Icon Data

(0055,30,VEPRO VIM 5.0 DATA) ICONDATA2

WARNING:  the icon stored in those private attribute do not conform to
definition of Icon Image Sequence (do not simply copy/paste). For
example some private icon can be expressed as 12bits pixel, while the
DICOM standard only allow 8bits icons.

See:   ImageReader

C++ includes: gdcmIconImageFilter.h ";

%feature("docstring")  gdcm::IconImageFilter::IconImageFilter "gdcm::IconImageFilter::IconImageFilter() ";

%feature("docstring")  gdcm::IconImageFilter::~IconImageFilter "gdcm::IconImageFilter::~IconImageFilter() ";

%feature("docstring")  gdcm::IconImageFilter::Extract "bool
gdcm::IconImageFilter::Extract()

Extract all Icon found in File. ";

%feature("docstring")  gdcm::IconImageFilter::GetFile "File&
gdcm::IconImageFilter::GetFile() ";

%feature("docstring")  gdcm::IconImageFilter::GetFile "const File&
gdcm::IconImageFilter::GetFile() const ";

%feature("docstring")  gdcm::IconImageFilter::GetIconImage "IconImage& gdcm::IconImageFilter::GetIconImage(unsigned int i) const
";

%feature("docstring")  gdcm::IconImageFilter::GetNumberOfIconImages "unsigned int gdcm::IconImageFilter::GetNumberOfIconImages() const

Retrieve extract IconImage (need to call Extract first) ";

%feature("docstring")  gdcm::IconImageFilter::SetFile "void
gdcm::IconImageFilter::SetFile(const File &f)

Set/Get File. ";


// File: classgdcm_1_1IconImageGenerator.xml
%feature("docstring") gdcm::IconImageGenerator "

IconImageGenerator This filter will generate a valid Icon from the
Pixel Data element (an instance of Pixmap). To generate a valid Icon,
one is only allowed the following Photometric Interpretation:

MONOCHROME1

MONOCHROME2

PALETTE_COLOR

The Pixel Bits Allocated is restricted to 8bits, therefore 16 bits
image needs to be rescaled. By default the filter will use the full
scalar range of 16bits image to rescale to unsigned 8bits. This may
not be ideal for some situation, in which case the API SetPixelMinMax
can be used to overwrite the default min,max interval used.

See:   ImageReader

C++ includes: gdcmIconImageGenerator.h ";

%feature("docstring")  gdcm::IconImageGenerator::IconImageGenerator "gdcm::IconImageGenerator::IconImageGenerator() ";

%feature("docstring")  gdcm::IconImageGenerator::~IconImageGenerator "gdcm::IconImageGenerator::~IconImageGenerator() ";

%feature("docstring")  gdcm::IconImageGenerator::AutoPixelMinMax "void gdcm::IconImageGenerator::AutoPixelMinMax(bool b)

Instead of explicitely specifying the min/max value for the rescale
operation, let the internal mechanism compute the min/max of icon and
rescale to best appropriate. ";

%feature("docstring")
gdcm::IconImageGenerator::ConvertRGBToPaletteColor "void
gdcm::IconImageGenerator::ConvertRGBToPaletteColor(bool b)

Converting from RGB to PALETTE_COLOR can be a slow operation. However
DICOM standard requires that color icon be described as palette. Set
this boolean to false only if you understand the consequences. default
value is true, false generates invalid Icon Image Sequence ";

%feature("docstring")  gdcm::IconImageGenerator::Generate "bool
gdcm::IconImageGenerator::Generate()

Generate Icon. ";

%feature("docstring")  gdcm::IconImageGenerator::GetIconImage "const
IconImage& gdcm::IconImageGenerator::GetIconImage() const

Retrieve generated Icon. ";

%feature("docstring")  gdcm::IconImageGenerator::GetPixmap "Pixmap&
gdcm::IconImageGenerator::GetPixmap() ";

%feature("docstring")  gdcm::IconImageGenerator::GetPixmap "const
Pixmap& gdcm::IconImageGenerator::GetPixmap() const ";

%feature("docstring")  gdcm::IconImageGenerator::SetOutputDimensions "void gdcm::IconImageGenerator::SetOutputDimensions(const unsigned int
dims[2])

Set Target dimension of output Icon. ";

%feature("docstring")  gdcm::IconImageGenerator::SetOutsideValuePixel
"void gdcm::IconImageGenerator::SetOutsideValuePixel(double v)

Set a pixel value that should be discarded. This happen typically for
CT image, where a pixel has been used to pad outside the image (see
Pixel Padding Value). Requires AutoPixelMinMax(true) ";

%feature("docstring")  gdcm::IconImageGenerator::SetPixelMinMax "void
gdcm::IconImageGenerator::SetPixelMinMax(double min, double max)

Override default min/max to compute best rescale for 16bits -> 8bits
downscale. Typically those value can be read from the
SmallestImagePixelValue LargestImagePixelValue DICOM attribute. ";

%feature("docstring")  gdcm::IconImageGenerator::SetPixmap "void
gdcm::IconImageGenerator::SetPixmap(const Pixmap &p)

Set/Get File. ";


// File: classstd_1_1ifstream.xml
%feature("docstring") std::ifstream "

STL class. ";


// File: structgdcm_1_1ignore__char.xml
%feature("docstring") gdcm::ignore_char "C++ includes: gdcmElement.h
";

%feature("docstring")  gdcm::ignore_char::ignore_char "gdcm::ignore_char::ignore_char(char c) ";


// File: classgdcm_1_1Image.xml
%feature("docstring") gdcm::Image "

Image This is the container for an Image in the general sense. From
this container you should be able to request information like:

Origin

Dimension

PixelFormat ... But also to retrieve the image as a raw buffer (char
*) Since we have to deal with both RAW data and JPEG stream (which
internally encode all the above information) this API might seems
redundant. One way to solve that would be to subclass Image with
JPEGImage which would from the stream extract the header info and fill
it to please Image...well except origin for instance

Basically you can see it as a storage for the Pixel Data element
(7fe0,0010).

WARNING:  This class does some heuristics to guess the Spacing but is
not compatible with DICOM CP-586. In case of doubt use PixmapReader
instead

See:   ImageReader PixmapReader

C++ includes: gdcmImage.h ";

%feature("docstring")  gdcm::Image::Image "gdcm::Image::Image() ";

%feature("docstring")  gdcm::Image::~Image "gdcm::Image::~Image() ";

%feature("docstring")  gdcm::Image::GetDirectionCosines "const
double* gdcm::Image::GetDirectionCosines() const

Return a 6-tuples specifying the direction cosines A default value of
(1,0,0,0,1,0) will be return when the direction cosines was not
specified. ";

%feature("docstring")  gdcm::Image::GetDirectionCosines "double
gdcm::Image::GetDirectionCosines(unsigned int idx) const ";

%feature("docstring")  gdcm::Image::GetIntercept "double
gdcm::Image::GetIntercept() const ";

%feature("docstring")  gdcm::Image::GetOrigin "const double*
gdcm::Image::GetOrigin() const

Return a 3-tuples specifying the origin Will return (0,0,0) if the
origin was not specified. ";

%feature("docstring")  gdcm::Image::GetOrigin "double
gdcm::Image::GetOrigin(unsigned int idx) const ";

%feature("docstring")  gdcm::Image::GetSlope "double
gdcm::Image::GetSlope() const ";

%feature("docstring")  gdcm::Image::GetSpacing "const double*
gdcm::Image::GetSpacing() const

Return a 3-tuples specifying the spacing NOTE: 3rd value can be an
aribtrary 1 value when the spacing was not specified (ex. 2D image).
WARNING: when the spacing is not specifier, a default value of 1 will
be returned ";

%feature("docstring")  gdcm::Image::GetSpacing "double
gdcm::Image::GetSpacing(unsigned int idx) const ";

%feature("docstring")  gdcm::Image::Print "void
gdcm::Image::Print(std::ostream &os) const

print ";

%feature("docstring")  gdcm::Image::SetDirectionCosines "void
gdcm::Image::SetDirectionCosines(const float *dircos) ";

%feature("docstring")  gdcm::Image::SetDirectionCosines "void
gdcm::Image::SetDirectionCosines(const double *dircos) ";

%feature("docstring")  gdcm::Image::SetDirectionCosines "void
gdcm::Image::SetDirectionCosines(unsigned int idx, double dircos) ";

%feature("docstring")  gdcm::Image::SetIntercept "void
gdcm::Image::SetIntercept(double intercept)

intercept ";

%feature("docstring")  gdcm::Image::SetOrigin "void
gdcm::Image::SetOrigin(const float *ori) ";

%feature("docstring")  gdcm::Image::SetOrigin "void
gdcm::Image::SetOrigin(const double *ori) ";

%feature("docstring")  gdcm::Image::SetOrigin "void
gdcm::Image::SetOrigin(unsigned int idx, double ori) ";

%feature("docstring")  gdcm::Image::SetSlope "void
gdcm::Image::SetSlope(double slope)

slope ";

%feature("docstring")  gdcm::Image::SetSpacing "void
gdcm::Image::SetSpacing(const double *spacing) ";

%feature("docstring")  gdcm::Image::SetSpacing "void
gdcm::Image::SetSpacing(unsigned int idx, double spacing) ";


// File: classgdcm_1_1ImageApplyLookupTable.xml
%feature("docstring") gdcm::ImageApplyLookupTable "

ImageApplyLookupTable class It applies the LUT the PixelData (only
PALETTE_COLOR images) Output will be a PhotometricInterpretation=RGB
image.

C++ includes: gdcmImageApplyLookupTable.h ";

%feature("docstring")
gdcm::ImageApplyLookupTable::ImageApplyLookupTable "gdcm::ImageApplyLookupTable::ImageApplyLookupTable() ";

%feature("docstring")
gdcm::ImageApplyLookupTable::~ImageApplyLookupTable "gdcm::ImageApplyLookupTable::~ImageApplyLookupTable() ";

%feature("docstring")  gdcm::ImageApplyLookupTable::Apply "bool
gdcm::ImageApplyLookupTable::Apply()

Apply. ";


// File: classgdcm_1_1ImageChangePhotometricInterpretation.xml
%feature("docstring") gdcm::ImageChangePhotometricInterpretation "

ImageChangePhotometricInterpretation class Class to change the
Photometric Interpetation of an input DICOM.

C++ includes: gdcmImageChangePhotometricInterpretation.h ";

%feature("docstring")
gdcm::ImageChangePhotometricInterpretation::ImageChangePhotometricInterpretation
"gdcm::ImageChangePhotometricInterpretation::ImageChangePhotometricInterpretation()
";

%feature("docstring")
gdcm::ImageChangePhotometricInterpretation::~ImageChangePhotometricInterpretation
"gdcm::ImageChangePhotometricInterpretation::~ImageChangePhotometricInterpretation()
";

%feature("docstring")
gdcm::ImageChangePhotometricInterpretation::Change "bool
gdcm::ImageChangePhotometricInterpretation::Change()

Change. ";

%feature("docstring")
gdcm::ImageChangePhotometricInterpretation::GetPhotometricInterpretation
"const PhotometricInterpretation&
gdcm::ImageChangePhotometricInterpretation::GetPhotometricInterpretation()
const ";

%feature("docstring")
gdcm::ImageChangePhotometricInterpretation::SetPhotometricInterpretation
"void
gdcm::ImageChangePhotometricInterpretation::SetPhotometricInterpretation(PhotometricInterpretation
const &pi)

Set/Get requested PhotometricInterpretation. ";


// File: classgdcm_1_1ImageChangePlanarConfiguration.xml
%feature("docstring") gdcm::ImageChangePlanarConfiguration "

ImageChangePlanarConfiguration class Class to change the Planar
configuration of an input DICOM By default it will change into the
more usual reprensentation: PlanarConfiguration = 0.

C++ includes: gdcmImageChangePlanarConfiguration.h ";

%feature("docstring")
gdcm::ImageChangePlanarConfiguration::ImageChangePlanarConfiguration "gdcm::ImageChangePlanarConfiguration::ImageChangePlanarConfiguration()
";

%feature("docstring")
gdcm::ImageChangePlanarConfiguration::~ImageChangePlanarConfiguration
"gdcm::ImageChangePlanarConfiguration::~ImageChangePlanarConfiguration()
";

%feature("docstring")  gdcm::ImageChangePlanarConfiguration::Change "bool gdcm::ImageChangePlanarConfiguration::Change()

Change. ";

%feature("docstring")
gdcm::ImageChangePlanarConfiguration::GetPlanarConfiguration "unsigned int
gdcm::ImageChangePlanarConfiguration::GetPlanarConfiguration() const
";

%feature("docstring")
gdcm::ImageChangePlanarConfiguration::SetPlanarConfiguration "void
gdcm::ImageChangePlanarConfiguration::SetPlanarConfiguration(unsigned
int pc)

Set/Get requested PlanarConfigation. ";


// File: classgdcm_1_1ImageChangeTransferSyntax.xml
%feature("docstring") gdcm::ImageChangeTransferSyntax "

ImageChangeTransferSyntax class Class to change the transfer syntax of
an input DICOM.

If only Force param is set but no input TransferSyntax is set, it is
assumed that user only wants to inspect encapsulated stream (advanced
dev. option).

When using UserCodec it is very important that the TransferSyntax (as
set in SetTransferSyntax) is actually understood by UserCodec (ie.
UserCodec->CanCode( TransferSyntax ) ). Otherwise the behavior is to
use a default codec.

See:   JPEGCodec JPEGLSCodec JPEG2000Codec

C++ includes: gdcmImageChangeTransferSyntax.h ";

%feature("docstring")
gdcm::ImageChangeTransferSyntax::ImageChangeTransferSyntax "gdcm::ImageChangeTransferSyntax::ImageChangeTransferSyntax() ";

%feature("docstring")
gdcm::ImageChangeTransferSyntax::~ImageChangeTransferSyntax "gdcm::ImageChangeTransferSyntax::~ImageChangeTransferSyntax() ";

%feature("docstring")  gdcm::ImageChangeTransferSyntax::Change "bool
gdcm::ImageChangeTransferSyntax::Change()

Change. ";

%feature("docstring")
gdcm::ImageChangeTransferSyntax::GetTransferSyntax "const
TransferSyntax& gdcm::ImageChangeTransferSyntax::GetTransferSyntax()
const

Get Transfer Syntax. ";

%feature("docstring")
gdcm::ImageChangeTransferSyntax::SetCompressIconImage "void
gdcm::ImageChangeTransferSyntax::SetCompressIconImage(bool b)

Decide whether or not to also compress the Icon Image using the same
Transfer Syntax. Default is to simply decompress icon image ";

%feature("docstring")  gdcm::ImageChangeTransferSyntax::SetForce "void gdcm::ImageChangeTransferSyntax::SetForce(bool f)

When target Transfer Syntax is identical to input target syntax, no
operation is actually done. This is an issue when someone wants to re-
compress using GDCM internal implementation a JPEG (for example) image
";

%feature("docstring")
gdcm::ImageChangeTransferSyntax::SetTransferSyntax "void
gdcm::ImageChangeTransferSyntax::SetTransferSyntax(const
TransferSyntax &ts)

Set target Transfer Syntax. ";

%feature("docstring")  gdcm::ImageChangeTransferSyntax::SetUserCodec "void gdcm::ImageChangeTransferSyntax::SetUserCodec(ImageCodec *ic)

Allow user to specify exactly which codec to use. this is needed to
specify special qualities or compression option. WARNING:  if the
codec 'ic' is not compatible with the TransferSyntax requested, it
will not be used. It is the user responsibility to check that
UserCodec->CanCode( TransferSyntax ) ";


// File: classgdcm_1_1ImageCodec.xml
%feature("docstring") gdcm::ImageCodec "

ImageCodec.

Main codec, this is a central place for all implementation

C++ includes: gdcmImageCodec.h ";

%feature("docstring")  gdcm::ImageCodec::ImageCodec "gdcm::ImageCodec::ImageCodec() ";

%feature("docstring")  gdcm::ImageCodec::~ImageCodec "gdcm::ImageCodec::~ImageCodec() ";

%feature("docstring")  gdcm::ImageCodec::CanCode "bool
gdcm::ImageCodec::CanCode(TransferSyntax const &) const

Return whether this coder support this transfer syntax (can code it)
";

%feature("docstring")  gdcm::ImageCodec::CanDecode "bool
gdcm::ImageCodec::CanDecode(TransferSyntax const &) const

Return whether this decoder support this transfer syntax (can decode
it) ";

%feature("docstring")  gdcm::ImageCodec::Clone "virtual ImageCodec*
gdcm::ImageCodec::Clone() const =0 ";

%feature("docstring")  gdcm::ImageCodec::Decode "bool
gdcm::ImageCodec::Decode(DataElement const &is_, DataElement &os)

Decode. ";

%feature("docstring")  gdcm::ImageCodec::GetDimensions "const
unsigned int* gdcm::ImageCodec::GetDimensions() const ";

%feature("docstring")  gdcm::ImageCodec::GetHeaderInfo "virtual bool
gdcm::ImageCodec::GetHeaderInfo(std::istream &is_, TransferSyntax &ts)
";

%feature("docstring")  gdcm::ImageCodec::GetLossyFlag "bool
gdcm::ImageCodec::GetLossyFlag() const ";

%feature("docstring")  gdcm::ImageCodec::GetLUT "const LookupTable&
gdcm::ImageCodec::GetLUT() const ";

%feature("docstring")  gdcm::ImageCodec::GetNeedByteSwap "bool
gdcm::ImageCodec::GetNeedByteSwap() const ";

%feature("docstring")  gdcm::ImageCodec::GetNumberOfDimensions "unsigned int gdcm::ImageCodec::GetNumberOfDimensions() const ";

%feature("docstring")  gdcm::ImageCodec::GetPhotometricInterpretation
"const PhotometricInterpretation&
gdcm::ImageCodec::GetPhotometricInterpretation() const ";

%feature("docstring")  gdcm::ImageCodec::GetPixelFormat "PixelFormat&
gdcm::ImageCodec::GetPixelFormat() ";

%feature("docstring")  gdcm::ImageCodec::GetPixelFormat "const
PixelFormat& gdcm::ImageCodec::GetPixelFormat() const ";

%feature("docstring")  gdcm::ImageCodec::GetPlanarConfiguration "unsigned int gdcm::ImageCodec::GetPlanarConfiguration() const ";

%feature("docstring")  gdcm::ImageCodec::IsLossy "bool
gdcm::ImageCodec::IsLossy() const ";

%feature("docstring")  gdcm::ImageCodec::SetDimensions "void
gdcm::ImageCodec::SetDimensions(const unsigned int d[3]) ";

%feature("docstring")  gdcm::ImageCodec::SetDimensions "void
gdcm::ImageCodec::SetDimensions(const std::vector< unsigned int > &d)
";

%feature("docstring")  gdcm::ImageCodec::SetLossyFlag "void
gdcm::ImageCodec::SetLossyFlag(bool l) ";

%feature("docstring")  gdcm::ImageCodec::SetLUT "void
gdcm::ImageCodec::SetLUT(LookupTable const &lut) ";

%feature("docstring")  gdcm::ImageCodec::SetNeedByteSwap "void
gdcm::ImageCodec::SetNeedByteSwap(bool b) ";

%feature("docstring")  gdcm::ImageCodec::SetNeedOverlayCleanup "void
gdcm::ImageCodec::SetNeedOverlayCleanup(bool b) ";

%feature("docstring")  gdcm::ImageCodec::SetNumberOfDimensions "void
gdcm::ImageCodec::SetNumberOfDimensions(unsigned int dim) ";

%feature("docstring")  gdcm::ImageCodec::SetPhotometricInterpretation
"void
gdcm::ImageCodec::SetPhotometricInterpretation(PhotometricInterpretation
const &pi) ";

%feature("docstring")  gdcm::ImageCodec::SetPixelFormat "virtual void
gdcm::ImageCodec::SetPixelFormat(PixelFormat const &pf) ";

%feature("docstring")  gdcm::ImageCodec::SetPlanarConfiguration "void
gdcm::ImageCodec::SetPlanarConfiguration(unsigned int pc) ";


// File: classgdcm_1_1ImageConverter.xml
%feature("docstring") gdcm::ImageConverter "

Image Converter.

This is the class used to convert from on Image to another This is
typically used to convert let say YBR JPEG compressed Image to a RAW
RGB Image. So that the buffer can be directly pass to third party
application. This filter is application level and not integrated
directly in GDCM

C++ includes: gdcmImageConverter.h ";

%feature("docstring")  gdcm::ImageConverter::ImageConverter "gdcm::ImageConverter::ImageConverter() ";

%feature("docstring")  gdcm::ImageConverter::~ImageConverter "gdcm::ImageConverter::~ImageConverter() ";

%feature("docstring")  gdcm::ImageConverter::Convert "void
gdcm::ImageConverter::Convert() ";

%feature("docstring")  gdcm::ImageConverter::GetOuput "const Image&
gdcm::ImageConverter::GetOuput() const ";

%feature("docstring")  gdcm::ImageConverter::SetInput "void
gdcm::ImageConverter::SetInput(Image const &input) ";


// File: classgdcm_1_1ImageFragmentSplitter.xml
%feature("docstring") gdcm::ImageFragmentSplitter "

ImageFragmentSplitter class For single frame image, DICOM standard
allow splitting the frame into multiple fragments.

C++ includes: gdcmImageFragmentSplitter.h ";

%feature("docstring")
gdcm::ImageFragmentSplitter::ImageFragmentSplitter "gdcm::ImageFragmentSplitter::ImageFragmentSplitter() ";

%feature("docstring")
gdcm::ImageFragmentSplitter::~ImageFragmentSplitter "gdcm::ImageFragmentSplitter::~ImageFragmentSplitter() ";

%feature("docstring")  gdcm::ImageFragmentSplitter::GetFragmentSizeMax
"unsigned int gdcm::ImageFragmentSplitter::GetFragmentSizeMax() const
";

%feature("docstring")  gdcm::ImageFragmentSplitter::SetForce "void
gdcm::ImageFragmentSplitter::SetForce(bool f)

When file already has all it's segment < FragmentSizeMax there is not
need to run the filter. Unless the user explicitly say 'force'
recomputation ! ";

%feature("docstring")  gdcm::ImageFragmentSplitter::SetFragmentSizeMax
"void gdcm::ImageFragmentSplitter::SetFragmentSizeMax(unsigned int
fragsize)

FragmentSizeMax needs to be an even number. ";

%feature("docstring")  gdcm::ImageFragmentSplitter::Split "bool
gdcm::ImageFragmentSplitter::Split()

Split. ";


// File: classgdcm_1_1ImageHelper.xml
%feature("docstring") gdcm::ImageHelper "

ImageHelper (internal class, not intended for user level)

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

C++ includes: gdcmImageHelper.h ";


// File: classgdcm_1_1ImageReader.xml
%feature("docstring") gdcm::ImageReader "

ImageReader.

its role is to convert the DICOM DataSet into a Image representation
Image is different from Pixmap has it has a position and a direction
in Space.

See:   Image

C++ includes: gdcmImageReader.h ";

%feature("docstring")  gdcm::ImageReader::ImageReader "gdcm::ImageReader::ImageReader() ";

%feature("docstring")  gdcm::ImageReader::~ImageReader "virtual
gdcm::ImageReader::~ImageReader() ";

%feature("docstring")  gdcm::ImageReader::GetImage "const Image&
gdcm::ImageReader::GetImage() const

Return the read image. ";

%feature("docstring")  gdcm::ImageReader::GetImage "Image&
gdcm::ImageReader::GetImage() ";

%feature("docstring")  gdcm::ImageReader::Read "virtual bool
gdcm::ImageReader::Read()

Read the DICOM image. There are two reason for failure: The input
filename is not DICOM

The input DICOM file does not contains an Image. ";


// File: classgdcm_1_1ImageRegionReader.xml
%feature("docstring") gdcm::ImageRegionReader "

ImageRegionReader.

See:   ImageReader

C++ includes: gdcmImageRegionReader.h ";

%feature("docstring")  gdcm::ImageRegionReader::ImageRegionReader "gdcm::ImageRegionReader::ImageRegionReader() ";

%feature("docstring")  gdcm::ImageRegionReader::~ImageRegionReader "gdcm::ImageRegionReader::~ImageRegionReader() ";

%feature("docstring")  gdcm::ImageRegionReader::ComputeBufferLength "size_t gdcm::ImageRegionReader::ComputeBufferLength() const

Explicit call which will compute the minimal buffer length that can
hold the whole uncompressed image as defined by Region region. 0 upon
error ";

%feature("docstring")  gdcm::ImageRegionReader::GetRegion "Region
const& gdcm::ImageRegionReader::GetRegion() const ";

%feature("docstring")  gdcm::ImageRegionReader::ReadInformation "bool
gdcm::ImageRegionReader::ReadInformation()

Read meta information (not Pixel Data) from the DICOM file. false upon
error ";

%feature("docstring")  gdcm::ImageRegionReader::ReadIntoBuffer "bool
gdcm::ImageRegionReader::ReadIntoBuffer(char *inreadbuffer, size_t
buflen)

Read into buffer: false upon error ";

%feature("docstring")  gdcm::ImageRegionReader::SetRegion "void
gdcm::ImageRegionReader::SetRegion(Region const &region)

Set/Get Region to be read. ";


// File: classgdcm_1_1ImageToImageFilter.xml
%feature("docstring") gdcm::ImageToImageFilter "

ImageToImageFilter class Super class for all filter taking an image
and producing an output image.

C++ includes: gdcmImageToImageFilter.h ";

%feature("docstring")  gdcm::ImageToImageFilter::ImageToImageFilter "gdcm::ImageToImageFilter::ImageToImageFilter() ";

%feature("docstring")  gdcm::ImageToImageFilter::~ImageToImageFilter "gdcm::ImageToImageFilter::~ImageToImageFilter() ";

%feature("docstring")  gdcm::ImageToImageFilter::GetInput "Image&
gdcm::ImageToImageFilter::GetInput() ";

%feature("docstring")  gdcm::ImageToImageFilter::GetOutput "const
Image& gdcm::ImageToImageFilter::GetOutput() const

Get Output image. ";


// File: classgdcm_1_1ImageWriter.xml
%feature("docstring") gdcm::ImageWriter "

ImageWriter.

C++ includes: gdcmImageWriter.h ";

%feature("docstring")  gdcm::ImageWriter::ImageWriter "gdcm::ImageWriter::ImageWriter() ";

%feature("docstring")  gdcm::ImageWriter::~ImageWriter "gdcm::ImageWriter::~ImageWriter() ";

%feature("docstring")  gdcm::ImageWriter::GetImage "const Image&
gdcm::ImageWriter::GetImage() const

Set/Get Image to be written It will overwrite anything Image infos
found in DataSet (see parent class to see how to pass dataset) ";

%feature("docstring")  gdcm::ImageWriter::GetImage "Image&
gdcm::ImageWriter::GetImage() ";

%feature("docstring")  gdcm::ImageWriter::Write "bool
gdcm::ImageWriter::Write()

Write. ";


// File: classgdcm_1_1network_1_1ImplementationClassUIDSub.xml
%feature("docstring") gdcm::network::ImplementationClassUIDSub "

ImplementationClassUIDSub PS 3.7 Table D.3-1 IMPLEMENTATION CLASS UID
SUB-ITEM FIELDS (A-ASSOCIATE-RQ)

C++ includes: gdcmImplementationClassUIDSub.h ";

%feature("docstring")
gdcm::network::ImplementationClassUIDSub::ImplementationClassUIDSub "gdcm::network::ImplementationClassUIDSub::ImplementationClassUIDSub()
";

%feature("docstring")  gdcm::network::ImplementationClassUIDSub::Print
"void gdcm::network::ImplementationClassUIDSub::Print(std::ostream
&os) const ";

%feature("docstring")  gdcm::network::ImplementationClassUIDSub::Read
"std::istream&
gdcm::network::ImplementationClassUIDSub::Read(std::istream &is) ";

%feature("docstring")  gdcm::network::ImplementationClassUIDSub::Size
"size_t gdcm::network::ImplementationClassUIDSub::Size() const ";

%feature("docstring")  gdcm::network::ImplementationClassUIDSub::Write
"const std::ostream&
gdcm::network::ImplementationClassUIDSub::Write(std::ostream &os)
const ";


// File: classgdcm_1_1network_1_1ImplementationUIDSub.xml
%feature("docstring") gdcm::network::ImplementationUIDSub "

ImplementationUIDSub Table D.3-2 IMPLEMENTATION UID SUB-ITEM FIELDS (A
-ASSOCIATE-AC)

C++ includes: gdcmImplementationUIDSub.h ";

%feature("docstring")
gdcm::network::ImplementationUIDSub::ImplementationUIDSub "gdcm::network::ImplementationUIDSub::ImplementationUIDSub() ";

%feature("docstring")  gdcm::network::ImplementationUIDSub::Write "const std::ostream&
gdcm::network::ImplementationUIDSub::Write(std::ostream &os) const ";


// File: classgdcm_1_1network_1_1ImplementationVersionNameSub.xml
%feature("docstring") gdcm::network::ImplementationVersionNameSub "

ImplementationVersionNameSub Table D.3-3 IMPLEMENTATION VERSION NAME
SUB-ITEM FIELDS (A-ASSOCIATE-RQ)

C++ includes: gdcmImplementationVersionNameSub.h ";

%feature("docstring")
gdcm::network::ImplementationVersionNameSub::ImplementationVersionNameSub
"gdcm::network::ImplementationVersionNameSub::ImplementationVersionNameSub()
";

%feature("docstring")
gdcm::network::ImplementationVersionNameSub::Print "void
gdcm::network::ImplementationVersionNameSub::Print(std::ostream &os)
const ";

%feature("docstring")
gdcm::network::ImplementationVersionNameSub::Read "std::istream&
gdcm::network::ImplementationVersionNameSub::Read(std::istream &is) ";

%feature("docstring")
gdcm::network::ImplementationVersionNameSub::Size "size_t
gdcm::network::ImplementationVersionNameSub::Size() const ";

%feature("docstring")
gdcm::network::ImplementationVersionNameSub::Write "const
std::ostream&
gdcm::network::ImplementationVersionNameSub::Write(std::ostream &os)
const ";


// File: classgdcm_1_1ImplicitDataElement.xml
%feature("docstring") gdcm::ImplicitDataElement "

Class to represent an Implicit VR Data Element.

bla

C++ includes: gdcmImplicitDataElement.h ";

%feature("docstring")  gdcm::ImplicitDataElement::GetLength "VL
gdcm::ImplicitDataElement::GetLength() const ";

%feature("docstring")  gdcm::ImplicitDataElement::Read "std::istream&
gdcm::ImplicitDataElement::Read(std::istream &is) ";

%feature("docstring")  gdcm::ImplicitDataElement::ReadPreValue "std::istream& gdcm::ImplicitDataElement::ReadPreValue(std::istream
&is) ";

%feature("docstring")  gdcm::ImplicitDataElement::ReadValue "std::istream& gdcm::ImplicitDataElement::ReadValue(std::istream &is,
bool readvalues=true) ";

%feature("docstring")  gdcm::ImplicitDataElement::ReadValueWithLength
"std::istream&
gdcm::ImplicitDataElement::ReadValueWithLength(std::istream &is, VL
&length, bool readvalues=true) ";

%feature("docstring")  gdcm::ImplicitDataElement::ReadWithLength "std::istream& gdcm::ImplicitDataElement::ReadWithLength(std::istream
&is, VL &length, bool readvalues=true) ";

%feature("docstring")  gdcm::ImplicitDataElement::Write "const
std::ostream& gdcm::ImplicitDataElement::Write(std::ostream &os) const
";


// File: classgdcm_1_1InitializeEvent.xml
%feature("docstring") gdcm::InitializeEvent "C++ includes:
gdcmEvent.h ";


// File: classstd_1_1invalid__argument.xml
%feature("docstring") std::invalid_argument "

STL class. ";


// File: classgdcm_1_1IOD.xml
%feature("docstring") gdcm::IOD "

Class for representing a IOD.

bla

See:   Dict

C++ includes: gdcmIOD.h ";

%feature("docstring")  gdcm::IOD::IOD "gdcm::IOD::IOD() ";

%feature("docstring")  gdcm::IOD::AddIODEntry "void
gdcm::IOD::AddIODEntry(const IODEntry &iode) ";

%feature("docstring")  gdcm::IOD::Clear "void gdcm::IOD::Clear() ";

%feature("docstring")  gdcm::IOD::GetIODEntry "const IODEntry&
gdcm::IOD::GetIODEntry(SizeType idx) const ";

%feature("docstring")  gdcm::IOD::GetNumberOfIODs "SizeType
gdcm::IOD::GetNumberOfIODs() const ";

%feature("docstring")  gdcm::IOD::GetTypeFromTag "Type
gdcm::IOD::GetTypeFromTag(const Defs &defs, const Tag &tag) const ";


// File: classgdcm_1_1IODEntry.xml
%feature("docstring") gdcm::IODEntry "

Class for representing a IODEntry.

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

C++ includes: gdcmIODEntry.h ";

%feature("docstring")  gdcm::IODEntry::IODEntry "gdcm::IODEntry::IODEntry(const char *name=\"\", const char *ref=\"\",
const char *usag=\"\") ";

%feature("docstring")  gdcm::IODEntry::GetIE "const char*
gdcm::IODEntry::GetIE() const ";

%feature("docstring")  gdcm::IODEntry::GetName "const char*
gdcm::IODEntry::GetName() const ";

%feature("docstring")  gdcm::IODEntry::GetRef "const char*
gdcm::IODEntry::GetRef() const ";

%feature("docstring")  gdcm::IODEntry::GetUsage "const char*
gdcm::IODEntry::GetUsage() const ";

%feature("docstring")  gdcm::IODEntry::GetUsageType "Usage::UsageType
gdcm::IODEntry::GetUsageType() const ";

%feature("docstring")  gdcm::IODEntry::SetIE "void
gdcm::IODEntry::SetIE(const char *ie) ";

%feature("docstring")  gdcm::IODEntry::SetName "void
gdcm::IODEntry::SetName(const char *name) ";

%feature("docstring")  gdcm::IODEntry::SetRef "void
gdcm::IODEntry::SetRef(const char *ref) ";

%feature("docstring")  gdcm::IODEntry::SetUsage "void
gdcm::IODEntry::SetUsage(const char *usag) ";


// File: classgdcm_1_1IODs.xml
%feature("docstring") gdcm::IODs "

Class for representing a IODs.

bla

See:   IOD

C++ includes: gdcmIODs.h ";

%feature("docstring")  gdcm::IODs::IODs "gdcm::IODs::IODs() ";

%feature("docstring")  gdcm::IODs::AddIOD "void
gdcm::IODs::AddIOD(const char *name, const IOD &module) ";

%feature("docstring")  gdcm::IODs::Begin "IODMapTypeConstIterator
gdcm::IODs::Begin() const ";

%feature("docstring")  gdcm::IODs::Clear "void gdcm::IODs::Clear() ";

%feature("docstring")  gdcm::IODs::End "IODMapTypeConstIterator
gdcm::IODs::End() const ";

%feature("docstring")  gdcm::IODs::GetIOD "const IOD&
gdcm::IODs::GetIOD(const char *name) const ";


// File: classstd_1_1ios.xml
%feature("docstring") std::ios "

STL class. ";


// File: classstd_1_1ios__base.xml
%feature("docstring") std::ios_base "

STL class. ";


// File: classgdcm_1_1IPPSorter.xml
%feature("docstring") gdcm::IPPSorter "

IPPSorter Implement a simple Image Position ( Patient) sorter, along
the Image Orientation ( Patient) direction. This algorithm does NOT
support duplicate and will FAIL in case of duplicate IPP.

WARNING:  See special note for SetZSpacingTolerance when computing the
ZSpacing from the IPP of each DICOM files (default tolerance for
consistent spacing is: 1e-6mm)  For more information on Spacing, and
how it is defined in DICOM, advanced users may refers to:

http://gdcm.sourceforge.net/wiki/index.php/Imager_Pixel_Spacing

Bug There are currently a couple of bugs in this implementation:

Gantry Tilt is not considered

C++ includes: gdcmIPPSorter.h ";

%feature("docstring")  gdcm::IPPSorter::IPPSorter "gdcm::IPPSorter::IPPSorter() ";

%feature("docstring")  gdcm::IPPSorter::GetDirectionCosinesTolerance "double gdcm::IPPSorter::GetDirectionCosinesTolerance() const ";

%feature("docstring")  gdcm::IPPSorter::GetZSpacing "double
gdcm::IPPSorter::GetZSpacing() const

Read-only function to provide access to the computed value for the
Z-Spacing The ComputeZSpacing must have been set to true before
execution of sort algorithm. Call this function after calling Sort();
Z-Spacing will be 0 on 2 occasions: Sorting simply failed, potentially
duplicate IPP => ZSpacing = 0

ZSpacing could not be computed (Z-Spacing is not constant, or
ZTolerance is too low) ";

%feature("docstring")  gdcm::IPPSorter::GetZSpacingTolerance "double
gdcm::IPPSorter::GetZSpacingTolerance() const ";

%feature("docstring")  gdcm::IPPSorter::SetComputeZSpacing "void
gdcm::IPPSorter::SetComputeZSpacing(bool b)

Functions related to Z-Spacing computation Set to true when sort
algorithm should also perform a regular Z-Spacing computation using
the Image Position ( Patient) Potential reason for failure: ALL slices
are taken into account, if one slice if missing then ZSpacing will be
set to 0 since the spacing will not be found to be regular along the
Series ";

%feature("docstring")  gdcm::IPPSorter::SetDirectionCosinesTolerance "void gdcm::IPPSorter::SetDirectionCosinesTolerance(double tol)

Sometimes IOP along a series is slightly changing for example:
\"0.999081\\\\\\\\0.0426953\\\\\\\\0.00369272\\\\-0.0419025\\\\\\\\0.955059\\\\\\\\0.293439\",
\"0.999081\\\\\\\\0.0426953\\\\\\\\0.00369275\\\\-0.0419025\\\\\\\\0.955059\\\\\\\\0.293439\",
\"0.999081\\\\\\\\0.0426952\\\\\\\\0.00369272\\\\-0.0419025\\\\\\\\0.955059\\\\\\\\0.293439\",
We need an API to define the tolerance which is allowed. Internally
the cross vector of each direction cosines is computed. The tolerance
then define the the distance in between 1. to the dot product of those
cross vectors. In a perfect world this dot product is of course 1.0
which imply a DirectionCosines tolerance of exactly 0.0 (default). ";

%feature("docstring")  gdcm::IPPSorter::SetDropDuplicatePositions "void gdcm::IPPSorter::SetDropDuplicatePositions(bool b)

Makes the IPPSorter ignore multiple images located at the same
position. Only the first occurence will be kept.
DropDuplicatePositions defaults to false. ";

%feature("docstring")  gdcm::IPPSorter::SetZSpacingTolerance "void
gdcm::IPPSorter::SetZSpacingTolerance(double tol)

Another reason for failure is that that Z-Spacing is only slightly
changing (eg 1e-3) along the serie, a human can determine that this is
ok and change the tolerance from its default value: 1e-6 ";

%feature("docstring")  gdcm::IPPSorter::Sort "virtual bool
gdcm::IPPSorter::Sort(std::vector< std::string > const &filenames)

Main entry point to the sorter. It will execute the filter, option
should be set before running this function (SetZSpacingTolerance, ...)
Return value indicate if sorting could be achived. Warning this does
NOT imply that spacing is consistent, it only means the file are
sorted according to IPP You should check if ZSpacing is 0 or not to
deduce if file are actually a 3D volume ";


// File: classstd_1_1istream.xml
%feature("docstring") std::istream "

STL class. ";


// File: classstd_1_1istringstream.xml
%feature("docstring") std::istringstream "

STL class. ";


// File: classgdcm_1_1Item.xml
%feature("docstring") gdcm::Item "

Class to represent an Item A component of the value of a Data Element
that is of Value Representation Sequence of Items. An Item contains a
Data Set . See PS 3.5 7.5.1 Item Encoding Rules Each Item of a Data
Element of VR SQ shall be encoded as a DICOM Standart Data Element
with a specific Data Element Tag of Value (FFFE,E000). The Item Tag is
followed by a 4 byte Item Length field encoded in one of the following
two ways Explicit/ Implicit.

ITEM: A component of the Value of a Data Element that is of Value
Representation Sequence of Items. An Item contains a Data Set.

C++ includes: gdcmItem.h ";

%feature("docstring")  gdcm::Item::Item "gdcm::Item::Item() ";

%feature("docstring")  gdcm::Item::Item "gdcm::Item::Item(Item const
&val) ";

%feature("docstring")  gdcm::Item::Clear "void gdcm::Item::Clear()

Clear Data Element (make Value empty and invalidate Tag & VR) ";

%feature("docstring")  gdcm::Item::FindDataElement "bool
gdcm::Item::FindDataElement(const Tag &t) const ";

%feature("docstring")  gdcm::Item::GetDataElement "const DataElement&
gdcm::Item::GetDataElement(const Tag &t) const ";

%feature("docstring")  gdcm::Item::GetLength "VL
gdcm::Item::GetLength() const ";

%feature("docstring")  gdcm::Item::GetNestedDataSet "const DataSet&
gdcm::Item::GetNestedDataSet() const ";

%feature("docstring")  gdcm::Item::GetNestedDataSet "DataSet&
gdcm::Item::GetNestedDataSet() ";

%feature("docstring")  gdcm::Item::InsertDataElement "void
gdcm::Item::InsertDataElement(const DataElement &de) ";

%feature("docstring")  gdcm::Item::Read "std::istream&
gdcm::Item::Read(std::istream &is) ";

%feature("docstring")  gdcm::Item::SetNestedDataSet "void
gdcm::Item::SetNestedDataSet(const DataSet &nested) ";

%feature("docstring")  gdcm::Item::Write "const std::ostream&
gdcm::Item::Write(std::ostream &os) const ";


// File: classgdcm_1_1IterationEvent.xml
%feature("docstring") gdcm::IterationEvent "C++ includes: gdcmEvent.h
";


// File: classstd_1_1list_1_1iterator.xml
%feature("docstring") std::list::iterator "

STL iterator class. ";


// File: classstd_1_1forward__list_1_1iterator.xml
%feature("docstring") std::forward_list::iterator "

STL iterator class. ";


// File: classstd_1_1map_1_1iterator.xml
%feature("docstring") std::map::iterator "

STL iterator class. ";


// File: classstd_1_1unordered__map_1_1iterator.xml
%feature("docstring") std::unordered_map::iterator "

STL iterator class. ";


// File: classstd_1_1multimap_1_1iterator.xml
%feature("docstring") std::multimap::iterator "

STL iterator class. ";


// File: classstd_1_1basic__string_1_1iterator.xml
%feature("docstring") std::basic_string::iterator "

STL iterator class. ";


// File: classstd_1_1unordered__multimap_1_1iterator.xml
%feature("docstring") std::unordered_multimap::iterator "

STL iterator class. ";


// File: classstd_1_1set_1_1iterator.xml
%feature("docstring") std::set::iterator "

STL iterator class. ";


// File: classstd_1_1string_1_1iterator.xml
%feature("docstring") std::string::iterator "

STL iterator class. ";


// File: classstd_1_1unordered__set_1_1iterator.xml
%feature("docstring") std::unordered_set::iterator "

STL iterator class. ";


// File: classstd_1_1wstring_1_1iterator.xml
%feature("docstring") std::wstring::iterator "

STL iterator class. ";


// File: classstd_1_1multiset_1_1iterator.xml
%feature("docstring") std::multiset::iterator "

STL iterator class. ";


// File: classstd_1_1unordered__multiset_1_1iterator.xml
%feature("docstring") std::unordered_multiset::iterator "

STL iterator class. ";


// File: classstd_1_1vector_1_1iterator.xml
%feature("docstring") std::vector::iterator "

STL iterator class. ";


// File: classstd_1_1deque_1_1iterator.xml
%feature("docstring") std::deque::iterator "

STL iterator class. ";


// File: classgdcm_1_1JPEG12Codec.xml
%feature("docstring") gdcm::JPEG12Codec "

Class to do JPEG 12bits (lossy & lossless)

internal class

C++ includes: gdcmJPEG12Codec.h ";

%feature("docstring")  gdcm::JPEG12Codec::JPEG12Codec "gdcm::JPEG12Codec::JPEG12Codec() ";

%feature("docstring")  gdcm::JPEG12Codec::~JPEG12Codec "gdcm::JPEG12Codec::~JPEG12Codec() ";

%feature("docstring")  gdcm::JPEG12Codec::DecodeByStreams "bool
gdcm::JPEG12Codec::DecodeByStreams(std::istream &is, std::ostream &os)
";

%feature("docstring")  gdcm::JPEG12Codec::GetHeaderInfo "bool
gdcm::JPEG12Codec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
";

%feature("docstring")  gdcm::JPEG12Codec::InternalCode "bool
gdcm::JPEG12Codec::InternalCode(const char *input, unsigned long len,
std::ostream &os) ";


// File: classgdcm_1_1JPEG16Codec.xml
%feature("docstring") gdcm::JPEG16Codec "

Class to do JPEG 16bits (lossless)

internal class

C++ includes: gdcmJPEG16Codec.h ";

%feature("docstring")  gdcm::JPEG16Codec::JPEG16Codec "gdcm::JPEG16Codec::JPEG16Codec() ";

%feature("docstring")  gdcm::JPEG16Codec::~JPEG16Codec "gdcm::JPEG16Codec::~JPEG16Codec() ";

%feature("docstring")  gdcm::JPEG16Codec::DecodeByStreams "bool
gdcm::JPEG16Codec::DecodeByStreams(std::istream &is, std::ostream &os)
";

%feature("docstring")  gdcm::JPEG16Codec::GetHeaderInfo "bool
gdcm::JPEG16Codec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
";

%feature("docstring")  gdcm::JPEG16Codec::InternalCode "bool
gdcm::JPEG16Codec::InternalCode(const char *input, unsigned long len,
std::ostream &os) ";


// File: classgdcm_1_1JPEG2000Codec.xml
%feature("docstring") gdcm::JPEG2000Codec "

Class to do JPEG 2000.

the class will produce JPC (JPEG 2000 codestream), since some private
implementor are using full jp2 file the decoder tolerate jp2 input
this is an implementation of an ImageCodec

C++ includes: gdcmJPEG2000Codec.h ";

%feature("docstring")  gdcm::JPEG2000Codec::JPEG2000Codec "gdcm::JPEG2000Codec::JPEG2000Codec() ";

%feature("docstring")  gdcm::JPEG2000Codec::~JPEG2000Codec "gdcm::JPEG2000Codec::~JPEG2000Codec() ";

%feature("docstring")  gdcm::JPEG2000Codec::CanCode "bool
gdcm::JPEG2000Codec::CanCode(TransferSyntax const &ts) const

Return whether this coder support this transfer syntax (can code it)
";

%feature("docstring")  gdcm::JPEG2000Codec::CanDecode "bool
gdcm::JPEG2000Codec::CanDecode(TransferSyntax const &ts) const

Return whether this decoder support this transfer syntax (can decode
it) ";

%feature("docstring")  gdcm::JPEG2000Codec::Clone "virtual
ImageCodec* gdcm::JPEG2000Codec::Clone() const ";

%feature("docstring")  gdcm::JPEG2000Codec::Code "bool
gdcm::JPEG2000Codec::Code(DataElement const &in, DataElement &out)

Code. ";

%feature("docstring")  gdcm::JPEG2000Codec::Decode "bool
gdcm::JPEG2000Codec::Decode(DataElement const &is, DataElement &os)

Decode. ";

%feature("docstring")  gdcm::JPEG2000Codec::GetHeaderInfo "virtual
bool gdcm::JPEG2000Codec::GetHeaderInfo(std::istream &is,
TransferSyntax &ts) ";

%feature("docstring")  gdcm::JPEG2000Codec::GetQuality "double
gdcm::JPEG2000Codec::GetQuality(unsigned int idx=0) const ";

%feature("docstring")  gdcm::JPEG2000Codec::GetRate "double
gdcm::JPEG2000Codec::GetRate(unsigned int idx=0) const ";

%feature("docstring")  gdcm::JPEG2000Codec::SetNumberOfResolutions "void gdcm::JPEG2000Codec::SetNumberOfResolutions(unsigned int nres) ";

%feature("docstring")  gdcm::JPEG2000Codec::SetQuality "void
gdcm::JPEG2000Codec::SetQuality(unsigned int idx, double q) ";

%feature("docstring")  gdcm::JPEG2000Codec::SetRate "void
gdcm::JPEG2000Codec::SetRate(unsigned int idx, double rate) ";

%feature("docstring")  gdcm::JPEG2000Codec::SetReversible "void
gdcm::JPEG2000Codec::SetReversible(bool res) ";

%feature("docstring")  gdcm::JPEG2000Codec::SetTileSize "void
gdcm::JPEG2000Codec::SetTileSize(unsigned int tx, unsigned int ty) ";


// File: classgdcm_1_1JPEG8Codec.xml
%feature("docstring") gdcm::JPEG8Codec "

Class to do JPEG 8bits (lossy & lossless)

internal class

C++ includes: gdcmJPEG8Codec.h ";

%feature("docstring")  gdcm::JPEG8Codec::JPEG8Codec "gdcm::JPEG8Codec::JPEG8Codec() ";

%feature("docstring")  gdcm::JPEG8Codec::~JPEG8Codec "gdcm::JPEG8Codec::~JPEG8Codec() ";

%feature("docstring")  gdcm::JPEG8Codec::DecodeByStreams "bool
gdcm::JPEG8Codec::DecodeByStreams(std::istream &is, std::ostream &os)
";

%feature("docstring")  gdcm::JPEG8Codec::GetHeaderInfo "bool
gdcm::JPEG8Codec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
";

%feature("docstring")  gdcm::JPEG8Codec::InternalCode "bool
gdcm::JPEG8Codec::InternalCode(const char *input, unsigned long len,
std::ostream &os) ";


// File: classgdcm_1_1JPEGCodec.xml
%feature("docstring") gdcm::JPEGCodec "

JPEG codec Class to do JPEG (8bits, 12bits, 16bits lossy & lossless).
It redispatch in between the different codec implementation:
JPEG8Codec, JPEG12Codec & JPEG16Codec It also support inconsistency in
between DICOM header and JPEG compressed stream ImageCodec
implementation for the JPEG case.

Things you should know if you ever want to dive into DICOM/JPEG world
(among other):

http://groups.google.com/group/comp.protocols.dicom/browse_thread/thread/625e46919f2080e1

http://groups.google.com/group/comp.protocols.dicom/browse_thread/thread/75fdfccc65a6243

http://groups.google.com/group/comp.protocols.dicom/browse_thread/thread/2d525ef6a2f093ed

http://groups.google.com/group/comp.protocols.dicom/browse_thread/thread/6b93af410f8c921f

C++ includes: gdcmJPEGCodec.h ";

%feature("docstring")  gdcm::JPEGCodec::JPEGCodec "gdcm::JPEGCodec::JPEGCodec() ";

%feature("docstring")  gdcm::JPEGCodec::~JPEGCodec "gdcm::JPEGCodec::~JPEGCodec() ";

%feature("docstring")  gdcm::JPEGCodec::CanCode "bool
gdcm::JPEGCodec::CanCode(TransferSyntax const &ts) const

Return whether this coder support this transfer syntax (can code it)
";

%feature("docstring")  gdcm::JPEGCodec::CanDecode "bool
gdcm::JPEGCodec::CanDecode(TransferSyntax const &ts) const

Return whether this decoder support this transfer syntax (can decode
it) ";

%feature("docstring")  gdcm::JPEGCodec::Clone "virtual ImageCodec*
gdcm::JPEGCodec::Clone() const ";

%feature("docstring")  gdcm::JPEGCodec::Code "bool
gdcm::JPEGCodec::Code(DataElement const &in, DataElement &out)

Compress into JPEG. ";

%feature("docstring")  gdcm::JPEGCodec::ComputeOffsetTable "void
gdcm::JPEGCodec::ComputeOffsetTable(bool b)

Compute the offset table: ";

%feature("docstring")  gdcm::JPEGCodec::Decode "bool
gdcm::JPEGCodec::Decode(DataElement const &is, DataElement &os)

Decode. ";

%feature("docstring")  gdcm::JPEGCodec::EncodeBuffer "virtual bool
gdcm::JPEGCodec::EncodeBuffer(std::ostream &out, const char *inbuffer,
size_t inlen) ";

%feature("docstring")  gdcm::JPEGCodec::GetHeaderInfo "virtual bool
gdcm::JPEGCodec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
";

%feature("docstring")  gdcm::JPEGCodec::GetLossless "bool
gdcm::JPEGCodec::GetLossless() const ";

%feature("docstring")  gdcm::JPEGCodec::GetQuality "double
gdcm::JPEGCodec::GetQuality() const ";

%feature("docstring")  gdcm::JPEGCodec::SetLossless "void
gdcm::JPEGCodec::SetLossless(bool l) ";

%feature("docstring")  gdcm::JPEGCodec::SetPixelFormat "void
gdcm::JPEGCodec::SetPixelFormat(PixelFormat const &pf) ";

%feature("docstring")  gdcm::JPEGCodec::SetQuality "void
gdcm::JPEGCodec::SetQuality(double q) ";


// File: classgdcm_1_1JPEGLSCodec.xml
%feature("docstring") gdcm::JPEGLSCodec "

JPEG-LS.

codec that implement the JPEG-LS compression this is an implementation
of ImageCodec for JPEG-LS  It uses the CharLS JPEG-LS
implementationhttp://charls.codeplex.com

C++ includes: gdcmJPEGLSCodec.h ";

%feature("docstring")  gdcm::JPEGLSCodec::JPEGLSCodec "gdcm::JPEGLSCodec::JPEGLSCodec() ";

%feature("docstring")  gdcm::JPEGLSCodec::~JPEGLSCodec "gdcm::JPEGLSCodec::~JPEGLSCodec() ";

%feature("docstring")  gdcm::JPEGLSCodec::CanCode "bool
gdcm::JPEGLSCodec::CanCode(TransferSyntax const &ts) const

Return whether this coder support this transfer syntax (can code it)
";

%feature("docstring")  gdcm::JPEGLSCodec::CanDecode "bool
gdcm::JPEGLSCodec::CanDecode(TransferSyntax const &ts) const

Return whether this decoder support this transfer syntax (can decode
it) ";

%feature("docstring")  gdcm::JPEGLSCodec::Clone "virtual ImageCodec*
gdcm::JPEGLSCodec::Clone() const ";

%feature("docstring")  gdcm::JPEGLSCodec::Code "bool
gdcm::JPEGLSCodec::Code(DataElement const &in, DataElement &out)

Code. ";

%feature("docstring")  gdcm::JPEGLSCodec::Decode "bool
gdcm::JPEGLSCodec::Decode(DataElement const &is, DataElement &os)

Decode. ";

%feature("docstring")  gdcm::JPEGLSCodec::Decode "bool
gdcm::JPEGLSCodec::Decode(DataElement const &in, char *outBuffer,
size_t inBufferLength, uint32_t inXMin, uint32_t inXMax, uint32_t
inYMin, uint32_t inYMax, uint32_t inZMin, uint32_t inZMax) ";

%feature("docstring")  gdcm::JPEGLSCodec::GetBufferLength "unsigned
long gdcm::JPEGLSCodec::GetBufferLength() const ";

%feature("docstring")  gdcm::JPEGLSCodec::GetHeaderInfo "bool
gdcm::JPEGLSCodec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
";

%feature("docstring")  gdcm::JPEGLSCodec::GetLossless "bool
gdcm::JPEGLSCodec::GetLossless() const ";

%feature("docstring")  gdcm::JPEGLSCodec::SetBufferLength "void
gdcm::JPEGLSCodec::SetBufferLength(unsigned long l) ";

%feature("docstring")  gdcm::JPEGLSCodec::SetLossless "void
gdcm::JPEGLSCodec::SetLossless(bool l) ";

%feature("docstring")  gdcm::JPEGLSCodec::SetLossyError "void
gdcm::JPEGLSCodec::SetLossyError(int error)

[0-3] generally ";


// File: classgdcm_1_1JSON.xml
%feature("docstring") gdcm::JSON "C++ includes: gdcmJSON.h ";

%feature("docstring")  gdcm::JSON::JSON "gdcm::JSON::JSON() ";

%feature("docstring")  gdcm::JSON::~JSON "gdcm::JSON::~JSON() ";

%feature("docstring")  gdcm::JSON::Code "bool
gdcm::JSON::Code(DataSet const &in, std::ostream &os) ";

%feature("docstring")  gdcm::JSON::Decode "bool
gdcm::JSON::Decode(std::istream &is, DataSet &out) ";

%feature("docstring")  gdcm::JSON::GetPrettyPrint "bool
gdcm::JSON::GetPrettyPrint() const ";

%feature("docstring")  gdcm::JSON::PrettyPrintOff "void
gdcm::JSON::PrettyPrintOff() ";

%feature("docstring")  gdcm::JSON::PrettyPrintOn "void
gdcm::JSON::PrettyPrintOn() ";

%feature("docstring")  gdcm::JSON::SetPrettyPrint "void
gdcm::JSON::SetPrettyPrint(bool onoff) ";


// File: classgdcm_1_1KAKADUCodec.xml
%feature("docstring") gdcm::KAKADUCodec "

KAKADUCodec.

C++ includes: gdcmKAKADUCodec.h ";

%feature("docstring")  gdcm::KAKADUCodec::KAKADUCodec "gdcm::KAKADUCodec::KAKADUCodec() ";

%feature("docstring")  gdcm::KAKADUCodec::~KAKADUCodec "gdcm::KAKADUCodec::~KAKADUCodec() ";

%feature("docstring")  gdcm::KAKADUCodec::CanCode "bool
gdcm::KAKADUCodec::CanCode(TransferSyntax const &ts) const

Return whether this coder support this transfer syntax (can code it)
";

%feature("docstring")  gdcm::KAKADUCodec::CanDecode "bool
gdcm::KAKADUCodec::CanDecode(TransferSyntax const &ts) const

Return whether this decoder support this transfer syntax (can decode
it) ";

%feature("docstring")  gdcm::KAKADUCodec::Clone "virtual ImageCodec*
gdcm::KAKADUCodec::Clone() const ";

%feature("docstring")  gdcm::KAKADUCodec::Code "bool
gdcm::KAKADUCodec::Code(DataElement const &in, DataElement &out)

Code. ";

%feature("docstring")  gdcm::KAKADUCodec::Decode "bool
gdcm::KAKADUCodec::Decode(DataElement const &is, DataElement &os)

Decode. ";


// File: classstd_1_1length__error.xml
%feature("docstring") std::length_error "

STL class. ";


// File: classstd_1_1list.xml
%feature("docstring") std::list "

STL class. ";


// File: classgdcm_1_1LO.xml
%feature("docstring") gdcm::LO "

LO.

TODO

C++ includes: gdcmLO.h ";

%feature("docstring")  gdcm::LO::LO "gdcm::LO::LO() ";

%feature("docstring")  gdcm::LO::LO "gdcm::LO::LO(const value_type
*s) ";

%feature("docstring")  gdcm::LO::LO "gdcm::LO::LO(const value_type
*s, size_type n) ";

%feature("docstring")  gdcm::LO::LO "gdcm::LO::LO(const Superclass
&s, size_type pos=0, size_type n=npos) ";

%feature("docstring")  gdcm::LO::IsValid "bool gdcm::LO::IsValid()
const ";


// File: classstd_1_1logic__error.xml
%feature("docstring") std::logic_error "

STL class. ";


// File: classgdcm_1_1LookupTable.xml
%feature("docstring") gdcm::LookupTable "

LookupTable class.

C++ includes: gdcmLookupTable.h ";

%feature("docstring")  gdcm::LookupTable::LookupTable "gdcm::LookupTable::LookupTable() ";

%feature("docstring")  gdcm::LookupTable::LookupTable "gdcm::LookupTable::LookupTable(LookupTable const &lut) ";

%feature("docstring")  gdcm::LookupTable::~LookupTable "gdcm::LookupTable::~LookupTable() ";

%feature("docstring")  gdcm::LookupTable::Allocate "void
gdcm::LookupTable::Allocate(unsigned short bitsample=8)

Allocate the LUT. ";

%feature("docstring")  gdcm::LookupTable::Clear "void
gdcm::LookupTable::Clear()

Clear the LUT. ";

%feature("docstring")  gdcm::LookupTable::Decode "void
gdcm::LookupTable::Decode(std::istream &is, std::ostream &os) const

Decode the LUT. ";

%feature("docstring")  gdcm::LookupTable::Decode "bool
gdcm::LookupTable::Decode(char *outputbuffer, size_t outlen, const
char *inputbuffer, size_t inlen) const

Decode the LUT outputbuffer will contains the RGB decoded PALETTE
COLOR input image of size inlen the outputbuffer should be at least 3
times the size of inlen ";

%feature("docstring")  gdcm::LookupTable::GetBitSample "unsigned
short gdcm::LookupTable::GetBitSample() const

return the bit sample ";

%feature("docstring")  gdcm::LookupTable::GetBufferAsRGBA "bool
gdcm::LookupTable::GetBufferAsRGBA(unsigned char *rgba) const

return the LUT as RGBA buffer ";

%feature("docstring")  gdcm::LookupTable::GetLUT "void
gdcm::LookupTable::GetLUT(LookupTableType type, unsigned char *array,
unsigned int &length) const ";

%feature("docstring")  gdcm::LookupTable::GetLUTDescriptor "void
gdcm::LookupTable::GetLUTDescriptor(LookupTableType type, unsigned
short &length, unsigned short &subscript, unsigned short &bitsize)
const ";

%feature("docstring")  gdcm::LookupTable::GetLUTLength "unsigned int
gdcm::LookupTable::GetLUTLength(LookupTableType type) const ";

%feature("docstring")  gdcm::LookupTable::GetPointer "const unsigned
char* gdcm::LookupTable::GetPointer() const

return a raw pointer to the LUT ";

%feature("docstring")  gdcm::LookupTable::InitializeBlueLUT "void
gdcm::LookupTable::InitializeBlueLUT(unsigned short length, unsigned
short subscript, unsigned short bitsize) ";

%feature("docstring")  gdcm::LookupTable::Initialized "bool
gdcm::LookupTable::Initialized() const

return whether the LUT has been initialized ";

%feature("docstring")  gdcm::LookupTable::InitializeGreenLUT "void
gdcm::LookupTable::InitializeGreenLUT(unsigned short length, unsigned
short subscript, unsigned short bitsize) ";

%feature("docstring")  gdcm::LookupTable::InitializeLUT "void
gdcm::LookupTable::InitializeLUT(LookupTableType type, unsigned short
length, unsigned short subscript, unsigned short bitsize)

Generic interface: ";

%feature("docstring")  gdcm::LookupTable::InitializeRedLUT "void
gdcm::LookupTable::InitializeRedLUT(unsigned short length, unsigned
short subscript, unsigned short bitsize)

RED / GREEN / BLUE specific: ";

%feature("docstring")  gdcm::LookupTable::Print "void
gdcm::LookupTable::Print(std::ostream &) const ";

%feature("docstring")  gdcm::LookupTable::SetBlueLUT "void
gdcm::LookupTable::SetBlueLUT(const unsigned char *blue, unsigned int
length) ";

%feature("docstring")  gdcm::LookupTable::SetGreenLUT "void
gdcm::LookupTable::SetGreenLUT(const unsigned char *green, unsigned
int length) ";

%feature("docstring")  gdcm::LookupTable::SetLUT "virtual void
gdcm::LookupTable::SetLUT(LookupTableType type, const unsigned char
*array, unsigned int length) ";

%feature("docstring")  gdcm::LookupTable::SetRedLUT "void
gdcm::LookupTable::SetRedLUT(const unsigned char *red, unsigned int
length) ";

%feature("docstring")  gdcm::LookupTable::WriteBufferAsRGBA "bool
gdcm::LookupTable::WriteBufferAsRGBA(const unsigned char *rgba)

Write the LUT as RGBA. ";


// File: structgdcm_1_1Scanner_1_1ltstr.xml
%feature("docstring") gdcm::Scanner::ltstr "C++ includes:
gdcmScanner.h ";


// File: classgdcm_1_1Macro.xml
%feature("docstring") gdcm::Macro "

Class for representing a Macro.

Attribute Macro: a set of Attributes that are described in a single
table that is referenced by multiple Module or other tables.

See:   Module

C++ includes: gdcmMacro.h ";

%feature("docstring")  gdcm::Macro::Macro "gdcm::Macro::Macro() ";

%feature("docstring")  gdcm::Macro::AddMacroEntry "void
gdcm::Macro::AddMacroEntry(const Tag &tag, const MacroEntry &module)

Will add a ModuleEntry direcly at root-level. See Macro for nested-
included level. ";

%feature("docstring")  gdcm::Macro::Clear "void gdcm::Macro::Clear()
";

%feature("docstring")  gdcm::Macro::FindMacroEntry "bool
gdcm::Macro::FindMacroEntry(const Tag &tag) const

Find or Get a ModuleEntry. ModuleEntry are either search are root-
level or within nested-macro included in module. ";

%feature("docstring")  gdcm::Macro::GetMacroEntry "const MacroEntry&
gdcm::Macro::GetMacroEntry(const Tag &tag) const ";

%feature("docstring")  gdcm::Macro::GetName "const char*
gdcm::Macro::GetName() const ";

%feature("docstring")  gdcm::Macro::SetName "void
gdcm::Macro::SetName(const char *name) ";

%feature("docstring")  gdcm::Macro::Verify "bool
gdcm::Macro::Verify(const DataSet &ds, Usage const &usage) const ";


// File: classgdcm_1_1Macros.xml
%feature("docstring") gdcm::Macros "

Class for representing a Modules.

bla

See:   Module

C++ includes: gdcmMacros.h ";

%feature("docstring")  gdcm::Macros::Macros "gdcm::Macros::Macros()
";

%feature("docstring")  gdcm::Macros::AddMacro "void
gdcm::Macros::AddMacro(const char *ref, const Macro &module) ";

%feature("docstring")  gdcm::Macros::Clear "void
gdcm::Macros::Clear() ";

%feature("docstring")  gdcm::Macros::GetMacro "const Macro&
gdcm::Macros::GetMacro(const char *name) const ";

%feature("docstring")  gdcm::Macros::IsEmpty "bool
gdcm::Macros::IsEmpty() const ";


// File: classstd_1_1map.xml
%feature("docstring") std::map "

STL class. ";


// File: classgdcm_1_1network_1_1MaximumLengthSub.xml
%feature("docstring") gdcm::network::MaximumLengthSub "

MaximumLengthSub Annex D Table D.1-1 MAXIMUM LENGTH SUB-ITEM FIELDS (A
-ASSOCIATE-RQ)

or

Table D.1-2 Maximum length sub-item fields (A-ASSOCIATE-AC)

C++ includes: gdcmMaximumLengthSub.h ";

%feature("docstring")
gdcm::network::MaximumLengthSub::MaximumLengthSub "gdcm::network::MaximumLengthSub::MaximumLengthSub() ";

%feature("docstring")
gdcm::network::MaximumLengthSub::GetMaximumLength "uint32_t
gdcm::network::MaximumLengthSub::GetMaximumLength() const ";

%feature("docstring")  gdcm::network::MaximumLengthSub::Print "void
gdcm::network::MaximumLengthSub::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::network::MaximumLengthSub::Read "std::istream& gdcm::network::MaximumLengthSub::Read(std::istream &is)
";

%feature("docstring")
gdcm::network::MaximumLengthSub::SetMaximumLength "void
gdcm::network::MaximumLengthSub::SetMaximumLength(uint32_t
maximumlength) ";

%feature("docstring")  gdcm::network::MaximumLengthSub::Size "size_t
gdcm::network::MaximumLengthSub::Size() const ";

%feature("docstring")  gdcm::network::MaximumLengthSub::Write "const
std::ostream& gdcm::network::MaximumLengthSub::Write(std::ostream &os)
const ";


// File: classgdcm_1_1MD5.xml
%feature("docstring") gdcm::MD5 "

Class for MD5.

WARNING:  this class is able to pick from two implementations:

a lightweight md5 implementation (when GDCM_BUILD_TESTING is turned
ON)

the one from OpenSSL (when GDCM_USE_SYSTEM_OPENSSL is turned ON)

In all other cases it will return an error

C++ includes: gdcmMD5.h ";

%feature("docstring")  gdcm::MD5::MD5 "gdcm::MD5::MD5() ";

%feature("docstring")  gdcm::MD5::~MD5 "gdcm::MD5::~MD5() ";


// File: classgdcm_1_1MediaStorage.xml
%feature("docstring") gdcm::MediaStorage "

MediaStorage.

FIXME There should not be any notion of Image and/or PDF at that point
Only the codec can answer yes I support this Media Storage or not...
For instance an ImageCodec will answer yes to most of them while a
PDFCodec will answer only for the Encapsulated PDF

See:   UIDs

C++ includes: gdcmMediaStorage.h ";

%feature("docstring")  gdcm::MediaStorage::MediaStorage "gdcm::MediaStorage::MediaStorage(MSType type=MS_END) ";

%feature("docstring")  gdcm::MediaStorage::GetModality "const char*
gdcm::MediaStorage::GetModality() const ";

%feature("docstring")  gdcm::MediaStorage::GetModalityDimension "unsigned int gdcm::MediaStorage::GetModalityDimension() const ";

%feature("docstring")  gdcm::MediaStorage::GetString "const char*
gdcm::MediaStorage::GetString() const

Return the Media String of the object. ";

%feature("docstring")  gdcm::MediaStorage::GuessFromModality "void
gdcm::MediaStorage::GuessFromModality(const char *modality, unsigned
int dimension=2) ";

%feature("docstring")  gdcm::MediaStorage::IsUndefined "bool
gdcm::MediaStorage::IsUndefined() const ";

%feature("docstring")  gdcm::MediaStorage::SetFromDataSet "bool
gdcm::MediaStorage::SetFromDataSet(DataSet const &ds)

Advanced user only (functions should be protected level...) Those
function are lower level than SetFromFile ";

%feature("docstring")  gdcm::MediaStorage::SetFromFile "bool
gdcm::MediaStorage::SetFromFile(File const &file)

Attempt to set the MediaStorage from a file: WARNING: When no
MediaStorage & Modality are found BUT a PixelData element is found
then MediaStorage is set to the default SecondaryCaptureImageStorage
(return value is false in this case) ";

%feature("docstring")  gdcm::MediaStorage::SetFromHeader "bool
gdcm::MediaStorage::SetFromHeader(FileMetaInformation const &fmi) ";

%feature("docstring")  gdcm::MediaStorage::SetFromModality "bool
gdcm::MediaStorage::SetFromModality(DataSet const &ds) ";


// File: classgdcm_1_1MemberCommand.xml
%feature("docstring") gdcm::MemberCommand "

Command subclass that calls a pointer to a member function.

MemberCommand calls a pointer to a member function with the same
arguments as Execute on Command.

C++ includes: gdcmCommand.h ";

%feature("docstring")  gdcm::MemberCommand::Execute "virtual void
gdcm::MemberCommand< T >::Execute(Subject *caller, const Event &event)

Invoke the member function. ";

%feature("docstring")  gdcm::MemberCommand::Execute "virtual void
gdcm::MemberCommand< T >::Execute(const Subject *caller, const Event
&event)

Invoke the member function with a const object. ";

%feature("docstring")  gdcm::MemberCommand::SetCallbackFunction "void
gdcm::MemberCommand< T >::SetCallbackFunction(T *object,
TMemberFunctionPointer memberFunction)

Run-time type information (and related methods). Set the callback
function along with the object that it will be invoked on. ";

%feature("docstring")  gdcm::MemberCommand::SetCallbackFunction "void
gdcm::MemberCommand< T >::SetCallbackFunction(T *object,
TConstMemberFunctionPointer memberFunction) ";


// File: classgdcm_1_1MeshPrimitive.xml
%feature("docstring") gdcm::MeshPrimitive "

This class defines surface mesh primitives. It is designed from
surface mesh primitives macro.

See:  PS 3.3 C.27.4

C++ includes: gdcmMeshPrimitive.h ";

%feature("docstring")  gdcm::MeshPrimitive::MeshPrimitive "gdcm::MeshPrimitive::MeshPrimitive() ";

%feature("docstring")  gdcm::MeshPrimitive::~MeshPrimitive "virtual
gdcm::MeshPrimitive::~MeshPrimitive() ";

%feature("docstring")  gdcm::MeshPrimitive::AddPrimitiveData "void
gdcm::MeshPrimitive::AddPrimitiveData(DataElement const &de) ";

%feature("docstring")  gdcm::MeshPrimitive::GetNumberOfPrimitivesData
"unsigned int gdcm::MeshPrimitive::GetNumberOfPrimitivesData() const
";

%feature("docstring")  gdcm::MeshPrimitive::GetPrimitiveData "const
DataElement& gdcm::MeshPrimitive::GetPrimitiveData() const ";

%feature("docstring")  gdcm::MeshPrimitive::GetPrimitiveData "DataElement& gdcm::MeshPrimitive::GetPrimitiveData() ";

%feature("docstring")  gdcm::MeshPrimitive::GetPrimitiveData "const
DataElement& gdcm::MeshPrimitive::GetPrimitiveData(const unsigned int
idx) const ";

%feature("docstring")  gdcm::MeshPrimitive::GetPrimitiveData "DataElement& gdcm::MeshPrimitive::GetPrimitiveData(const unsigned int
idx) ";

%feature("docstring")  gdcm::MeshPrimitive::GetPrimitivesData "const
PrimitivesData& gdcm::MeshPrimitive::GetPrimitivesData() const ";

%feature("docstring")  gdcm::MeshPrimitive::GetPrimitivesData "PrimitivesData& gdcm::MeshPrimitive::GetPrimitivesData() ";

%feature("docstring")  gdcm::MeshPrimitive::GetPrimitiveType "MPType
gdcm::MeshPrimitive::GetPrimitiveType() const ";

%feature("docstring")  gdcm::MeshPrimitive::SetPrimitiveData "void
gdcm::MeshPrimitive::SetPrimitiveData(DataElement const &de) ";

%feature("docstring")  gdcm::MeshPrimitive::SetPrimitiveData "void
gdcm::MeshPrimitive::SetPrimitiveData(const unsigned int idx,
DataElement const &de) ";

%feature("docstring")  gdcm::MeshPrimitive::SetPrimitivesData "void
gdcm::MeshPrimitive::SetPrimitivesData(PrimitivesData const &DEs) ";

%feature("docstring")  gdcm::MeshPrimitive::SetPrimitiveType "void
gdcm::MeshPrimitive::SetPrimitiveType(const MPType type) ";


// File: classgdcm_1_1ModifiedEvent.xml
%feature("docstring") gdcm::ModifiedEvent "C++ includes: gdcmEvent.h
";


// File: classgdcm_1_1Module.xml
%feature("docstring") gdcm::Module "

Class for representing a Module.

Module: A set of Attributes within an Information Entity or Normalized
IOD which are logically related to each other.

See:   Macro

C++ includes: gdcmModule.h ";

%feature("docstring")  gdcm::Module::Module "gdcm::Module::Module()
";

%feature("docstring")  gdcm::Module::AddMacro "void
gdcm::Module::AddMacro(const char *include) ";

%feature("docstring")  gdcm::Module::AddModuleEntry "void
gdcm::Module::AddModuleEntry(const Tag &tag, const ModuleEntry
&module)

Will add a ModuleEntry direcly at root-level. See Macro for nested-
included level. ";

%feature("docstring")  gdcm::Module::Clear "void
gdcm::Module::Clear() ";

%feature("docstring")  gdcm::Module::FindModuleEntryInMacros "bool
gdcm::Module::FindModuleEntryInMacros(Macros const &macros, const Tag
&tag) const

Find or Get a ModuleEntry. ModuleEntry are either search are root-
level or within nested-macro included in module. ";

%feature("docstring")  gdcm::Module::GetModuleEntryInMacros "const
ModuleEntry& gdcm::Module::GetModuleEntryInMacros(Macros const
&macros, const Tag &tag) const ";

%feature("docstring")  gdcm::Module::GetName "const char*
gdcm::Module::GetName() const ";

%feature("docstring")  gdcm::Module::SetName "void
gdcm::Module::SetName(const char *name) ";

%feature("docstring")  gdcm::Module::Verify "bool
gdcm::Module::Verify(const DataSet &ds, Usage const &usage) const ";


// File: classgdcm_1_1ModuleEntry.xml
%feature("docstring") gdcm::ModuleEntry "

Class for representing a ModuleEntry.

bla

See:   DictEntry

C++ includes: gdcmModuleEntry.h ";

%feature("docstring")  gdcm::ModuleEntry::ModuleEntry "gdcm::ModuleEntry::ModuleEntry(const char *name=\"\", const char
*type=\"3\", const char *description=\"\") ";

%feature("docstring")  gdcm::ModuleEntry::~ModuleEntry "virtual
gdcm::ModuleEntry::~ModuleEntry() ";

%feature("docstring")  gdcm::ModuleEntry::GetDescription "const
Description& gdcm::ModuleEntry::GetDescription() const ";

%feature("docstring")  gdcm::ModuleEntry::GetName "const char*
gdcm::ModuleEntry::GetName() const ";

%feature("docstring")  gdcm::ModuleEntry::GetType "const Type&
gdcm::ModuleEntry::GetType() const ";

%feature("docstring")  gdcm::ModuleEntry::SetDescription "void
gdcm::ModuleEntry::SetDescription(const char *d) ";

%feature("docstring")  gdcm::ModuleEntry::SetName "void
gdcm::ModuleEntry::SetName(const char *name) ";

%feature("docstring")  gdcm::ModuleEntry::SetType "void
gdcm::ModuleEntry::SetType(const Type &type) ";


// File: classgdcm_1_1Modules.xml
%feature("docstring") gdcm::Modules "

Class for representing a Modules.

bla

See:   Module

C++ includes: gdcmModules.h ";

%feature("docstring")  gdcm::Modules::Modules "gdcm::Modules::Modules() ";

%feature("docstring")  gdcm::Modules::AddModule "void
gdcm::Modules::AddModule(const char *ref, const Module &module) ";

%feature("docstring")  gdcm::Modules::Clear "void
gdcm::Modules::Clear() ";

%feature("docstring")  gdcm::Modules::GetModule "const Module&
gdcm::Modules::GetModule(const char *name) const ";

%feature("docstring")  gdcm::Modules::IsEmpty "bool
gdcm::Modules::IsEmpty() const ";


// File: classgdcm_1_1MovePatientRootQuery.xml
%feature("docstring") gdcm::MovePatientRootQuery "

MovePatientRootQuery contains: the class which will produce a dataset
for c-move with patient root.

C++ includes: gdcmMovePatientRootQuery.h ";

%feature("docstring")
gdcm::MovePatientRootQuery::MovePatientRootQuery "gdcm::MovePatientRootQuery::MovePatientRootQuery() ";

%feature("docstring")
gdcm::MovePatientRootQuery::GetAbstractSyntaxUID "UIDs::TSName
gdcm::MovePatientRootQuery::GetAbstractSyntaxUID() const ";

%feature("docstring")  gdcm::MovePatientRootQuery::GetTagListByLevel "std::vector<Tag> gdcm::MovePatientRootQuery::GetTagListByLevel(const
EQueryLevel &inQueryLevel)

this function will return all tags at a given query level, so that
they maybe selected for searching. The boolean forFind is true if the
query is a find query, or false for a move query. ";

%feature("docstring")  gdcm::MovePatientRootQuery::InitializeDataSet "void gdcm::MovePatientRootQuery::InitializeDataSet(const EQueryLevel
&inQueryLevel)

this function sets tag 8,52 to the appropriate value based on query
level also fills in the right unique tags, as per the standard's
requirements should allow for connection with dcmtk ";

%feature("docstring")  gdcm::MovePatientRootQuery::ValidateQuery "bool gdcm::MovePatientRootQuery::ValidateQuery(bool inStrict=true)
const

have to be able to ensure that 0x8,0x52 is set (which will be true if
InitializeDataSet is called...) that the level is appropriate (ie, not
setting PATIENT for a study query that the tags in the query match the
right level (either required, unique, optional) by default, this
function checks to see if the query is for finding, which is more
permissive than for moving. For moving, only the unique tags are
allowed. 10 Jan 2011: adding in the 'strict' mode. according to the
standard (at least, how I've read it), only tags for a particular
level should be allowed in a particular query (ie, just series level
tags in a series level query). However, it seems that dcm4chee doesn't
share that interpretation. So, if 'inStrict' is false, then tags from
the current level and all higher levels are now considered valid. So,
if you're doing a non-strict series-level query, tags from the patient
and study level can be passed along as well. ";


// File: classgdcm_1_1MoveStudyRootQuery.xml
%feature("docstring") gdcm::MoveStudyRootQuery "

MoveStudyRootQuery contains: the class which will produce a dataset
for C-MOVE with study root.

C++ includes: gdcmMoveStudyRootQuery.h ";

%feature("docstring")  gdcm::MoveStudyRootQuery::MoveStudyRootQuery "gdcm::MoveStudyRootQuery::MoveStudyRootQuery() ";

%feature("docstring")  gdcm::MoveStudyRootQuery::GetAbstractSyntaxUID
"UIDs::TSName gdcm::MoveStudyRootQuery::GetAbstractSyntaxUID() const
";

%feature("docstring")  gdcm::MoveStudyRootQuery::GetTagListByLevel "std::vector<Tag> gdcm::MoveStudyRootQuery::GetTagListByLevel(const
EQueryLevel &inQueryLevel)

this function will return all tags at a given query level, so that
they maybe selected for searching. The boolean forFind is true if the
query is a find query, or false for a move query. ";

%feature("docstring")  gdcm::MoveStudyRootQuery::InitializeDataSet "void gdcm::MoveStudyRootQuery::InitializeDataSet(const EQueryLevel
&inQueryLevel)

this function sets tag 8,52 to the appropriate value based on query
level also fills in the right unique tags, as per the standard's
requirements should allow for connection with dcmtk ";

%feature("docstring")  gdcm::MoveStudyRootQuery::ValidateQuery "bool
gdcm::MoveStudyRootQuery::ValidateQuery(bool inStrict=true) const

have to be able to ensure that 0x8,0x52 is set (which will be true if
InitializeDataSet is called...) that the level is appropriate (ie, not
setting PATIENT for a study query that the tags in the query match the
right level (either required, unique, optional) by default, this
function checks to see if the query is for finding, which is more
permissive than for moving. For moving, only the unique tags are
allowed. 10 Jan 2011: adding in the 'strict' mode. according to the
standard (at least, how I've read it), only tags for a particular
level should be allowed in a particular query (ie, just series level
tags in a series level query). However, it seems that dcm4chee doesn't
share that interpretation. So, if 'inStrict' is false, then tags from
the current level and all higher levels are now considered valid. So,
if you're doing a non-strict series-level query, tags from the patient
and study level can be passed along as well. ";


// File: classstd_1_1multimap.xml
%feature("docstring") std::multimap "

STL class. ";


// File: classstd_1_1multiset.xml
%feature("docstring") std::multiset "

STL class. ";


// File: classgdcm_1_1NestedModuleEntries.xml
%feature("docstring") gdcm::NestedModuleEntries "

Class for representing a NestedModuleEntries.

bla

See:   ModuleEntry

C++ includes: gdcmNestedModuleEntries.h ";

%feature("docstring")  gdcm::NestedModuleEntries::NestedModuleEntries
"gdcm::NestedModuleEntries::NestedModuleEntries(const char
*name=\"\", const char *type=\"3\", const char *description=\"\") ";

%feature("docstring")  gdcm::NestedModuleEntries::AddModuleEntry "void gdcm::NestedModuleEntries::AddModuleEntry(const ModuleEntry &me)
";

%feature("docstring")  gdcm::NestedModuleEntries::GetModuleEntry "const ModuleEntry& gdcm::NestedModuleEntries::GetModuleEntry(SizeType
idx) const ";

%feature("docstring")  gdcm::NestedModuleEntries::GetModuleEntry "ModuleEntry& gdcm::NestedModuleEntries::GetModuleEntry(SizeType idx)
";

%feature("docstring")
gdcm::NestedModuleEntries::GetNumberOfModuleEntries "SizeType
gdcm::NestedModuleEntries::GetNumberOfModuleEntries() ";


// File: classgdcm_1_1NoEvent.xml
%feature("docstring") gdcm::NoEvent "

Define some common GDCM events

C++ includes: gdcmEvent.h ";


// File: classgdcm_1_1Object.xml
%feature("docstring") gdcm::Object "

Object.

main superclass for object that want to use SmartPointer invasive ref
counting system

See:   SmartPointer

C++ includes: gdcmObject.h ";

%feature("docstring")  gdcm::Object::Object "gdcm::Object::Object()
";

%feature("docstring")  gdcm::Object::Object "gdcm::Object::Object(const Object &)

Special requirement for copy/cstor, assignment operator. ";

%feature("docstring")  gdcm::Object::~Object "virtual
gdcm::Object::~Object() ";

%feature("docstring")  gdcm::Object::Print "virtual void
gdcm::Object::Print(std::ostream &) const ";


// File: classstd_1_1ofstream.xml
%feature("docstring") std::ofstream "

STL class. ";


// File: classgdcm_1_1OpenSSLCryptoFactory.xml
%feature("docstring") gdcm::OpenSSLCryptoFactory "C++ includes:
gdcmOpenSSLCryptoFactory.h ";

%feature("docstring")
gdcm::OpenSSLCryptoFactory::OpenSSLCryptoFactory "gdcm::OpenSSLCryptoFactory::OpenSSLCryptoFactory(CryptoLib id) ";

%feature("docstring")  gdcm::OpenSSLCryptoFactory::CreateCMSProvider "CryptographicMessageSyntax*
gdcm::OpenSSLCryptoFactory::CreateCMSProvider() ";


// File: classgdcm_1_1OpenSSLCryptographicMessageSyntax.xml
%feature("docstring") gdcm::OpenSSLCryptographicMessageSyntax "C++
includes: gdcmOpenSSLCryptographicMessageSyntax.h ";

%feature("docstring")
gdcm::OpenSSLCryptographicMessageSyntax::OpenSSLCryptographicMessageSyntax
"gdcm::OpenSSLCryptographicMessageSyntax::OpenSSLCryptographicMessageSyntax()
";

%feature("docstring")
gdcm::OpenSSLCryptographicMessageSyntax::~OpenSSLCryptographicMessageSyntax
"gdcm::OpenSSLCryptographicMessageSyntax::~OpenSSLCryptographicMessageSyntax()
";

%feature("docstring")
gdcm::OpenSSLCryptographicMessageSyntax::Decrypt "bool
gdcm::OpenSSLCryptographicMessageSyntax::Decrypt(char *output, size_t
&outlen, const char *array, size_t len) const

decrypt content from a PKCS#7 envelopedData structure ";

%feature("docstring")
gdcm::OpenSSLCryptographicMessageSyntax::Encrypt "bool
gdcm::OpenSSLCryptographicMessageSyntax::Encrypt(char *output, size_t
&outlen, const char *array, size_t len) const

create a CMS envelopedData structure ";

%feature("docstring")
gdcm::OpenSSLCryptographicMessageSyntax::GetCipherType "CipherTypes
gdcm::OpenSSLCryptographicMessageSyntax::GetCipherType() const ";

%feature("docstring")
gdcm::OpenSSLCryptographicMessageSyntax::ParseCertificateFile "bool
gdcm::OpenSSLCryptographicMessageSyntax::ParseCertificateFile(const
char *filename) ";

%feature("docstring")
gdcm::OpenSSLCryptographicMessageSyntax::ParseKeyFile "bool
gdcm::OpenSSLCryptographicMessageSyntax::ParseKeyFile(const char
*filename) ";

%feature("docstring")
gdcm::OpenSSLCryptographicMessageSyntax::SetCipherType "void
gdcm::OpenSSLCryptographicMessageSyntax::SetCipherType(CipherTypes
type)

Set Cipher Type. Default is: AES256_CIPHER ";

%feature("docstring")
gdcm::OpenSSLCryptographicMessageSyntax::SetPassword "bool
gdcm::OpenSSLCryptographicMessageSyntax::SetPassword(const char *pass,
size_t passLen) ";


// File: classgdcm_1_1OpenSSLP7CryptoFactory.xml
%feature("docstring") gdcm::OpenSSLP7CryptoFactory "C++ includes:
gdcmOpenSSLP7CryptoFactory.h ";

%feature("docstring")
gdcm::OpenSSLP7CryptoFactory::OpenSSLP7CryptoFactory "gdcm::OpenSSLP7CryptoFactory::OpenSSLP7CryptoFactory(CryptoLib id) ";

%feature("docstring")  gdcm::OpenSSLP7CryptoFactory::CreateCMSProvider
"CryptographicMessageSyntax*
gdcm::OpenSSLP7CryptoFactory::CreateCMSProvider() ";


// File: classgdcm_1_1OpenSSLP7CryptographicMessageSyntax.xml
%feature("docstring") gdcm::OpenSSLP7CryptographicMessageSyntax "

Class for CryptographicMessageSyntax encryption. This is just a simple
wrapper around openssl PKCS7_encrypt functionalities.

See online
documentationhttp://www.openssl.org/docs/crypto/PKCS7_encrypt.html

C++ includes: gdcmOpenSSLP7CryptographicMessageSyntax.h ";

%feature("docstring")
gdcm::OpenSSLP7CryptographicMessageSyntax::OpenSSLP7CryptographicMessageSyntax
"gdcm::OpenSSLP7CryptographicMessageSyntax::OpenSSLP7CryptographicMessageSyntax()
";

%feature("docstring")
gdcm::OpenSSLP7CryptographicMessageSyntax::~OpenSSLP7CryptographicMessageSyntax
"gdcm::OpenSSLP7CryptographicMessageSyntax::~OpenSSLP7CryptographicMessageSyntax()
";

%feature("docstring")
gdcm::OpenSSLP7CryptographicMessageSyntax::Decrypt "bool
gdcm::OpenSSLP7CryptographicMessageSyntax::Decrypt(char *output,
size_t &outlen, const char *array, size_t len) const

decrypt content from a PKCS#7 envelopedData structure ";

%feature("docstring")
gdcm::OpenSSLP7CryptographicMessageSyntax::Encrypt "bool
gdcm::OpenSSLP7CryptographicMessageSyntax::Encrypt(char *output,
size_t &outlen, const char *array, size_t len) const

create a PKCS#7 envelopedData structure ";

%feature("docstring")
gdcm::OpenSSLP7CryptographicMessageSyntax::GetCipherType "CipherTypes
gdcm::OpenSSLP7CryptographicMessageSyntax::GetCipherType() const ";

%feature("docstring")
gdcm::OpenSSLP7CryptographicMessageSyntax::ParseCertificateFile "bool
gdcm::OpenSSLP7CryptographicMessageSyntax::ParseCertificateFile(const
char *filename) ";

%feature("docstring")
gdcm::OpenSSLP7CryptographicMessageSyntax::ParseKeyFile "bool
gdcm::OpenSSLP7CryptographicMessageSyntax::ParseKeyFile(const char
*filename) ";

%feature("docstring")
gdcm::OpenSSLP7CryptographicMessageSyntax::SetCipherType "void
gdcm::OpenSSLP7CryptographicMessageSyntax::SetCipherType(CipherTypes
type)

Set Cipher Type. Default is: AES256_CIPHER ";

%feature("docstring")
gdcm::OpenSSLP7CryptographicMessageSyntax::SetPassword "bool
gdcm::OpenSSLP7CryptographicMessageSyntax::SetPassword(const char *,
size_t) ";


// File: classgdcm_1_1Orientation.xml
%feature("docstring") gdcm::Orientation "

class to handle Orientation

C++ includes: gdcmOrientation.h ";

%feature("docstring")  gdcm::Orientation::Orientation "gdcm::Orientation::Orientation() ";

%feature("docstring")  gdcm::Orientation::~Orientation "gdcm::Orientation::~Orientation() ";

%feature("docstring")  gdcm::Orientation::Print "void
gdcm::Orientation::Print(std::ostream &) const

Print. ";


// File: classstd_1_1ostream.xml
%feature("docstring") std::ostream "

STL class. ";


// File: classstd_1_1ostringstream.xml
%feature("docstring") std::ostringstream "

STL class. ";


// File: classstd_1_1out__of__range.xml
%feature("docstring") std::out_of_range "

STL class. ";


// File: classstd_1_1overflow__error.xml
%feature("docstring") std::overflow_error "

STL class. ";


// File: classgdcm_1_1Overlay.xml
%feature("docstring") gdcm::Overlay "

Overlay class.

see AreOverlaysInPixelData Todo Is there actually any way to recognize
an overlay ? On images with multiple overlay I do not see any way to
differenciate them (other than the group tag).

Example:

C++ includes: gdcmOverlay.h ";

%feature("docstring")  gdcm::Overlay::Overlay "gdcm::Overlay::Overlay() ";

%feature("docstring")  gdcm::Overlay::Overlay "gdcm::Overlay::Overlay(Overlay const &ov) ";

%feature("docstring")  gdcm::Overlay::~Overlay "gdcm::Overlay::~Overlay() ";

%feature("docstring")  gdcm::Overlay::Decompress "void
gdcm::Overlay::Decompress(std::ostream &os) const

Decode the internal OverlayData (packed bits) into unpacked
representation. ";

%feature("docstring")  gdcm::Overlay::GetBitPosition "unsigned short
gdcm::Overlay::GetBitPosition() const

return bit position ";

%feature("docstring")  gdcm::Overlay::GetBitsAllocated "unsigned
short gdcm::Overlay::GetBitsAllocated() const

return bits allocated ";

%feature("docstring")  gdcm::Overlay::GetColumns "unsigned short
gdcm::Overlay::GetColumns() const

get columns ";

%feature("docstring")  gdcm::Overlay::GetDescription "const char*
gdcm::Overlay::GetDescription() const

get description ";

%feature("docstring")  gdcm::Overlay::GetGroup "unsigned short
gdcm::Overlay::GetGroup() const

Get Group number. ";

%feature("docstring")  gdcm::Overlay::GetOrigin "const signed short*
gdcm::Overlay::GetOrigin() const

get origin ";

%feature("docstring")  gdcm::Overlay::GetOverlayData "const
ByteValue& gdcm::Overlay::GetOverlayData() const

Return the Overlay Data as ByteValue: Not thread safe ";

%feature("docstring")  gdcm::Overlay::GetRows "unsigned short
gdcm::Overlay::GetRows() const

get rows ";

%feature("docstring")  gdcm::Overlay::GetType "const char*
gdcm::Overlay::GetType() const

get type ";

%feature("docstring")  gdcm::Overlay::GetTypeAsEnum "OverlayType
gdcm::Overlay::GetTypeAsEnum() const ";

%feature("docstring")  gdcm::Overlay::GetUnpackBuffer "bool
gdcm::Overlay::GetUnpackBuffer(char *buffer, size_t len) const

Retrieve the unpack buffer for Overlay. This is an error if the size
if below GetUnpackBufferLength() ";

%feature("docstring")  gdcm::Overlay::GetUnpackBufferLength "size_t
gdcm::Overlay::GetUnpackBufferLength() const

Retrieve the size of the buffer needed to hold the Overlay as
specified by Col & Row parameters ";

%feature("docstring")  gdcm::Overlay::GrabOverlayFromPixelData "bool
gdcm::Overlay::GrabOverlayFromPixelData(DataSet const &ds) ";

%feature("docstring")  gdcm::Overlay::IsEmpty "bool
gdcm::Overlay::IsEmpty() const

Return whether or not the Overlay is empty: ";

%feature("docstring")  gdcm::Overlay::IsInPixelData "bool
gdcm::Overlay::IsInPixelData() const

return if the Overlay is stored in the pixel data or not ";

%feature("docstring")  gdcm::Overlay::IsInPixelData "void
gdcm::Overlay::IsInPixelData(bool b)

Set wether or no the OverlayData is in the Pixel Data: ";

%feature("docstring")  gdcm::Overlay::IsZero "bool
gdcm::Overlay::IsZero() const

return true if all bits are set to 0 ";

%feature("docstring")  gdcm::Overlay::Print "void
gdcm::Overlay::Print(std::ostream &) const

Print. ";

%feature("docstring")  gdcm::Overlay::SetBitPosition "void
gdcm::Overlay::SetBitPosition(unsigned short bitposition)

set bit position ";

%feature("docstring")  gdcm::Overlay::SetBitsAllocated "void
gdcm::Overlay::SetBitsAllocated(unsigned short bitsallocated)

set bits allocated ";

%feature("docstring")  gdcm::Overlay::SetColumns "void
gdcm::Overlay::SetColumns(unsigned short columns)

set columns ";

%feature("docstring")  gdcm::Overlay::SetDescription "void
gdcm::Overlay::SetDescription(const char *description)

set description ";

%feature("docstring")  gdcm::Overlay::SetFrameOrigin "void
gdcm::Overlay::SetFrameOrigin(unsigned short frameorigin)

set frame origin ";

%feature("docstring")  gdcm::Overlay::SetGroup "void
gdcm::Overlay::SetGroup(unsigned short group)

Set Group number. ";

%feature("docstring")  gdcm::Overlay::SetNumberOfFrames "void
gdcm::Overlay::SetNumberOfFrames(unsigned int numberofframes)

set number of frames ";

%feature("docstring")  gdcm::Overlay::SetOrigin "void
gdcm::Overlay::SetOrigin(const signed short origin[2])

set origin ";

%feature("docstring")  gdcm::Overlay::SetOverlay "void
gdcm::Overlay::SetOverlay(const char *array, size_t length)

set overlay from byte array + length ";

%feature("docstring")  gdcm::Overlay::SetRows "void
gdcm::Overlay::SetRows(unsigned short rows)

set rows ";

%feature("docstring")  gdcm::Overlay::SetType "void
gdcm::Overlay::SetType(const char *type)

set type ";

%feature("docstring")  gdcm::Overlay::Update "void
gdcm::Overlay::Update(const DataElement &de)

Update overlay from data element de: ";


// File: classgdcm_1_1ParseException.xml
%feature("docstring") gdcm::ParseException "

ParseException Standard exception handling object.

C++ includes: gdcmParseException.h ";

%feature("docstring")  gdcm::ParseException::ParseException "gdcm::ParseException::ParseException() ";

%feature("docstring")  gdcm::ParseException::~ParseException "virtual
gdcm::ParseException::~ParseException()  throw ()";

%feature("docstring")  gdcm::ParseException::GetLastElement "const
DataElement& gdcm::ParseException::GetLastElement() const ";

%feature("docstring")  gdcm::ParseException::SetLastElement "void
gdcm::ParseException::SetLastElement(DataElement &de)

Equivalence operator. ";


// File: classgdcm_1_1Parser.xml
%feature("docstring") gdcm::Parser "

Parser ala XML_Parser from expat (SAX)

Detailled description here Simple API for DICOM

C++ includes: gdcmParser.h ";

%feature("docstring")  gdcm::Parser::Parser "gdcm::Parser::Parser()
";

%feature("docstring")  gdcm::Parser::~Parser "gdcm::Parser::~Parser()
";

%feature("docstring")  gdcm::Parser::GetCurrentByteIndex "unsigned
long gdcm::Parser::GetCurrentByteIndex() const ";

%feature("docstring")  gdcm::Parser::GetErrorCode "ErrorType
gdcm::Parser::GetErrorCode() const ";

%feature("docstring")  gdcm::Parser::GetUserData "void*
gdcm::Parser::GetUserData() const ";

%feature("docstring")  gdcm::Parser::Parse "bool
gdcm::Parser::Parse(const char *s, int len, bool isFinal) ";

%feature("docstring")  gdcm::Parser::SetElementHandler "void
gdcm::Parser::SetElementHandler(StartElementHandler start,
EndElementHandler end) ";

%feature("docstring")  gdcm::Parser::SetUserData "void
gdcm::Parser::SetUserData(void *userData) ";


// File: classgdcm_1_1Patient.xml
%feature("docstring") gdcm::Patient "

See PS 3.3 - 2007 DICOM MODEL OF THE REAL-WORLD, p 54.

C++ includes: gdcmPatient.h ";

%feature("docstring")  gdcm::Patient::Patient "gdcm::Patient::Patient() ";


// File: classgdcm_1_1network_1_1PDataTFPDU.xml
%feature("docstring") gdcm::network::PDataTFPDU "

PDataTFPDU Table 9-22 P-DATA-TF PDU FIELDS.

C++ includes: gdcmPDataTFPDU.h ";

%feature("docstring")  gdcm::network::PDataTFPDU::PDataTFPDU "gdcm::network::PDataTFPDU::PDataTFPDU() ";

%feature("docstring")
gdcm::network::PDataTFPDU::AddPresentationDataValue "void
gdcm::network::PDataTFPDU::AddPresentationDataValue(PresentationDataValue
const &pdv) ";

%feature("docstring")
gdcm::network::PDataTFPDU::GetNumberOfPresentationDataValues "SizeType
gdcm::network::PDataTFPDU::GetNumberOfPresentationDataValues() const
";

%feature("docstring")
gdcm::network::PDataTFPDU::GetPresentationDataValue "PresentationDataValue const&
gdcm::network::PDataTFPDU::GetPresentationDataValue(SizeType i) const
";

%feature("docstring")  gdcm::network::PDataTFPDU::IsLastFragment "bool gdcm::network::PDataTFPDU::IsLastFragment() const ";

%feature("docstring")  gdcm::network::PDataTFPDU::Print "void
gdcm::network::PDataTFPDU::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::network::PDataTFPDU::Read "std::istream&
gdcm::network::PDataTFPDU::Read(std::istream &is) ";

%feature("docstring")  gdcm::network::PDataTFPDU::Size "size_t
gdcm::network::PDataTFPDU::Size() const ";

%feature("docstring")  gdcm::network::PDataTFPDU::Write "const
std::ostream& gdcm::network::PDataTFPDU::Write(std::ostream &os) const
";


// File: classgdcm_1_1PDBElement.xml
%feature("docstring") gdcm::PDBElement "

Class to represent a PDB Element.

See:   PDBHeader

C++ includes: gdcmPDBElement.h ";

%feature("docstring")  gdcm::PDBElement::PDBElement "gdcm::PDBElement::PDBElement() ";

%feature("docstring")  gdcm::PDBElement::GetName "const char*
gdcm::PDBElement::GetName() const

Set/Get Name. ";

%feature("docstring")  gdcm::PDBElement::GetValue "const char*
gdcm::PDBElement::GetValue() const

Set/Get Value. ";

%feature("docstring")  gdcm::PDBElement::SetName "void
gdcm::PDBElement::SetName(const char *name) ";

%feature("docstring")  gdcm::PDBElement::SetValue "void
gdcm::PDBElement::SetValue(const char *value) ";


// File: classgdcm_1_1PDBHeader.xml
%feature("docstring") gdcm::PDBHeader "

Class for PDBHeader.

GEMS MR Image have an Attribute (0025,1b,GEMS_SERS_01) which store the
Acquisition parameter of the MR Image. It is compressed and can
therefore not be used as is. This class de- encapsulated the Protocol
Data Block and allow users to query element by name.

WARNING:  Everything you do with this code is at your own risk, since
decoding process was not written from specification documents.

: the API of this class might change.

See:   CSAHeader

C++ includes: gdcmPDBHeader.h ";

%feature("docstring")  gdcm::PDBHeader::PDBHeader "gdcm::PDBHeader::PDBHeader() ";

%feature("docstring")  gdcm::PDBHeader::~PDBHeader "gdcm::PDBHeader::~PDBHeader() ";

%feature("docstring")  gdcm::PDBHeader::FindPDBElementByName "bool
gdcm::PDBHeader::FindPDBElementByName(const char *name)

Return true if the PDB element matching name is found or not. ";

%feature("docstring")  gdcm::PDBHeader::GetPDBElementByName "const
PDBElement& gdcm::PDBHeader::GetPDBElementByName(const char *name)

Lookup in the PDB header if a PDB element match the name 'name':
WARNING:  Case Sensitive ";

%feature("docstring")  gdcm::PDBHeader::LoadFromDataElement "bool
gdcm::PDBHeader::LoadFromDataElement(DataElement const &de)

Load the PDB Header from a DataElement of a DataSet. ";

%feature("docstring")  gdcm::PDBHeader::Print "void
gdcm::PDBHeader::Print(std::ostream &os) const

Print. ";


// File: classgdcm_1_1PDFCodec.xml
%feature("docstring") gdcm::PDFCodec "

PDFCodec class.

C++ includes: gdcmPDFCodec.h ";

%feature("docstring")  gdcm::PDFCodec::PDFCodec "gdcm::PDFCodec::PDFCodec() ";

%feature("docstring")  gdcm::PDFCodec::~PDFCodec "gdcm::PDFCodec::~PDFCodec() ";

%feature("docstring")  gdcm::PDFCodec::CanCode "bool
gdcm::PDFCodec::CanCode(TransferSyntax const &) const

Return whether this coder support this transfer syntax (can code it)
";

%feature("docstring")  gdcm::PDFCodec::CanDecode "bool
gdcm::PDFCodec::CanDecode(TransferSyntax const &) const

Return whether this decoder support this transfer syntax (can decode
it) ";

%feature("docstring")  gdcm::PDFCodec::Decode "bool
gdcm::PDFCodec::Decode(DataElement const &is, DataElement &os)

Decode. ";


// File: classgdcm_1_1network_1_1PDUFactory.xml
%feature("docstring") gdcm::network::PDUFactory "

PDUFactory basically, given an initial byte, construct the appropriate
PDU. This way, the event loop doesn't have to know about all the
different PDU types.

C++ includes: gdcmPDUFactory.h ";


// File: classgdcm_1_1PersonName.xml
%feature("docstring") gdcm::PersonName "

PersonName class.

C++ includes: gdcmPersonName.h ";

%feature("docstring")  gdcm::PersonName::GetMaxLength "unsigned int
gdcm::PersonName::GetMaxLength() const ";

%feature("docstring")  gdcm::PersonName::GetNumberOfComponents "unsigned int gdcm::PersonName::GetNumberOfComponents() const ";

%feature("docstring")  gdcm::PersonName::Print "void
gdcm::PersonName::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::PersonName::SetBlob "void
gdcm::PersonName::SetBlob(const std::vector< char > &v) ";

%feature("docstring")  gdcm::PersonName::SetComponents "void
gdcm::PersonName::SetComponents(const char *comp1=\"\", const char
*comp2=\"\", const char *comp3=\"\", const char *comp4=\"\", const
char *comp5=\"\") ";

%feature("docstring")  gdcm::PersonName::SetComponents "void
gdcm::PersonName::SetComponents(const char *components[]) ";


// File: classgdcm_1_1PGXCodec.xml
%feature("docstring") gdcm::PGXCodec "

Class to do PGX See PGX as used in JPEG 2000 implementation and
reference images.

C++ includes: gdcmPGXCodec.h ";

%feature("docstring")  gdcm::PGXCodec::PGXCodec "gdcm::PGXCodec::PGXCodec() ";

%feature("docstring")  gdcm::PGXCodec::~PGXCodec "gdcm::PGXCodec::~PGXCodec() ";

%feature("docstring")  gdcm::PGXCodec::CanCode "bool
gdcm::PGXCodec::CanCode(TransferSyntax const &ts) const

Return whether this coder support this transfer syntax (can code it)
";

%feature("docstring")  gdcm::PGXCodec::CanDecode "bool
gdcm::PGXCodec::CanDecode(TransferSyntax const &ts) const

Return whether this decoder support this transfer syntax (can decode
it) ";

%feature("docstring")  gdcm::PGXCodec::Clone "virtual ImageCodec*
gdcm::PGXCodec::Clone() const ";

%feature("docstring")  gdcm::PGXCodec::GetHeaderInfo "bool
gdcm::PGXCodec::GetHeaderInfo(std::istream &is, TransferSyntax &ts) ";

%feature("docstring")  gdcm::PGXCodec::Read "bool
gdcm::PGXCodec::Read(const char *filename, DataElement &out) const ";

%feature("docstring")  gdcm::PGXCodec::Write "bool
gdcm::PGXCodec::Write(const char *filename, const DataElement &out)
const ";


// File: classgdcm_1_1PhotometricInterpretation.xml
%feature("docstring") gdcm::PhotometricInterpretation "

Class to represent an PhotometricInterpretation.

C++ includes: gdcmPhotometricInterpretation.h ";

%feature("docstring")
gdcm::PhotometricInterpretation::PhotometricInterpretation "gdcm::PhotometricInterpretation::PhotometricInterpretation(PIType
pi=UNKNOW) ";

%feature("docstring")
gdcm::PhotometricInterpretation::GetSamplesPerPixel "unsigned short
gdcm::PhotometricInterpretation::GetSamplesPerPixel() const

return the value for Sample Per Pixel associated with a particular
Photometric Interpretation ";

%feature("docstring")  gdcm::PhotometricInterpretation::GetString "const char* gdcm::PhotometricInterpretation::GetString() const ";

%feature("docstring")  gdcm::PhotometricInterpretation::GetType "PIType gdcm::PhotometricInterpretation::GetType() const ";

%feature("docstring")  gdcm::PhotometricInterpretation::IsLossless "bool gdcm::PhotometricInterpretation::IsLossless() const ";

%feature("docstring")  gdcm::PhotometricInterpretation::IsLossy "bool
gdcm::PhotometricInterpretation::IsLossy() const ";

%feature("docstring")
gdcm::PhotometricInterpretation::IsSameColorSpace "bool
gdcm::PhotometricInterpretation::IsSameColorSpace(PhotometricInterpretation
const &pi) const ";


// File: classgdcm_1_1PixelFormat.xml
%feature("docstring") gdcm::PixelFormat "

PixelFormat.

By default the Pixel Type will be instanciated with the following
parameters: SamplesPerPixel : 1

BitsAllocated : 8

BitsStored : 8

HighBit : 7

PixelRepresentation : 0

Fundamentally PixelFormat is very close to what DICOM allows. It will
be very hard to extend this class for the upcoming DICOM standard
where Floating 32 and 64bits will be allowed.

It is also very hard for this class to fully support 64bits integer
type (see GetMin / GetMax signature restricted to 64bits signed).

C++ includes: gdcmPixelFormat.h ";

%feature("docstring")  gdcm::PixelFormat::PixelFormat "gdcm::PixelFormat::PixelFormat(unsigned short samplesperpixel=1,
unsigned short bitsallocated=8, unsigned short bitsstored=8, unsigned
short highbit=7, unsigned short pixelrepresentation=0) ";

%feature("docstring")  gdcm::PixelFormat::PixelFormat "gdcm::PixelFormat::PixelFormat(ScalarType st) ";

%feature("docstring")  gdcm::PixelFormat::GetBitsAllocated "unsigned
short gdcm::PixelFormat::GetBitsAllocated() const

BitsAllocated see Tag (0028,0100) US Bits Allocated. ";

%feature("docstring")  gdcm::PixelFormat::GetBitsStored "unsigned
short gdcm::PixelFormat::GetBitsStored() const

BitsStored see Tag (0028,0101) US Bits Stored. ";

%feature("docstring")  gdcm::PixelFormat::GetHighBit "unsigned short
gdcm::PixelFormat::GetHighBit() const

HighBit see Tag (0028,0102) US High Bit. ";

%feature("docstring")  gdcm::PixelFormat::GetMax "int64_t
gdcm::PixelFormat::GetMax() const

return the max possible of the pixel ";

%feature("docstring")  gdcm::PixelFormat::GetMin "int64_t
gdcm::PixelFormat::GetMin() const

return the min possible of the pixel ";

%feature("docstring")  gdcm::PixelFormat::GetPixelRepresentation "unsigned short gdcm::PixelFormat::GetPixelRepresentation() const

PixelRepresentation: 0 or 1, see Tag (0028,0103) US Pixel
Representation. ";

%feature("docstring")  gdcm::PixelFormat::GetPixelSize "uint8_t
gdcm::PixelFormat::GetPixelSize() const

return the size of the pixel This is the number of words it would take
to store one pixel WARNING:  the return value takes into account the
SamplesPerPixel

in the rare case when BitsAllocated == 12, the function assume word
padding and value returned will be identical as if BitsAllocated == 16
";

%feature("docstring")  gdcm::PixelFormat::GetSamplesPerPixel "unsigned short gdcm::PixelFormat::GetSamplesPerPixel() const

Samples Per Pixel see (0028,0002) US Samples Per Pixel DICOM - only
allows 1, 3 and 4 as valid value. Other value are undefined behavior.
";

%feature("docstring")  gdcm::PixelFormat::GetScalarType "ScalarType
gdcm::PixelFormat::GetScalarType() const

ScalarType does not take into account the sample per pixel. ";

%feature("docstring")  gdcm::PixelFormat::GetScalarTypeAsString "const char* gdcm::PixelFormat::GetScalarTypeAsString() const ";

%feature("docstring")  gdcm::PixelFormat::IsCompatible "bool
gdcm::PixelFormat::IsCompatible(const TransferSyntax &ts) const ";

%feature("docstring")  gdcm::PixelFormat::IsValid "bool
gdcm::PixelFormat::IsValid() const

return IsValid ";

%feature("docstring")  gdcm::PixelFormat::Print "void
gdcm::PixelFormat::Print(std::ostream &os) const

Print. ";

%feature("docstring")  gdcm::PixelFormat::SetBitsAllocated "void
gdcm::PixelFormat::SetBitsAllocated(unsigned short ba) ";

%feature("docstring")  gdcm::PixelFormat::SetBitsStored "void
gdcm::PixelFormat::SetBitsStored(unsigned short bs) ";

%feature("docstring")  gdcm::PixelFormat::SetHighBit "void
gdcm::PixelFormat::SetHighBit(unsigned short hb) ";

%feature("docstring")  gdcm::PixelFormat::SetPixelRepresentation "void gdcm::PixelFormat::SetPixelRepresentation(unsigned short pr) ";

%feature("docstring")  gdcm::PixelFormat::SetSamplesPerPixel "void
gdcm::PixelFormat::SetSamplesPerPixel(unsigned short spp) ";

%feature("docstring")  gdcm::PixelFormat::SetScalarType "void
gdcm::PixelFormat::SetScalarType(ScalarType st)

Set PixelFormat based only on the ScalarType WARNING:  : You need to
call SetScalarType before SetSamplesPerPixel ";


// File: classgdcm_1_1Pixmap.xml
%feature("docstring") gdcm::Pixmap "

Pixmap class A bitmap based image. Used as parent for both IconImage
and the main Pixel Data Image It does not contains any World Space
information (IPP, IOP)

See:   PixmapReader

C++ includes: gdcmPixmap.h ";

%feature("docstring")  gdcm::Pixmap::Pixmap "gdcm::Pixmap::Pixmap()
";

%feature("docstring")  gdcm::Pixmap::~Pixmap "gdcm::Pixmap::~Pixmap()
";

%feature("docstring")  gdcm::Pixmap::AreOverlaysInPixelData "bool
gdcm::Pixmap::AreOverlaysInPixelData() const

returns if Overlays are stored in the unused bit of the pixel data: ";

%feature("docstring")  gdcm::Pixmap::GetCurve "Curve&
gdcm::Pixmap::GetCurve(size_t i=0)

Curve: group 50xx. ";

%feature("docstring")  gdcm::Pixmap::GetCurve "const Curve&
gdcm::Pixmap::GetCurve(size_t i=0) const ";

%feature("docstring")  gdcm::Pixmap::GetIconImage "const IconImage&
gdcm::Pixmap::GetIconImage() const

Set/Get Icon Image. ";

%feature("docstring")  gdcm::Pixmap::GetIconImage "IconImage&
gdcm::Pixmap::GetIconImage() ";

%feature("docstring")  gdcm::Pixmap::GetNumberOfCurves "size_t
gdcm::Pixmap::GetNumberOfCurves() const ";

%feature("docstring")  gdcm::Pixmap::GetNumberOfOverlays "size_t
gdcm::Pixmap::GetNumberOfOverlays() const ";

%feature("docstring")  gdcm::Pixmap::GetOverlay "Overlay&
gdcm::Pixmap::GetOverlay(size_t i=0)

Overlay: group 60xx. ";

%feature("docstring")  gdcm::Pixmap::GetOverlay "const Overlay&
gdcm::Pixmap::GetOverlay(size_t i=0) const ";

%feature("docstring")  gdcm::Pixmap::Print "void
gdcm::Pixmap::Print(std::ostream &) const ";

%feature("docstring")  gdcm::Pixmap::RemoveOverlay "void
gdcm::Pixmap::RemoveOverlay(size_t i) ";

%feature("docstring")  gdcm::Pixmap::SetIconImage "void
gdcm::Pixmap::SetIconImage(IconImage const &ii) ";

%feature("docstring")  gdcm::Pixmap::SetNumberOfCurves "void
gdcm::Pixmap::SetNumberOfCurves(size_t n) ";

%feature("docstring")  gdcm::Pixmap::SetNumberOfOverlays "void
gdcm::Pixmap::SetNumberOfOverlays(size_t n) ";


// File: classgdcm_1_1PixmapReader.xml
%feature("docstring") gdcm::PixmapReader "

PixmapReader.

its role is to convert the DICOM DataSet into a Pixmap representation
By default it is also loading the lookup table and overlay when found
as they impact the rendering or the image  See PS 3.3-2008, Table
C.7-11b IMAGE PIXEL MACRO ATTRIBUTES for the list of attribute that
belong to what gdcm calls a ' Pixmap'

WARNING:  the API ReadUpToTag and ReadSelectedTag

See:   Pixmap

C++ includes: gdcmPixmapReader.h ";

%feature("docstring")  gdcm::PixmapReader::PixmapReader "gdcm::PixmapReader::PixmapReader() ";

%feature("docstring")  gdcm::PixmapReader::~PixmapReader "virtual
gdcm::PixmapReader::~PixmapReader() ";

%feature("docstring")  gdcm::PixmapReader::GetPixmap "const Pixmap&
gdcm::PixmapReader::GetPixmap() const

Return the read image (need to call Read() first) ";

%feature("docstring")  gdcm::PixmapReader::GetPixmap "Pixmap&
gdcm::PixmapReader::GetPixmap() ";

%feature("docstring")  gdcm::PixmapReader::Read "virtual bool
gdcm::PixmapReader::Read()

Read the DICOM image. There are two reason for failure: The input
filename is not DICOM

The input DICOM file does not contains an Pixmap. ";


// File: classgdcm_1_1PixmapToPixmapFilter.xml
%feature("docstring") gdcm::PixmapToPixmapFilter "

PixmapToPixmapFilter class Super class for all filter taking an image
and producing an output image.

C++ includes: gdcmPixmapToPixmapFilter.h ";

%feature("docstring")
gdcm::PixmapToPixmapFilter::PixmapToPixmapFilter "gdcm::PixmapToPixmapFilter::PixmapToPixmapFilter() ";

%feature("docstring")
gdcm::PixmapToPixmapFilter::~PixmapToPixmapFilter "gdcm::PixmapToPixmapFilter::~PixmapToPixmapFilter() ";

%feature("docstring")  gdcm::PixmapToPixmapFilter::GetInput "Pixmap&
gdcm::PixmapToPixmapFilter::GetInput() ";

%feature("docstring")  gdcm::PixmapToPixmapFilter::GetOutput "const
Pixmap& gdcm::PixmapToPixmapFilter::GetOutput() const

Get Output image. ";

%feature("docstring")  gdcm::PixmapToPixmapFilter::GetOutputAsPixmap "const Pixmap& gdcm::PixmapToPixmapFilter::GetOutputAsPixmap() const ";


// File: classgdcm_1_1PixmapWriter.xml
%feature("docstring") gdcm::PixmapWriter "

PixmapWriter This class will takes two inputs:

The DICOM DataSet

The Image input It will override any info from the Image over the
DataSet.

For instance when one read in a lossy compressed image and write out
as unencapsulated (ie implicitely lossless) then some attribute are
definitely needed to mark this dataset as Lossy (typically 0028,2114)

C++ includes: gdcmPixmapWriter.h ";

%feature("docstring")  gdcm::PixmapWriter::PixmapWriter "gdcm::PixmapWriter::PixmapWriter() ";

%feature("docstring")  gdcm::PixmapWriter::~PixmapWriter "gdcm::PixmapWriter::~PixmapWriter() ";

%feature("docstring")  gdcm::PixmapWriter::GetImage "virtual const
Pixmap& gdcm::PixmapWriter::GetImage() const

Set/Get Pixmap to be written It will overwrite anything Pixmap infos
found in DataSet (see parent class to see how to pass dataset) ";

%feature("docstring")  gdcm::PixmapWriter::GetImage "virtual Pixmap&
gdcm::PixmapWriter::GetImage() ";

%feature("docstring")  gdcm::PixmapWriter::GetPixmap "const Pixmap&
gdcm::PixmapWriter::GetPixmap() const ";

%feature("docstring")  gdcm::PixmapWriter::GetPixmap "Pixmap&
gdcm::PixmapWriter::GetPixmap() ";

%feature("docstring")  gdcm::PixmapWriter::SetImage "virtual void
gdcm::PixmapWriter::SetImage(Pixmap const &img) ";

%feature("docstring")  gdcm::PixmapWriter::SetPixmap "void
gdcm::PixmapWriter::SetPixmap(Pixmap const &img) ";

%feature("docstring")  gdcm::PixmapWriter::Write "bool
gdcm::PixmapWriter::Write()

Write. ";


// File: classgdcm_1_1PNMCodec.xml
%feature("docstring") gdcm::PNMCodec "

Class to do PNM PNM is the Portable anymap file format. The main web
page can be found at:http://netpbm.sourceforge.net/.

Only support P5 & P6 PNM file (binary grayscale and binary rgb)

C++ includes: gdcmPNMCodec.h ";

%feature("docstring")  gdcm::PNMCodec::PNMCodec "gdcm::PNMCodec::PNMCodec() ";

%feature("docstring")  gdcm::PNMCodec::~PNMCodec "gdcm::PNMCodec::~PNMCodec() ";

%feature("docstring")  gdcm::PNMCodec::CanCode "bool
gdcm::PNMCodec::CanCode(TransferSyntax const &ts) const

Return whether this coder support this transfer syntax (can code it)
";

%feature("docstring")  gdcm::PNMCodec::CanDecode "bool
gdcm::PNMCodec::CanDecode(TransferSyntax const &ts) const

Return whether this decoder support this transfer syntax (can decode
it) ";

%feature("docstring")  gdcm::PNMCodec::Clone "virtual ImageCodec*
gdcm::PNMCodec::Clone() const ";

%feature("docstring")  gdcm::PNMCodec::GetBufferLength "unsigned long
gdcm::PNMCodec::GetBufferLength() const ";

%feature("docstring")  gdcm::PNMCodec::GetHeaderInfo "bool
gdcm::PNMCodec::GetHeaderInfo(std::istream &is, TransferSyntax &ts) ";

%feature("docstring")  gdcm::PNMCodec::Read "bool
gdcm::PNMCodec::Read(const char *filename, DataElement &out) const ";

%feature("docstring")  gdcm::PNMCodec::SetBufferLength "void
gdcm::PNMCodec::SetBufferLength(unsigned long l) ";

%feature("docstring")  gdcm::PNMCodec::Write "bool
gdcm::PNMCodec::Write(const char *filename, const DataElement &out)
const ";


// File: classgdcm_1_1Preamble.xml
%feature("docstring") gdcm::Preamble "

DICOM Preamble (Part 10)

C++ includes: gdcmPreamble.h ";

%feature("docstring")  gdcm::Preamble::Preamble "gdcm::Preamble::Preamble() ";

%feature("docstring")  gdcm::Preamble::Preamble "gdcm::Preamble::Preamble(Preamble const &) ";

%feature("docstring")  gdcm::Preamble::~Preamble "gdcm::Preamble::~Preamble() ";

%feature("docstring")  gdcm::Preamble::Clear "void
gdcm::Preamble::Clear() ";

%feature("docstring")  gdcm::Preamble::Create "void
gdcm::Preamble::Create() ";

%feature("docstring")  gdcm::Preamble::GetInternal "const char*
gdcm::Preamble::GetInternal() const ";

%feature("docstring")  gdcm::Preamble::GetLength "VL
gdcm::Preamble::GetLength() const ";

%feature("docstring")  gdcm::Preamble::IsEmpty "bool
gdcm::Preamble::IsEmpty() const ";

%feature("docstring")  gdcm::Preamble::Print "void
gdcm::Preamble::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::Preamble::Read "std::istream&
gdcm::Preamble::Read(std::istream &is) ";

%feature("docstring")  gdcm::Preamble::Remove "void
gdcm::Preamble::Remove() ";

%feature("docstring")  gdcm::Preamble::Valid "void
gdcm::Preamble::Valid() ";

%feature("docstring")  gdcm::Preamble::Write "std::ostream const&
gdcm::Preamble::Write(std::ostream &os) const ";


// File: classgdcm_1_1PresentationContext.xml
%feature("docstring") gdcm::PresentationContext "

PresentationContext.

See:  PresentationContextAC PresentationContextRQ

C++ includes: gdcmPresentationContext.h ";

%feature("docstring")  gdcm::PresentationContext::PresentationContext
"gdcm::PresentationContext::PresentationContext() ";

%feature("docstring")  gdcm::PresentationContext::PresentationContext
"gdcm::PresentationContext::PresentationContext(UIDs::TSName asname,
UIDs::TSName
tsname=UIDs::ImplicitVRLittleEndianDefaultTransferSyntaxforDICOM)

Initialize Presentation Context with AbstractSyntax set to asname and
with a single TransferSyntax set to tsname (default to Implicit VR
LittleEndian when not specified ). ";

%feature("docstring")  gdcm::PresentationContext::AddTransferSyntax "void gdcm::PresentationContext::AddTransferSyntax(const char *tsstr)
";

%feature("docstring")  gdcm::PresentationContext::GetAbstractSyntax "const char* gdcm::PresentationContext::GetAbstractSyntax() const ";

%feature("docstring")
gdcm::PresentationContext::GetNumberOfTransferSyntaxes "SizeType
gdcm::PresentationContext::GetNumberOfTransferSyntaxes() const ";

%feature("docstring")
gdcm::PresentationContext::GetPresentationContextID "uint8_t
gdcm::PresentationContext::GetPresentationContextID() const ";

%feature("docstring")  gdcm::PresentationContext::GetTransferSyntax "const char* gdcm::PresentationContext::GetTransferSyntax(SizeType i)
const ";

%feature("docstring")  gdcm::PresentationContext::Print "void
gdcm::PresentationContext::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::PresentationContext::SetAbstractSyntax "void gdcm::PresentationContext::SetAbstractSyntax(const char *as) ";

%feature("docstring")
gdcm::PresentationContext::SetPresentationContextID "void
gdcm::PresentationContext::SetPresentationContextID(uint8_t id) ";


// File: classgdcm_1_1network_1_1PresentationContextAC.xml
%feature("docstring") gdcm::network::PresentationContextAC "

PresentationContextAC Table 9-18 PRESENTATION CONTEXT ITEM FIELDS.

See:   PresentationContext

C++ includes: gdcmPresentationContextAC.h ";

%feature("docstring")
gdcm::network::PresentationContextAC::PresentationContextAC "gdcm::network::PresentationContextAC::PresentationContextAC() ";

%feature("docstring")
gdcm::network::PresentationContextAC::GetPresentationContextID "uint8_t
gdcm::network::PresentationContextAC::GetPresentationContextID() const
";

%feature("docstring")  gdcm::network::PresentationContextAC::GetReason
"uint8_t gdcm::network::PresentationContextAC::GetReason() const ";

%feature("docstring")
gdcm::network::PresentationContextAC::GetTransferSyntax "TransferSyntaxSub const&
gdcm::network::PresentationContextAC::GetTransferSyntax() const ";

%feature("docstring")  gdcm::network::PresentationContextAC::Print "void gdcm::network::PresentationContextAC::Print(std::ostream &os)
const ";

%feature("docstring")  gdcm::network::PresentationContextAC::Read "std::istream& gdcm::network::PresentationContextAC::Read(std::istream
&is) ";

%feature("docstring")
gdcm::network::PresentationContextAC::SetPresentationContextID "void
gdcm::network::PresentationContextAC::SetPresentationContextID(uint8_t
id) ";

%feature("docstring")  gdcm::network::PresentationContextAC::SetReason
"void gdcm::network::PresentationContextAC::SetReason(uint8_t r) ";

%feature("docstring")
gdcm::network::PresentationContextAC::SetTransferSyntax "void
gdcm::network::PresentationContextAC::SetTransferSyntax(TransferSyntaxSub
const &ts) ";

%feature("docstring")  gdcm::network::PresentationContextAC::Size "size_t gdcm::network::PresentationContextAC::Size() const ";

%feature("docstring")  gdcm::network::PresentationContextAC::Write "const std::ostream&
gdcm::network::PresentationContextAC::Write(std::ostream &os) const ";


// File: classgdcm_1_1PresentationContextGenerator.xml
%feature("docstring") gdcm::PresentationContextGenerator "

PresentationContextGenerator This class is responsible for generating
the proper PresentationContext that will be used in subsequent
operation during a DICOM Query/Retrieve association. The step of the
association is very sensible as special care need to be taken to
explicitly define what instance are going to be send and how they are
encoded.

For example a PresentationContext will express that negotiation
requires that CT Image Storage are send using JPEG Lossless, while US
Image Storage are sent using RLE Transfer Syntax.

Two very different API are exposed one which will always default to
little endian transfer syntax see GenerateFromUID() This API is used
for C-ECHO, C-FIND and C-MOVE (SCU). Another API:
GenerateFromFilenames() is used for C-STORE (SCU) as it will loop over
all filenames argument to detect the actual encoding. and therefore
find the proper encoding to be used.

Two modes are available. The default mode
(SetMergeModeToAbstractSyntax) append PresentationContext (one
AbstractSyntax and one TransferSyntax), as long a they are different.
Eg MR Image Storage/JPEG2000 and MR Image Storage/JPEGLossless would
be considered different. the other mode SetMergeModeToTransferSyntax
merge any new TransferSyntax to the already existing
PresentationContext in order to re-use the same AbstractSyntax.

See:   PresentationContext

C++ includes: gdcmPresentationContextGenerator.h ";

%feature("docstring")
gdcm::PresentationContextGenerator::PresentationContextGenerator "gdcm::PresentationContextGenerator::PresentationContextGenerator() ";

%feature("docstring")
gdcm::PresentationContextGenerator::GenerateFromFilenames "bool
gdcm::PresentationContextGenerator::GenerateFromFilenames(const
Directory::FilenamesType &files)

Generate the PresentationContext array from a File-Set. File specified
needs to be valid DICOM files. Used for C-STORE operations ";

%feature("docstring")
gdcm::PresentationContextGenerator::GenerateFromUID "bool
gdcm::PresentationContextGenerator::GenerateFromUID(UIDs::TSName
asname)

Generate the PresentationContext array from a UID (eg.
VerificationSOPClass) ";

%feature("docstring")
gdcm::PresentationContextGenerator::GetPresentationContexts "PresentationContextArrayType const&
gdcm::PresentationContextGenerator::GetPresentationContexts() ";

%feature("docstring")
gdcm::PresentationContextGenerator::SetDefaultTransferSyntax "void
gdcm::PresentationContextGenerator::SetDefaultTransferSyntax(const
TransferSyntax &ts)

Not implemented for now. GDCM internally uses Implicit Little Endian.
";

%feature("docstring")
gdcm::PresentationContextGenerator::SetMergeModeToAbstractSyntax "void
gdcm::PresentationContextGenerator::SetMergeModeToAbstractSyntax() ";

%feature("docstring")
gdcm::PresentationContextGenerator::SetMergeModeToTransferSyntax "void
gdcm::PresentationContextGenerator::SetMergeModeToTransferSyntax() ";


// File: classgdcm_1_1network_1_1PresentationContextRQ.xml
%feature("docstring") gdcm::network::PresentationContextRQ "

PresentationContextRQ Table 9-13 PRESENTATION CONTEXT ITEM FIELDS.

See:   PresentationContextAC

C++ includes: gdcmPresentationContextRQ.h ";

%feature("docstring")
gdcm::network::PresentationContextRQ::PresentationContextRQ "gdcm::network::PresentationContextRQ::PresentationContextRQ() ";

%feature("docstring")
gdcm::network::PresentationContextRQ::PresentationContextRQ "gdcm::network::PresentationContextRQ::PresentationContextRQ(UIDs::TSName
asname, UIDs::TSName
tsname=UIDs::ImplicitVRLittleEndianDefaultTransferSyntaxforDICOM)

Initialize Presentation Context with AbstractSyntax set to asname and
with a single TransferSyntax set to tsname (dfault to Implicit VR
LittleEndian when not specified ). ";

%feature("docstring")
gdcm::network::PresentationContextRQ::PresentationContextRQ "gdcm::network::PresentationContextRQ::PresentationContextRQ(const
PresentationContext &pc) ";

%feature("docstring")
gdcm::network::PresentationContextRQ::AddTransferSyntax "void
gdcm::network::PresentationContextRQ::AddTransferSyntax(TransferSyntaxSub
const &ts) ";

%feature("docstring")
gdcm::network::PresentationContextRQ::GetAbstractSyntax "AbstractSyntax const&
gdcm::network::PresentationContextRQ::GetAbstractSyntax() const ";

%feature("docstring")
gdcm::network::PresentationContextRQ::GetAbstractSyntax "AbstractSyntax&
gdcm::network::PresentationContextRQ::GetAbstractSyntax() ";

%feature("docstring")
gdcm::network::PresentationContextRQ::GetNumberOfTransferSyntaxes "SizeType
gdcm::network::PresentationContextRQ::GetNumberOfTransferSyntaxes()
const ";

%feature("docstring")
gdcm::network::PresentationContextRQ::GetPresentationContextID "uint8_t
gdcm::network::PresentationContextRQ::GetPresentationContextID() const
";

%feature("docstring")
gdcm::network::PresentationContextRQ::GetTransferSyntax "TransferSyntaxSub const&
gdcm::network::PresentationContextRQ::GetTransferSyntax(SizeType i)
const ";

%feature("docstring")
gdcm::network::PresentationContextRQ::GetTransferSyntax "TransferSyntaxSub&
gdcm::network::PresentationContextRQ::GetTransferSyntax(SizeType i) ";

%feature("docstring")
gdcm::network::PresentationContextRQ::GetTransferSyntaxes "std::vector<TransferSyntaxSub> const&
gdcm::network::PresentationContextRQ::GetTransferSyntaxes() const ";

%feature("docstring")  gdcm::network::PresentationContextRQ::Print "void gdcm::network::PresentationContextRQ::Print(std::ostream &os)
const ";

%feature("docstring")  gdcm::network::PresentationContextRQ::Read "std::istream& gdcm::network::PresentationContextRQ::Read(std::istream
&is) ";

%feature("docstring")
gdcm::network::PresentationContextRQ::SetAbstractSyntax "void
gdcm::network::PresentationContextRQ::SetAbstractSyntax(AbstractSyntax
const &as) ";

%feature("docstring")
gdcm::network::PresentationContextRQ::SetPresentationContextID "void
gdcm::network::PresentationContextRQ::SetPresentationContextID(uint8_t
id) ";

%feature("docstring")  gdcm::network::PresentationContextRQ::Size "size_t gdcm::network::PresentationContextRQ::Size() const ";

%feature("docstring")  gdcm::network::PresentationContextRQ::Write "const std::ostream&
gdcm::network::PresentationContextRQ::Write(std::ostream &os) const ";


// File: classgdcm_1_1network_1_1PresentationDataValue.xml
%feature("docstring") gdcm::network::PresentationDataValue "

PresentationDataValue Table 9-23 PRESENTATION-DATA-VALUE ITEM FIELDS.

C++ includes: gdcmPresentationDataValue.h ";

%feature("docstring")
gdcm::network::PresentationDataValue::PresentationDataValue "gdcm::network::PresentationDataValue::PresentationDataValue() ";

%feature("docstring")  gdcm::network::PresentationDataValue::GetBlob "const std::string& gdcm::network::PresentationDataValue::GetBlob()
const ";

%feature("docstring")
gdcm::network::PresentationDataValue::GetIsCommand "bool
gdcm::network::PresentationDataValue::GetIsCommand() const ";

%feature("docstring")
gdcm::network::PresentationDataValue::GetIsLastFragment "bool
gdcm::network::PresentationDataValue::GetIsLastFragment() const ";

%feature("docstring")
gdcm::network::PresentationDataValue::GetMessageHeader "uint8_t
gdcm::network::PresentationDataValue::GetMessageHeader() const ";

%feature("docstring")
gdcm::network::PresentationDataValue::GetPresentationContextID "uint8_t
gdcm::network::PresentationDataValue::GetPresentationContextID() const
";

%feature("docstring")  gdcm::network::PresentationDataValue::Print "void gdcm::network::PresentationDataValue::Print(std::ostream &os)
const ";

%feature("docstring")  gdcm::network::PresentationDataValue::Read "std::istream& gdcm::network::PresentationDataValue::Read(std::istream
&is) ";

%feature("docstring")  gdcm::network::PresentationDataValue::ReadInto
"std::istream&
gdcm::network::PresentationDataValue::ReadInto(std::istream &is,
std::ostream &os) ";

%feature("docstring")  gdcm::network::PresentationDataValue::SetBlob "void gdcm::network::PresentationDataValue::SetBlob(const std::string
&partialblob) ";

%feature("docstring")
gdcm::network::PresentationDataValue::SetCommand "void
gdcm::network::PresentationDataValue::SetCommand(bool inCommand) ";

%feature("docstring")
gdcm::network::PresentationDataValue::SetDataSet "void
gdcm::network::PresentationDataValue::SetDataSet(const DataSet &ds)

Set DataSet. Write DataSet in implicit. WARNING:  size of dataset
should be below maxpdusize ";

%feature("docstring")
gdcm::network::PresentationDataValue::SetLastFragment "void
gdcm::network::PresentationDataValue::SetLastFragment(bool inLast) ";

%feature("docstring")
gdcm::network::PresentationDataValue::SetMessageHeader "void
gdcm::network::PresentationDataValue::SetMessageHeader(uint8_t
messageheader) ";

%feature("docstring")
gdcm::network::PresentationDataValue::SetPresentationContextID "void
gdcm::network::PresentationDataValue::SetPresentationContextID(uint8_t
id) ";

%feature("docstring")  gdcm::network::PresentationDataValue::Size "size_t gdcm::network::PresentationDataValue::Size() const ";

%feature("docstring")  gdcm::network::PresentationDataValue::Write "const std::ostream&
gdcm::network::PresentationDataValue::Write(std::ostream &os) const ";


// File: classgdcm_1_1Printer.xml
%feature("docstring") gdcm::Printer "

Printer class.

C++ includes: gdcmPrinter.h ";

%feature("docstring")  gdcm::Printer::Printer "gdcm::Printer::Printer() ";

%feature("docstring")  gdcm::Printer::~Printer "gdcm::Printer::~Printer() ";

%feature("docstring")  gdcm::Printer::GetPrintStyle "PrintStyles
gdcm::Printer::GetPrintStyle() const

Get PrintStyle value. ";

%feature("docstring")  gdcm::Printer::Print "void
gdcm::Printer::Print(std::ostream &os)

Print. ";

%feature("docstring")  gdcm::Printer::PrintDataSet "void
gdcm::Printer::PrintDataSet(const DataSet &ds, std::ostream &os, const
std::string &s=\"\")

Print an individual dataset. ";

%feature("docstring")  gdcm::Printer::SetColor "void
gdcm::Printer::SetColor(bool c)

Set color mode or not. ";

%feature("docstring")  gdcm::Printer::SetFile "void
gdcm::Printer::SetFile(File const &f)

Set file. ";

%feature("docstring")  gdcm::Printer::SetStyle "void
gdcm::Printer::SetStyle(PrintStyles ps)

Set PrintStyle value. ";


// File: classstd_1_1priority__queue.xml
%feature("docstring") std::priority_queue "

STL class. ";


// File: classgdcm_1_1PrivateDict.xml
%feature("docstring") gdcm::PrivateDict "

Private Dict.

C++ includes: gdcmDict.h ";

%feature("docstring")  gdcm::PrivateDict::PrivateDict "gdcm::PrivateDict::PrivateDict() ";

%feature("docstring")  gdcm::PrivateDict::~PrivateDict "gdcm::PrivateDict::~PrivateDict() ";

%feature("docstring")  gdcm::PrivateDict::AddDictEntry "void
gdcm::PrivateDict::AddDictEntry(const PrivateTag &tag, const DictEntry
&de) ";

%feature("docstring")  gdcm::PrivateDict::FindDictEntry "bool
gdcm::PrivateDict::FindDictEntry(const PrivateTag &tag) const ";

%feature("docstring")  gdcm::PrivateDict::GetDictEntry "const
DictEntry& gdcm::PrivateDict::GetDictEntry(const PrivateTag &tag)
const ";

%feature("docstring")  gdcm::PrivateDict::IsEmpty "bool
gdcm::PrivateDict::IsEmpty() const ";

%feature("docstring")  gdcm::PrivateDict::PrintXML "void
gdcm::PrivateDict::PrintXML() const ";

%feature("docstring")  gdcm::PrivateDict::RemoveDictEntry "bool
gdcm::PrivateDict::RemoveDictEntry(const PrivateTag &tag)

Remove entry 'tag'. Return true on success (element was found and
remove). return false if element was not found. ";


// File: classgdcm_1_1PrivateTag.xml
%feature("docstring") gdcm::PrivateTag "

Class to represent a Private DICOM Data Element ( Attribute) Tag
(Group, Element, Owner)

private tag have element value in: [0x10,0xff], for instance
0x0009,0x0000 is NOT a private tag

C++ includes: gdcmPrivateTag.h ";

%feature("docstring")  gdcm::PrivateTag::PrivateTag "gdcm::PrivateTag::PrivateTag(uint16_t group=0, uint16_t element=0,
const char *owner=\"\") ";

%feature("docstring")  gdcm::PrivateTag::PrivateTag "gdcm::PrivateTag::PrivateTag(Tag const &t, const char *owner=\"\") ";

%feature("docstring")  gdcm::PrivateTag::GetAsDataElement "DataElement gdcm::PrivateTag::GetAsDataElement() const ";

%feature("docstring")  gdcm::PrivateTag::GetOwner "const char*
gdcm::PrivateTag::GetOwner() const ";

%feature("docstring")  gdcm::PrivateTag::ReadFromCommaSeparatedString
"bool gdcm::PrivateTag::ReadFromCommaSeparatedString(const char *str)

Read PrivateTag from a string. Element number will be truncated to
8bits. Eg: \"1234,5678,GDCM\" is private tag: (1234,78,\"GDCM\") ";

%feature("docstring")  gdcm::PrivateTag::SetOwner "void
gdcm::PrivateTag::SetOwner(const char *owner) ";


// File: classgdcm_1_1ProgressEvent.xml
%feature("docstring") gdcm::ProgressEvent "

ProgressEvent Special type of event triggered during.

See:   AnyEvent

C++ includes: gdcmProgressEvent.h ";

%feature("docstring")  gdcm::ProgressEvent::ProgressEvent "gdcm::ProgressEvent::ProgressEvent(double p=0) ";

%feature("docstring")  gdcm::ProgressEvent::ProgressEvent "gdcm::ProgressEvent::ProgressEvent(const Self &s) ";

%feature("docstring")  gdcm::ProgressEvent::~ProgressEvent "virtual
gdcm::ProgressEvent::~ProgressEvent() ";

%feature("docstring")  gdcm::ProgressEvent::CheckEvent "virtual bool
gdcm::ProgressEvent::CheckEvent(const ::gdcm::Event *e) const ";

%feature("docstring")  gdcm::ProgressEvent::GetEventName "virtual
const char* gdcm::ProgressEvent::GetEventName() const

Return the StringName associated with the event. ";

%feature("docstring")  gdcm::ProgressEvent::GetProgress "double
gdcm::ProgressEvent::GetProgress() const ";

%feature("docstring")  gdcm::ProgressEvent::MakeObject "virtual
::gdcm::Event* gdcm::ProgressEvent::MakeObject() const

Create an Event of this type This method work as a Factory for
creating events of each particular type. ";

%feature("docstring")  gdcm::ProgressEvent::SetProgress "void
gdcm::ProgressEvent::SetProgress(double p) ";


// File: classgdcm_1_1PVRGCodec.xml
%feature("docstring") gdcm::PVRGCodec "

PVRGCodec.

pvrg is a broken implementation of the JPEG standard. It is known to
have a bug in the 16bits lossless implementation of the standard.  In
an ideal world, you should not need this codec at all. But to support
some broken file such as:

PHILIPS_Gyroscan-12-Jpeg_Extended_Process_2_4.dcm

we have to...

C++ includes: gdcmPVRGCodec.h ";

%feature("docstring")  gdcm::PVRGCodec::PVRGCodec "gdcm::PVRGCodec::PVRGCodec() ";

%feature("docstring")  gdcm::PVRGCodec::~PVRGCodec "gdcm::PVRGCodec::~PVRGCodec() ";

%feature("docstring")  gdcm::PVRGCodec::CanCode "bool
gdcm::PVRGCodec::CanCode(TransferSyntax const &ts) const

Return whether this coder support this transfer syntax (can code it)
";

%feature("docstring")  gdcm::PVRGCodec::CanDecode "bool
gdcm::PVRGCodec::CanDecode(TransferSyntax const &ts) const

Return whether this decoder support this transfer syntax (can decode
it) ";

%feature("docstring")  gdcm::PVRGCodec::Clone "virtual ImageCodec*
gdcm::PVRGCodec::Clone() const ";

%feature("docstring")  gdcm::PVRGCodec::Code "bool
gdcm::PVRGCodec::Code(DataElement const &in, DataElement &out)

Code. ";

%feature("docstring")  gdcm::PVRGCodec::Decode "bool
gdcm::PVRGCodec::Decode(DataElement const &is, DataElement &os)

Decode. ";

%feature("docstring")  gdcm::PVRGCodec::SetLossyFlag "void
gdcm::PVRGCodec::SetLossyFlag(bool l) ";


// File: classgdcm_1_1PythonFilter.xml
%feature("docstring") gdcm::PythonFilter "

PythonFilter PythonFilter is the class that make gdcm2.x looks more
like gdcm1 and transform the binary blob contained in a DataElement
into a string, typically this is a nice feature to have for wrapped
language.

C++ includes: gdcmPythonFilter.h ";

%feature("docstring")  gdcm::PythonFilter::PythonFilter "gdcm::PythonFilter::PythonFilter() ";

%feature("docstring")  gdcm::PythonFilter::~PythonFilter "gdcm::PythonFilter::~PythonFilter() ";

%feature("docstring")  gdcm::PythonFilter::GetFile "File&
gdcm::PythonFilter::GetFile() ";

%feature("docstring")  gdcm::PythonFilter::GetFile "const File&
gdcm::PythonFilter::GetFile() const ";

%feature("docstring")  gdcm::PythonFilter::SetDicts "void
gdcm::PythonFilter::SetDicts(const Dicts &dicts) ";

%feature("docstring")  gdcm::PythonFilter::SetFile "void
gdcm::PythonFilter::SetFile(const File &f) ";

%feature("docstring")  gdcm::PythonFilter::ToPyObject "PyObject*
gdcm::PythonFilter::ToPyObject(const Tag &t) const ";

%feature("docstring")  gdcm::PythonFilter::UseDictAlways "void
gdcm::PythonFilter::UseDictAlways(bool use) ";


// File: classgdcm_1_1QueryBase.xml
%feature("docstring") gdcm::QueryBase "

QueryBase contains: the base class for constructing a query dataset
for a C-FIND and a C-MOVE.

There are four levels of C-FIND and C-MOVE query:  Patient

Study

Series

Image  Each one has its own required and optional tags. This class
provides an interface for getting those tags. This is an interface
class.

See 3.4 C 6.1 and 3.4 C 6.2 for the patient and study root query
types. These sections define the tags allowed by a particular query.
The caller must pass in which root type they want, patient or study. A
third root type, Modality Worklist Query, isn't yet supported.

This class (or rather it's derived classes) will be held in the
RootQuery types. These query types actually make the dataset, and will
use this dataset to list the required, unique, and optional tags for
each type of query. This design is somewhat overly complicated, but is
kept so that if we ever wanted to try to guess the query type from the
given tags, we could do so.

C++ includes: gdcmQueryBase.h ";

%feature("docstring")  gdcm::QueryBase::~QueryBase "virtual
gdcm::QueryBase::~QueryBase() ";

%feature("docstring")  gdcm::QueryBase::GetAllRequiredTags "std::vector<Tag> gdcm::QueryBase::GetAllRequiredTags(const ERootType
&inRootType) const

In order to validate a query dataset we need to check that there
exists at least one required (or unique) key ";

%feature("docstring")  gdcm::QueryBase::GetAllTags "std::vector<Tag>
gdcm::QueryBase::GetAllTags(const ERootType &inRootType) const

In order to validate a query dataset, just check for the presence of a
tag, not it's requirement level in the spec ";

%feature("docstring")  gdcm::QueryBase::GetHierachicalSearchTags "virtual std::vector<Tag>
gdcm::QueryBase::GetHierachicalSearchTags(const ERootType &inRootType)
const =0

Return all Unique Key for a particular Query Root type (from the same
level and above). ";

%feature("docstring")  gdcm::QueryBase::GetName "virtual const char*
gdcm::QueryBase::GetName() const =0 ";

%feature("docstring")  gdcm::QueryBase::GetOptionalTags "virtual
std::vector<Tag> gdcm::QueryBase::GetOptionalTags(const ERootType
&inRootType) const =0 ";

%feature("docstring")  gdcm::QueryBase::GetQueryLevel "virtual
DataElement gdcm::QueryBase::GetQueryLevel() const =0 ";

%feature("docstring")  gdcm::QueryBase::GetRequiredTags "virtual
std::vector<Tag> gdcm::QueryBase::GetRequiredTags(const ERootType
&inRootType) const =0 ";

%feature("docstring")  gdcm::QueryBase::GetUniqueTags "virtual
std::vector<Tag> gdcm::QueryBase::GetUniqueTags(const ERootType
&inRootType) const =0 ";


// File: classgdcm_1_1QueryFactory.xml
%feature("docstring") gdcm::QueryFactory "

QueryFactory.h.

contains: a class to produce a query based off of user-entered
information  Essentially, this class is used to construct a query
based off of user input (typically from the command line; if in code
directly, the query itself could just be instantiated)

In theory, could also be used as the interface to validate incoming
datasets as belonging to a particular query style

C++ includes: gdcmQueryFactory.h ";


// File: classgdcm_1_1QueryImage.xml
%feature("docstring") gdcm::QueryImage "

QueryImage contains: class to construct an image-based query for
C-FIND and C-MOVE.

C++ includes: gdcmQueryImage.h ";

%feature("docstring")  gdcm::QueryImage::GetHierachicalSearchTags "std::vector<Tag> gdcm::QueryImage::GetHierachicalSearchTags(const
ERootType &inRootType) const

Return all Unique Key for a particular Query Root type (from the same
level and above). ";

%feature("docstring")  gdcm::QueryImage::GetName "const char*
gdcm::QueryImage::GetName() const ";

%feature("docstring")  gdcm::QueryImage::GetOptionalTags "std::vector<Tag> gdcm::QueryImage::GetOptionalTags(const ERootType
&inRootType) const ";

%feature("docstring")  gdcm::QueryImage::GetQueryLevel "DataElement
gdcm::QueryImage::GetQueryLevel() const ";

%feature("docstring")  gdcm::QueryImage::GetRequiredTags "std::vector<Tag> gdcm::QueryImage::GetRequiredTags(const ERootType
&inRootType) const ";

%feature("docstring")  gdcm::QueryImage::GetUniqueTags "std::vector<Tag> gdcm::QueryImage::GetUniqueTags(const ERootType
&inRootType) const ";


// File: classgdcm_1_1QueryPatient.xml
%feature("docstring") gdcm::QueryPatient "

QueryPatient contains: class to construct a patient-based query for
c-find and c-move.

C++ includes: gdcmQueryPatient.h ";

%feature("docstring")  gdcm::QueryPatient::GetHierachicalSearchTags "std::vector<Tag> gdcm::QueryPatient::GetHierachicalSearchTags(const
ERootType &inRootType) const

Return all Unique Key for a particular Query Root type (from the same
level and above). ";

%feature("docstring")  gdcm::QueryPatient::GetName "const char*
gdcm::QueryPatient::GetName() const ";

%feature("docstring")  gdcm::QueryPatient::GetOptionalTags "std::vector<Tag> gdcm::QueryPatient::GetOptionalTags(const ERootType
&inRootType) const ";

%feature("docstring")  gdcm::QueryPatient::GetQueryLevel "DataElement
gdcm::QueryPatient::GetQueryLevel() const ";

%feature("docstring")  gdcm::QueryPatient::GetRequiredTags "std::vector<Tag> gdcm::QueryPatient::GetRequiredTags(const ERootType
&inRootType) const ";

%feature("docstring")  gdcm::QueryPatient::GetUniqueTags "std::vector<Tag> gdcm::QueryPatient::GetUniqueTags(const ERootType
&inRootType) const ";


// File: classgdcm_1_1QuerySeries.xml
%feature("docstring") gdcm::QuerySeries "

QuerySeries contains: class to construct a series-based query for
c-find and c-move.

C++ includes: gdcmQuerySeries.h ";

%feature("docstring")  gdcm::QuerySeries::GetHierachicalSearchTags "std::vector<Tag> gdcm::QuerySeries::GetHierachicalSearchTags(const
ERootType &inRootType) const

Return all Unique Key for a particular Query Root type (from the same
level and above). ";

%feature("docstring")  gdcm::QuerySeries::GetName "const char*
gdcm::QuerySeries::GetName() const ";

%feature("docstring")  gdcm::QuerySeries::GetOptionalTags "std::vector<Tag> gdcm::QuerySeries::GetOptionalTags(const ERootType
&inRootType) const ";

%feature("docstring")  gdcm::QuerySeries::GetQueryLevel "DataElement
gdcm::QuerySeries::GetQueryLevel() const ";

%feature("docstring")  gdcm::QuerySeries::GetRequiredTags "std::vector<Tag> gdcm::QuerySeries::GetRequiredTags(const ERootType
&inRootType) const ";

%feature("docstring")  gdcm::QuerySeries::GetUniqueTags "std::vector<Tag> gdcm::QuerySeries::GetUniqueTags(const ERootType
&inRootType) const ";


// File: classgdcm_1_1QueryStudy.xml
%feature("docstring") gdcm::QueryStudy "

QueryStudy.h contains: class to construct a study-based query for
C-FIND and C-MOVE.

C++ includes: gdcmQueryStudy.h ";

%feature("docstring")  gdcm::QueryStudy::GetHierachicalSearchTags "std::vector<Tag> gdcm::QueryStudy::GetHierachicalSearchTags(const
ERootType &inRootType) const

Return all Unique Key for a particular Query Root type (from the same
level and above). ";

%feature("docstring")  gdcm::QueryStudy::GetName "const char*
gdcm::QueryStudy::GetName() const ";

%feature("docstring")  gdcm::QueryStudy::GetOptionalTags "std::vector<Tag> gdcm::QueryStudy::GetOptionalTags(const ERootType
&inRootType) const ";

%feature("docstring")  gdcm::QueryStudy::GetQueryLevel "DataElement
gdcm::QueryStudy::GetQueryLevel() const ";

%feature("docstring")  gdcm::QueryStudy::GetRequiredTags "std::vector<Tag> gdcm::QueryStudy::GetRequiredTags(const ERootType
&inRootType) const ";

%feature("docstring")  gdcm::QueryStudy::GetUniqueTags "std::vector<Tag> gdcm::QueryStudy::GetUniqueTags(const ERootType
&inRootType) const ";


// File: classstd_1_1queue.xml
%feature("docstring") std::queue "

STL class. ";


// File: classstd_1_1range__error.xml
%feature("docstring") std::range_error "

STL class. ";


// File: classgdcm_1_1RAWCodec.xml
%feature("docstring") gdcm::RAWCodec "

RAWCodec class.

C++ includes: gdcmRAWCodec.h ";

%feature("docstring")  gdcm::RAWCodec::RAWCodec "gdcm::RAWCodec::RAWCodec() ";

%feature("docstring")  gdcm::RAWCodec::~RAWCodec "gdcm::RAWCodec::~RAWCodec() ";

%feature("docstring")  gdcm::RAWCodec::CanCode "bool
gdcm::RAWCodec::CanCode(TransferSyntax const &ts) const

Return whether this coder support this transfer syntax (can code it)
";

%feature("docstring")  gdcm::RAWCodec::CanDecode "bool
gdcm::RAWCodec::CanDecode(TransferSyntax const &ts) const

Return whether this decoder support this transfer syntax (can decode
it) ";

%feature("docstring")  gdcm::RAWCodec::Clone "virtual ImageCodec*
gdcm::RAWCodec::Clone() const ";

%feature("docstring")  gdcm::RAWCodec::Code "bool
gdcm::RAWCodec::Code(DataElement const &in, DataElement &out)

Code. ";

%feature("docstring")  gdcm::RAWCodec::Decode "bool
gdcm::RAWCodec::Decode(DataElement const &is, DataElement &os)

Decode. ";

%feature("docstring")  gdcm::RAWCodec::DecodeBytes "bool
gdcm::RAWCodec::DecodeBytes(const char *inBytes, size_t
inBufferLength, char *outBytes, size_t inOutBufferLength)

Used by the ImageStreamReader converts a read in buffer into one with
the proper encodings. ";

%feature("docstring")  gdcm::RAWCodec::GetHeaderInfo "bool
gdcm::RAWCodec::GetHeaderInfo(std::istream &is, TransferSyntax &ts) ";


// File: classgdcm_1_1Reader.xml
%feature("docstring") gdcm::Reader "

Reader ala DOM (Document Object Model)

This class is a non-validating reader, it will only performs well-
formedness check only, and to some extent catch known error (non well-
formed document).

Detailled description here

A DataSet DOES NOT contains group 0x0002 (see FileMetaInformation)

This is really a DataSet reader. This will not make sure the dataset
conform to any IOD at all. This is a completely different step. The
reasoning was that user could control the IOD there lib would handle
and thus we would not be able to read a DataSet if the IOD was not
found Instead we separate the reading from the validation.

From GDCM1.x. Users will realize that one feature is missing from this
DOM implementation. In GDCM 1.x user used to be able to control the
size of the Value to be read. By default it was 0xfff. The main author
of GDCM2 thought this was too dangerous and harmful and therefore this
feature did not make it into GDCM2

WARNING:  GDCM will not produce warning for unorder (non-alphabetical
order).

See:   Writer FileMetaInformation DataSet File

C++ includes: gdcmReader.h ";

%feature("docstring")  gdcm::Reader::Reader "gdcm::Reader::Reader()
";

%feature("docstring")  gdcm::Reader::~Reader "virtual
gdcm::Reader::~Reader() ";

%feature("docstring")  gdcm::Reader::CanRead "bool
gdcm::Reader::CanRead() const

Test whether this is a DICOM file WARNING:  need to call either
SetFileName or SetStream first ";

%feature("docstring")  gdcm::Reader::GetFile "const File&
gdcm::Reader::GetFile() const

Set/Get File. ";

%feature("docstring")  gdcm::Reader::GetFile "File&
gdcm::Reader::GetFile()

Set/Get File. ";

%feature("docstring")  gdcm::Reader::GetStreamCurrentPosition "size_t
gdcm::Reader::GetStreamCurrentPosition() const

For wrapped language. return type is compatible with System::FileSize
return type Use native std::streampos / std::streamoff directly from
the stream from C++ ";

%feature("docstring")  gdcm::Reader::Read "virtual bool
gdcm::Reader::Read()

Main function to read a file. ";

%feature("docstring")  gdcm::Reader::ReadSelectedPrivateTags "bool
gdcm::Reader::ReadSelectedPrivateTags(std::set< PrivateTag > const
&ptags, bool readvalues=true)

Will only read the specified selected private tags. ";

%feature("docstring")  gdcm::Reader::ReadSelectedTags "bool
gdcm::Reader::ReadSelectedTags(std::set< Tag > const &tags, bool
readvalues=true)

Will only read the specified selected tags. ";

%feature("docstring")  gdcm::Reader::ReadUpToTag "bool
gdcm::Reader::ReadUpToTag(const Tag &tag, std::set< Tag > const
&skiptags=std::set< Tag >())

Will read only up to Tag

Parameters:
-----------

tag:  and skipping any tag specified in

skiptags:  ";

%feature("docstring")  gdcm::Reader::SetFile "void
gdcm::Reader::SetFile(File &file)

Set/Get File. ";

%feature("docstring")  gdcm::Reader::SetFileName "void
gdcm::Reader::SetFileName(const char *filename_native)

Set the filename to open. This will create a std::ifstream internally
See SetStream if you are dealing with different std::istream object ";

%feature("docstring")  gdcm::Reader::SetStream "void
gdcm::Reader::SetStream(std::istream &input_stream)

Set the open-ed stream directly. ";


// File: classgdcm_1_1Region.xml
%feature("docstring") gdcm::Region "

Class for manipulation region.

C++ includes: gdcmRegion.h ";

%feature("docstring")  gdcm::Region::Region "gdcm::Region::Region()
";

%feature("docstring")  gdcm::Region::~Region "virtual
gdcm::Region::~Region() ";

%feature("docstring")  gdcm::Region::Area "virtual size_t
gdcm::Region::Area() const =0

compute the area ";

%feature("docstring")  gdcm::Region::Clone "virtual Region*
gdcm::Region::Clone() const =0 ";

%feature("docstring")  gdcm::Region::ComputeBoundingBox "virtual
BoxRegion gdcm::Region::ComputeBoundingBox()=0

Return the Axis-Aligned minimum bounding box for all regions. ";

%feature("docstring")  gdcm::Region::Empty "virtual bool
gdcm::Region::Empty() const =0

return whether this domain is empty: ";

%feature("docstring")  gdcm::Region::IsValid "virtual bool
gdcm::Region::IsValid() const =0

return whether this is valid domain ";

%feature("docstring")  gdcm::Region::Print "virtual void
gdcm::Region::Print(std::ostream &os=std::cout) const

Print. ";


// File: classgdcm_1_1Rescaler.xml
%feature("docstring") gdcm::Rescaler "

Rescale class This class is meant to apply the linear transform of
Stored Pixel Value to Real World Value. This is mostly found in CT or
PET dataset, where the value are stored using one type, but need to be
converted to another scale using a linear transform. There are
basically two cases: In CT: the linear transform is generally integer
based. E.g. the Stored Pixel Type is unsigned short 12bits, but to get
Hounsfield unit, one need to apply the linear transform: \\\\[ RWV =
1. * SV - 1024 \\\\] So the best scalar to store the Real World Value
will be 16 bits signed type.

In PET: the linear transform is generally floating point based. Since
the dynamic range can be quite high, the Rescale Slope / Rescale
Intercept can be changing throughout the Series. So it is important to
read all linear transform and deduce the best Pixel Type only at the
end (when all the images to be read have been parsed).

WARNING:  Internally any time a floating point value is found either
in the Rescale Slope or the Rescale Intercept it is assumed that the
best matching output pixel type is FLOAT64 (in previous implementation
it was FLOAT32). Because VR:DS is closer to a 64bits floating point
type FLOAT64 is thus a best matching pixel type for the floating point
transformation.  Example: Let say input is FLOAT64, and we want UINT16
as ouput, we would do:

handle floating point transformation back and forth to integer
properly (no loss)

See:   Unpacker12Bits

C++ includes: gdcmRescaler.h ";

%feature("docstring")  gdcm::Rescaler::Rescaler "gdcm::Rescaler::Rescaler() ";

%feature("docstring")  gdcm::Rescaler::~Rescaler "gdcm::Rescaler::~Rescaler() ";

%feature("docstring")  gdcm::Rescaler::ComputeInterceptSlopePixelType
"PixelFormat::ScalarType
gdcm::Rescaler::ComputeInterceptSlopePixelType()

Compute the Pixel Format of the output data Used for direct
transformation ";

%feature("docstring")  gdcm::Rescaler::ComputePixelTypeFromMinMax "PixelFormat gdcm::Rescaler::ComputePixelTypeFromMinMax()

Compute the Pixel Format of the output data Used for inverse
transformation ";

%feature("docstring")  gdcm::Rescaler::GetIntercept "double
gdcm::Rescaler::GetIntercept() const ";

%feature("docstring")  gdcm::Rescaler::GetSlope "double
gdcm::Rescaler::GetSlope() const ";

%feature("docstring")  gdcm::Rescaler::InverseRescale "bool
gdcm::Rescaler::InverseRescale(char *out, const char *in, size_t n)

Inverse transform. ";

%feature("docstring")  gdcm::Rescaler::Rescale "bool
gdcm::Rescaler::Rescale(char *out, const char *in, size_t n)

Direct transform. ";

%feature("docstring")  gdcm::Rescaler::SetIntercept "void
gdcm::Rescaler::SetIntercept(double i)

Set Intercept: used for both direct&inverse transformation. ";

%feature("docstring")  gdcm::Rescaler::SetMinMaxForPixelType "void
gdcm::Rescaler::SetMinMaxForPixelType(double min, double max)

Set target interval for output data. A best match will be computed (if
possible) Used for inverse transformation ";

%feature("docstring")  gdcm::Rescaler::SetPixelFormat "void
gdcm::Rescaler::SetPixelFormat(PixelFormat const &pf)

Set Pixel Format of input data. ";

%feature("docstring")  gdcm::Rescaler::SetSlope "void
gdcm::Rescaler::SetSlope(double s)

Set Slope: user for both direct&inverse transformation. ";

%feature("docstring")  gdcm::Rescaler::SetTargetPixelType "void
gdcm::Rescaler::SetTargetPixelType(PixelFormat const &targetst)

By default (when UseTargetPixelType is false), a best matching Target
Pixel Type is computed. However user can override this auto selection
by switching UseTargetPixelType:true and also specifying the specifix
Target Pixel Type ";

%feature("docstring")  gdcm::Rescaler::SetUseTargetPixelType "void
gdcm::Rescaler::SetUseTargetPixelType(bool b)

Override default behavior of Rescale. ";


// File: classstd_1_1multimap_1_1reverse__iterator.xml
%feature("docstring") std::multimap::reverse_iterator "

STL iterator class. ";


// File: classstd_1_1list_1_1reverse__iterator.xml
%feature("docstring") std::list::reverse_iterator "

STL iterator class. ";


// File: classstd_1_1forward__list_1_1reverse__iterator.xml
%feature("docstring") std::forward_list::reverse_iterator "

STL iterator class. ";


// File: classstd_1_1unordered__map_1_1reverse__iterator.xml
%feature("docstring") std::unordered_map::reverse_iterator "

STL iterator class. ";


// File: classstd_1_1basic__string_1_1reverse__iterator.xml
%feature("docstring") std::basic_string::reverse_iterator "

STL iterator class. ";


// File: classstd_1_1unordered__multimap_1_1reverse__iterator.xml
%feature("docstring") std::unordered_multimap::reverse_iterator "

STL iterator class. ";


// File: classstd_1_1set_1_1reverse__iterator.xml
%feature("docstring") std::set::reverse_iterator "

STL iterator class. ";


// File: classstd_1_1string_1_1reverse__iterator.xml
%feature("docstring") std::string::reverse_iterator "

STL iterator class. ";


// File: classstd_1_1unordered__set_1_1reverse__iterator.xml
%feature("docstring") std::unordered_set::reverse_iterator "

STL iterator class. ";


// File: classstd_1_1multiset_1_1reverse__iterator.xml
%feature("docstring") std::multiset::reverse_iterator "

STL iterator class. ";


// File: classstd_1_1wstring_1_1reverse__iterator.xml
%feature("docstring") std::wstring::reverse_iterator "

STL iterator class. ";


// File: classstd_1_1unordered__multiset_1_1reverse__iterator.xml
%feature("docstring") std::unordered_multiset::reverse_iterator "

STL iterator class. ";


// File: classstd_1_1vector_1_1reverse__iterator.xml
%feature("docstring") std::vector::reverse_iterator "

STL iterator class. ";


// File: classstd_1_1deque_1_1reverse__iterator.xml
%feature("docstring") std::deque::reverse_iterator "

STL iterator class. ";


// File: classstd_1_1map_1_1reverse__iterator.xml
%feature("docstring") std::map::reverse_iterator "

STL iterator class. ";


// File: classgdcm_1_1RLECodec.xml
%feature("docstring") gdcm::RLECodec "

Class to do RLE.

ANSI X3.9 A.4.2 RLE Compression Annex G defines a RLE Compression
Transfer Syntax. This transfer Syntax is identified by the UID value
\"1.2.840.10008.1.2.5\". If the object allows multi-frame images in
the pixel data field, then each frame shall be encoded separately.
Each frame shall be encoded in one and only one Fragment (see PS
3.5.8.2).

C++ includes: gdcmRLECodec.h ";

%feature("docstring")  gdcm::RLECodec::RLECodec "gdcm::RLECodec::RLECodec() ";

%feature("docstring")  gdcm::RLECodec::~RLECodec "gdcm::RLECodec::~RLECodec() ";

%feature("docstring")  gdcm::RLECodec::CanCode "bool
gdcm::RLECodec::CanCode(TransferSyntax const &ts) const

Return whether this coder support this transfer syntax (can code it)
";

%feature("docstring")  gdcm::RLECodec::CanDecode "bool
gdcm::RLECodec::CanDecode(TransferSyntax const &ts) const

Return whether this decoder support this transfer syntax (can decode
it) ";

%feature("docstring")  gdcm::RLECodec::Clone "virtual ImageCodec*
gdcm::RLECodec::Clone() const ";

%feature("docstring")  gdcm::RLECodec::Code "bool
gdcm::RLECodec::Code(DataElement const &in, DataElement &out)

Code. ";

%feature("docstring")  gdcm::RLECodec::Decode "bool
gdcm::RLECodec::Decode(DataElement const &is, DataElement &os)

Decode. ";

%feature("docstring")  gdcm::RLECodec::GetBufferLength "unsigned long
gdcm::RLECodec::GetBufferLength() const ";

%feature("docstring")  gdcm::RLECodec::GetHeaderInfo "bool
gdcm::RLECodec::GetHeaderInfo(std::istream &is, TransferSyntax &ts) ";

%feature("docstring")  gdcm::RLECodec::SetBufferLength "void
gdcm::RLECodec::SetBufferLength(unsigned long l) ";

%feature("docstring")  gdcm::RLECodec::SetLength "void
gdcm::RLECodec::SetLength(unsigned long l) ";


// File: classgdcm_1_1network_1_1RoleSelectionSub.xml
%feature("docstring") gdcm::network::RoleSelectionSub "

RoleSelectionSub PS 3.7 Table D.3-9 SCP/SCU ROLE SELECTION SUB-ITEM
FIELDS (A-ASSOCIATE-RQ)

C++ includes: gdcmRoleSelectionSub.h ";

%feature("docstring")
gdcm::network::RoleSelectionSub::RoleSelectionSub "gdcm::network::RoleSelectionSub::RoleSelectionSub() ";

%feature("docstring")  gdcm::network::RoleSelectionSub::Print "void
gdcm::network::RoleSelectionSub::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::network::RoleSelectionSub::Read "std::istream& gdcm::network::RoleSelectionSub::Read(std::istream &is)
";

%feature("docstring")  gdcm::network::RoleSelectionSub::SetTuple "void gdcm::network::RoleSelectionSub::SetTuple(const char *uid,
uint8_t scurole, uint8_t scprole) ";

%feature("docstring")  gdcm::network::RoleSelectionSub::Size "size_t
gdcm::network::RoleSelectionSub::Size() const ";

%feature("docstring")  gdcm::network::RoleSelectionSub::Write "const
std::ostream& gdcm::network::RoleSelectionSub::Write(std::ostream &os)
const ";


// File: structgdcm_1_1SerieHelper_1_1Rule.xml


// File: classstd_1_1runtime__error.xml
%feature("docstring") std::runtime_error "

STL class. ";


// File: classgdcm_1_1Scanner.xml
%feature("docstring") gdcm::Scanner "

Scanner This filter is meant for quickly browsing a FileSet (a set of
files on disk). Special consideration are taken so as to read the
mimimum amount of information in each file in order to retrieve the
user specified set of DICOM Attribute.

This filter is dealing with both VRASCII and VRBINARY element, thanks
to the help of StringFilter

WARNING:  IMPORTANT In case of file where tags are not ordered
(illegal as per DICOM specification), the output will be missing
information

implementation details. All values are stored in a std::set of
std::string. Then the address of the cstring underlying the
std::string is used in the std::map.  This class implement the
Subject/Observer pattern trigger the following events:  ProgressEvent

StartEvent

EndEvent

C++ includes: gdcmScanner.h ";

%feature("docstring")  gdcm::Scanner::Scanner "gdcm::Scanner::Scanner() ";

%feature("docstring")  gdcm::Scanner::~Scanner "gdcm::Scanner::~Scanner() ";

%feature("docstring")  gdcm::Scanner::AddPrivateTag "void
gdcm::Scanner::AddPrivateTag(PrivateTag const &t) ";

%feature("docstring")  gdcm::Scanner::AddSkipTag "void
gdcm::Scanner::AddSkipTag(Tag const &t)

Add a tag that will need to be skipped. Those are root level skip
tags. ";

%feature("docstring")  gdcm::Scanner::AddTag "void
gdcm::Scanner::AddTag(Tag const &t)

Add a tag that will need to be read. Those are root level skip tags.
";

%feature("docstring")  gdcm::Scanner::Begin "ConstIterator
gdcm::Scanner::Begin() const ";

%feature("docstring")  gdcm::Scanner::ClearSkipTags "void
gdcm::Scanner::ClearSkipTags() ";

%feature("docstring")  gdcm::Scanner::ClearTags "void
gdcm::Scanner::ClearTags() ";

%feature("docstring")  gdcm::Scanner::End "ConstIterator
gdcm::Scanner::End() const ";

%feature("docstring")  gdcm::Scanner::GetAllFilenamesFromTagToValue "Directory::FilenamesType
gdcm::Scanner::GetAllFilenamesFromTagToValue(Tag const &t, const char
*valueref) const

Will loop over all files and return a vector of std::strings of
filenames where value match the reference value 'valueref' ";

%feature("docstring")  gdcm::Scanner::GetFilenameFromTagToValue "const char* gdcm::Scanner::GetFilenameFromTagToValue(Tag const &t,
const char *valueref) const

Will loop over all files and return the first file where value match
the reference value 'valueref' ";

%feature("docstring")  gdcm::Scanner::GetFilenames "Directory::FilenamesType const& gdcm::Scanner::GetFilenames() const ";

%feature("docstring")  gdcm::Scanner::GetKeys "Directory::FilenamesType gdcm::Scanner::GetKeys() const

Return the list of filename that are key in the internal map, which
means those filename were properly parsed ";

%feature("docstring")  gdcm::Scanner::GetMapping "TagToValue const&
gdcm::Scanner::GetMapping(const char *filename) const

Get the std::map mapping filenames to value for file 'filename'. ";

%feature("docstring")  gdcm::Scanner::GetMappingFromTagToValue "TagToValue const& gdcm::Scanner::GetMappingFromTagToValue(Tag const
&t, const char *value) const

See GetFilenameFromTagToValue(). This is simply
GetFilenameFromTagToValue followed. ";

%feature("docstring")  gdcm::Scanner::GetMappings "MappingType const&
gdcm::Scanner::GetMappings() const

Mappings are the mapping from a particular tag to the map, mapping
filename to value: ";

%feature("docstring")  gdcm::Scanner::GetOrderedValues "Directory::FilenamesType gdcm::Scanner::GetOrderedValues(Tag const &t)
const

Get all the values found (in a vector) associated with Tag 't' This
function is identical to GetValues, but is accessible from the wrapped
layer (python, C#, java) ";

%feature("docstring")  gdcm::Scanner::GetValue "const char*
gdcm::Scanner::GetValue(const char *filename, Tag const &t) const

Retrieve the value found for tag: t associated with file: filename
This is meant for a single short call. If multiple calls (multiple
tags) should be done, prefer the GetMapping function, and then reuse
the TagToValue hash table. WARNING:   Tag 't' should have been added
via AddTag() prior to the Scan() call ! ";

%feature("docstring")  gdcm::Scanner::GetValues "ValuesType const&
gdcm::Scanner::GetValues() const

Get all the values found (in lexicographic order) ";

%feature("docstring")  gdcm::Scanner::GetValues "ValuesType
gdcm::Scanner::GetValues(Tag const &t) const

Get all the values found (in lexicographic order) associated with Tag
't'. ";

%feature("docstring")  gdcm::Scanner::IsKey "bool
gdcm::Scanner::IsKey(const char *filename) const

Check if filename is a key in the Mapping table. returns true only of
file can be found, which means the file was indeed a DICOM file that
could be processed ";

%feature("docstring")  gdcm::Scanner::Print "void
gdcm::Scanner::Print(std::ostream &os) const

Print result. ";

%feature("docstring")  gdcm::Scanner::Scan "bool
gdcm::Scanner::Scan(Directory::FilenamesType const &filenames)

Start the scan ! ";


// File: classgdcm_1_1Segment.xml
%feature("docstring") gdcm::Segment "

This class defines a segment. It mainly contains attributes of group
0x0062. In addition, it can be associated with surface.

See:  PS 3.3 C.8.20.2 and C.8.23

C++ includes: gdcmSegment.h ";

%feature("docstring")  gdcm::Segment::Segment "gdcm::Segment::Segment() ";

%feature("docstring")  gdcm::Segment::~Segment "virtual
gdcm::Segment::~Segment() ";

%feature("docstring")  gdcm::Segment::AddSurface "void
gdcm::Segment::AddSurface(SmartPointer< Surface > surface) ";

%feature("docstring")  gdcm::Segment::GetAnatomicRegion "SegmentHelper::BasicCodedEntry const&
gdcm::Segment::GetAnatomicRegion() const ";

%feature("docstring")  gdcm::Segment::GetAnatomicRegion "SegmentHelper::BasicCodedEntry& gdcm::Segment::GetAnatomicRegion() ";

%feature("docstring")  gdcm::Segment::GetPropertyCategory "SegmentHelper::BasicCodedEntry const&
gdcm::Segment::GetPropertyCategory() const ";

%feature("docstring")  gdcm::Segment::GetPropertyCategory "SegmentHelper::BasicCodedEntry& gdcm::Segment::GetPropertyCategory()
";

%feature("docstring")  gdcm::Segment::GetPropertyType "SegmentHelper::BasicCodedEntry const& gdcm::Segment::GetPropertyType()
const ";

%feature("docstring")  gdcm::Segment::GetPropertyType "SegmentHelper::BasicCodedEntry& gdcm::Segment::GetPropertyType() ";

%feature("docstring")  gdcm::Segment::GetSegmentAlgorithmName "const
char* gdcm::Segment::GetSegmentAlgorithmName() const ";

%feature("docstring")  gdcm::Segment::GetSegmentAlgorithmType "ALGOType gdcm::Segment::GetSegmentAlgorithmType() const ";

%feature("docstring")  gdcm::Segment::GetSegmentDescription "const
char* gdcm::Segment::GetSegmentDescription() const ";

%feature("docstring")  gdcm::Segment::GetSegmentLabel "const char*
gdcm::Segment::GetSegmentLabel() const ";

%feature("docstring")  gdcm::Segment::GetSegmentNumber "unsigned
short gdcm::Segment::GetSegmentNumber() const ";

%feature("docstring")  gdcm::Segment::GetSurface "SmartPointer<
Surface > gdcm::Segment::GetSurface(const unsigned int idx=0) const ";

%feature("docstring")  gdcm::Segment::GetSurfaceCount "unsigned long
gdcm::Segment::GetSurfaceCount() ";

%feature("docstring")  gdcm::Segment::GetSurfaces "SurfaceVector
const& gdcm::Segment::GetSurfaces() const ";

%feature("docstring")  gdcm::Segment::GetSurfaces "SurfaceVector&
gdcm::Segment::GetSurfaces() ";

%feature("docstring")  gdcm::Segment::SetAnatomicRegion "void
gdcm::Segment::SetAnatomicRegion(SegmentHelper::BasicCodedEntry const
&BSE) ";

%feature("docstring")  gdcm::Segment::SetPropertyCategory "void
gdcm::Segment::SetPropertyCategory(SegmentHelper::BasicCodedEntry
const &BSE) ";

%feature("docstring")  gdcm::Segment::SetPropertyType "void
gdcm::Segment::SetPropertyType(SegmentHelper::BasicCodedEntry const
&BSE) ";

%feature("docstring")  gdcm::Segment::SetSegmentAlgorithmName "void
gdcm::Segment::SetSegmentAlgorithmName(const char *name) ";

%feature("docstring")  gdcm::Segment::SetSegmentAlgorithmType "void
gdcm::Segment::SetSegmentAlgorithmType(ALGOType type) ";

%feature("docstring")  gdcm::Segment::SetSegmentAlgorithmType "void
gdcm::Segment::SetSegmentAlgorithmType(const char *typeStr) ";

%feature("docstring")  gdcm::Segment::SetSegmentDescription "void
gdcm::Segment::SetSegmentDescription(const char *description) ";

%feature("docstring")  gdcm::Segment::SetSegmentLabel "void
gdcm::Segment::SetSegmentLabel(const char *label) ";

%feature("docstring")  gdcm::Segment::SetSegmentNumber "void
gdcm::Segment::SetSegmentNumber(const unsigned short num) ";

%feature("docstring")  gdcm::Segment::SetSurfaceCount "void
gdcm::Segment::SetSurfaceCount(const unsigned long nb) ";


// File: classgdcm_1_1SegmentedPaletteColorLookupTable.xml
%feature("docstring") gdcm::SegmentedPaletteColorLookupTable "

SegmentedPaletteColorLookupTable class.

C++ includes: gdcmSegmentedPaletteColorLookupTable.h ";

%feature("docstring")
gdcm::SegmentedPaletteColorLookupTable::SegmentedPaletteColorLookupTable
"gdcm::SegmentedPaletteColorLookupTable::SegmentedPaletteColorLookupTable()
";

%feature("docstring")
gdcm::SegmentedPaletteColorLookupTable::~SegmentedPaletteColorLookupTable
"gdcm::SegmentedPaletteColorLookupTable::~SegmentedPaletteColorLookupTable()
";

%feature("docstring")  gdcm::SegmentedPaletteColorLookupTable::Print "void gdcm::SegmentedPaletteColorLookupTable::Print(std::ostream &)
const ";

%feature("docstring")  gdcm::SegmentedPaletteColorLookupTable::SetLUT
"void gdcm::SegmentedPaletteColorLookupTable::SetLUT(LookupTableType
type, const unsigned char *array, unsigned int length)

Initialize a SegmentedPaletteColorLookupTable. ";


// File: classgdcm_1_1SegmentReader.xml
%feature("docstring") gdcm::SegmentReader "

This class defines a segment reader. It reads attributes of group
0x0062.

See:  PS 3.3 C.8.20.2 and C.8.23

C++ includes: gdcmSegmentReader.h ";

%feature("docstring")  gdcm::SegmentReader::SegmentReader "gdcm::SegmentReader::SegmentReader() ";

%feature("docstring")  gdcm::SegmentReader::~SegmentReader "virtual
gdcm::SegmentReader::~SegmentReader() ";

%feature("docstring")  gdcm::SegmentReader::GetSegments "const
SegmentVector gdcm::SegmentReader::GetSegments() const ";

%feature("docstring")  gdcm::SegmentReader::GetSegments "SegmentVector gdcm::SegmentReader::GetSegments() ";

%feature("docstring")  gdcm::SegmentReader::Read "virtual bool
gdcm::SegmentReader::Read()

Read. ";


// File: classgdcm_1_1SegmentWriter.xml
%feature("docstring") gdcm::SegmentWriter "

This class defines a segment writer. It writes attributes of group
0x0062.

See:  PS 3.3 C.8.20.2 and C.8.23

C++ includes: gdcmSegmentWriter.h ";

%feature("docstring")  gdcm::SegmentWriter::SegmentWriter "gdcm::SegmentWriter::SegmentWriter() ";

%feature("docstring")  gdcm::SegmentWriter::~SegmentWriter "virtual
gdcm::SegmentWriter::~SegmentWriter() ";

%feature("docstring")  gdcm::SegmentWriter::AddSegment "void
gdcm::SegmentWriter::AddSegment(SmartPointer< Segment > segment) ";

%feature("docstring")  gdcm::SegmentWriter::GetNumberOfSegments "unsigned int gdcm::SegmentWriter::GetNumberOfSegments() const ";

%feature("docstring")  gdcm::SegmentWriter::GetSegment "SmartPointer<
Segment > gdcm::SegmentWriter::GetSegment(const unsigned int idx=0)
const ";

%feature("docstring")  gdcm::SegmentWriter::GetSegments "const
SegmentVector& gdcm::SegmentWriter::GetSegments() const ";

%feature("docstring")  gdcm::SegmentWriter::GetSegments "SegmentVector& gdcm::SegmentWriter::GetSegments() ";

%feature("docstring")  gdcm::SegmentWriter::SetNumberOfSegments "void
gdcm::SegmentWriter::SetNumberOfSegments(const unsigned int size) ";

%feature("docstring")  gdcm::SegmentWriter::SetSegments "void
gdcm::SegmentWriter::SetSegments(SegmentVector &segments) ";

%feature("docstring")  gdcm::SegmentWriter::Write "bool
gdcm::SegmentWriter::Write()

Write. ";


// File: classgdcm_1_1SequenceOfFragments.xml
%feature("docstring") gdcm::SequenceOfFragments "

Class to represent a Sequence Of Fragments.

Todo I do not enforce that Sequence of Fragments ends with a SQ end
del

C++ includes: gdcmSequenceOfFragments.h ";

%feature("docstring")  gdcm::SequenceOfFragments::SequenceOfFragments
"gdcm::SequenceOfFragments::SequenceOfFragments()

constructor (UndefinedLength by default) ";

%feature("docstring")  gdcm::SequenceOfFragments::AddFragment "void
gdcm::SequenceOfFragments::AddFragment(Fragment const &item)

Appends a Fragment to the already added ones. ";

%feature("docstring")  gdcm::SequenceOfFragments::Begin "Iterator
gdcm::SequenceOfFragments::Begin() ";

%feature("docstring")  gdcm::SequenceOfFragments::Begin "ConstIterator gdcm::SequenceOfFragments::Begin() const ";

%feature("docstring")  gdcm::SequenceOfFragments::Clear "void
gdcm::SequenceOfFragments::Clear()

Clear. ";

%feature("docstring")  gdcm::SequenceOfFragments::ComputeByteLength "unsigned long gdcm::SequenceOfFragments::ComputeByteLength() const ";

%feature("docstring")  gdcm::SequenceOfFragments::ComputeLength "VL
gdcm::SequenceOfFragments::ComputeLength() const ";

%feature("docstring")  gdcm::SequenceOfFragments::End "Iterator
gdcm::SequenceOfFragments::End() ";

%feature("docstring")  gdcm::SequenceOfFragments::End "ConstIterator
gdcm::SequenceOfFragments::End() const ";

%feature("docstring")  gdcm::SequenceOfFragments::GetBuffer "bool
gdcm::SequenceOfFragments::GetBuffer(char *buffer, unsigned long
length) const ";

%feature("docstring")  gdcm::SequenceOfFragments::GetFragBuffer "bool
gdcm::SequenceOfFragments::GetFragBuffer(unsigned int fragNb, char
*buffer, unsigned long &length) const ";

%feature("docstring")  gdcm::SequenceOfFragments::GetFragment "const
Fragment& gdcm::SequenceOfFragments::GetFragment(SizeType num) const
";

%feature("docstring")  gdcm::SequenceOfFragments::GetLength "VL
gdcm::SequenceOfFragments::GetLength() const

Returns the SQ length, as read from disk. ";

%feature("docstring")  gdcm::SequenceOfFragments::GetNumberOfFragments
"SizeType gdcm::SequenceOfFragments::GetNumberOfFragments() const ";

%feature("docstring")  gdcm::SequenceOfFragments::GetTable "const
BasicOffsetTable& gdcm::SequenceOfFragments::GetTable() const ";

%feature("docstring")  gdcm::SequenceOfFragments::GetTable "BasicOffsetTable& gdcm::SequenceOfFragments::GetTable() ";

%feature("docstring")  gdcm::SequenceOfFragments::Print "void
gdcm::SequenceOfFragments::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::SequenceOfFragments::Read "std::istream&
gdcm::SequenceOfFragments::Read(std::istream &is, bool
readvalues=true) ";

%feature("docstring")  gdcm::SequenceOfFragments::ReadPreValue "std::istream& gdcm::SequenceOfFragments::ReadPreValue(std::istream
&is) ";

%feature("docstring")  gdcm::SequenceOfFragments::ReadValue "std::istream& gdcm::SequenceOfFragments::ReadValue(std::istream &is,
bool) ";

%feature("docstring")  gdcm::SequenceOfFragments::SetLength "void
gdcm::SequenceOfFragments::SetLength(VL length)

Sets the actual SQ length. ";

%feature("docstring")  gdcm::SequenceOfFragments::Write "std::ostream
const& gdcm::SequenceOfFragments::Write(std::ostream &os) const ";

%feature("docstring")  gdcm::SequenceOfFragments::WriteBuffer "bool
gdcm::SequenceOfFragments::WriteBuffer(std::ostream &os) const ";


// File: classgdcm_1_1SequenceOfItems.xml
%feature("docstring") gdcm::SequenceOfItems "

Class to represent a Sequence Of Items (value representation : SQ)

a Value Representation for Data Elements that contains a sequence of
Data Sets.

Sequence of Item allows for Nested Data Sets

See PS 3.5, 7.4.6 Data Element Type Within a Sequence SEQUENCE OF
ITEMS (VALUE REPRESENTATION SQ) A Value Representation for Data
Elements that contain a sequence of Data Sets. Sequence of Items
allows for Nested Data Sets.

C++ includes: gdcmSequenceOfItems.h ";

%feature("docstring")  gdcm::SequenceOfItems::SequenceOfItems "gdcm::SequenceOfItems::SequenceOfItems()

constructor (UndefinedLength by default) ";

%feature("docstring")  gdcm::SequenceOfItems::AddItem "void
gdcm::SequenceOfItems::AddItem(Item const &item)

Appends an Item to the already added ones. ";

%feature("docstring")  gdcm::SequenceOfItems::Begin "Iterator
gdcm::SequenceOfItems::Begin() ";

%feature("docstring")  gdcm::SequenceOfItems::Begin "ConstIterator
gdcm::SequenceOfItems::Begin() const ";

%feature("docstring")  gdcm::SequenceOfItems::Clear "void
gdcm::SequenceOfItems::Clear()

remove all items within the sequence ";

%feature("docstring")  gdcm::SequenceOfItems::ComputeLength "VL
gdcm::SequenceOfItems::ComputeLength() const ";

%feature("docstring")  gdcm::SequenceOfItems::End "Iterator
gdcm::SequenceOfItems::End() ";

%feature("docstring")  gdcm::SequenceOfItems::End "ConstIterator
gdcm::SequenceOfItems::End() const ";

%feature("docstring")  gdcm::SequenceOfItems::FindDataElement "bool
gdcm::SequenceOfItems::FindDataElement(const Tag &t) const ";

%feature("docstring")  gdcm::SequenceOfItems::GetItem "const Item&
gdcm::SequenceOfItems::GetItem(SizeType position) const ";

%feature("docstring")  gdcm::SequenceOfItems::GetItem "Item&
gdcm::SequenceOfItems::GetItem(SizeType position) ";

%feature("docstring")  gdcm::SequenceOfItems::GetLength "VL
gdcm::SequenceOfItems::GetLength() const

Returns the SQ length, as read from disk. ";

%feature("docstring")  gdcm::SequenceOfItems::GetNumberOfItems "SizeType gdcm::SequenceOfItems::GetNumberOfItems() const ";

%feature("docstring")  gdcm::SequenceOfItems::IsUndefinedLength "bool
gdcm::SequenceOfItems::IsUndefinedLength() const

return if Value Length if of undefined length ";

%feature("docstring")  gdcm::SequenceOfItems::Print "void
gdcm::SequenceOfItems::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::SequenceOfItems::Read "std::istream&
gdcm::SequenceOfItems::Read(std::istream &is, bool readvalues=true) ";

%feature("docstring")  gdcm::SequenceOfItems::RemoveItemByIndex "bool
gdcm::SequenceOfItems::RemoveItemByIndex(const SizeType index)

Remove an Item as specified by its index, if index > size, false is
returned Index starts at 1 not 0 ";

%feature("docstring")  gdcm::SequenceOfItems::SetLength "void
gdcm::SequenceOfItems::SetLength(VL length)

Sets the actual SQ length. ";

%feature("docstring")  gdcm::SequenceOfItems::SetLengthToUndefined "void gdcm::SequenceOfItems::SetLengthToUndefined()

Properly set the Sequence of Item to be undefined length. ";

%feature("docstring")  gdcm::SequenceOfItems::SetNumberOfItems "void
gdcm::SequenceOfItems::SetNumberOfItems(SizeType n) ";

%feature("docstring")  gdcm::SequenceOfItems::Write "std::ostream
const& gdcm::SequenceOfItems::Write(std::ostream &os) const ";


// File: classgdcm_1_1SerieHelper.xml
%feature("docstring") gdcm::SerieHelper "

SerieHelper DO NOT USE this class, it is only a temporary solution for
ITK migration from GDCM 1.x to GDCM 2.x It will disapear soon, you've
been warned.

Instead see ImageHelper or IPPSorter

C++ includes: gdcmSerieHelper.h ";

%feature("docstring")  gdcm::SerieHelper::SerieHelper "gdcm::SerieHelper::SerieHelper() ";

%feature("docstring")  gdcm::SerieHelper::~SerieHelper "gdcm::SerieHelper::~SerieHelper() ";

%feature("docstring")  gdcm::SerieHelper::AddRestriction "void
gdcm::SerieHelper::AddRestriction(const std::string &tag) ";

%feature("docstring")  gdcm::SerieHelper::AddRestriction "void
gdcm::SerieHelper::AddRestriction(uint16_t group, uint16_t elem,
std::string const &value, int op) ";

%feature("docstring")  gdcm::SerieHelper::Clear "void
gdcm::SerieHelper::Clear() ";

%feature("docstring")
gdcm::SerieHelper::CreateDefaultUniqueSeriesIdentifier "void
gdcm::SerieHelper::CreateDefaultUniqueSeriesIdentifier() ";

%feature("docstring")  gdcm::SerieHelper::CreateUniqueSeriesIdentifier
"std::string gdcm::SerieHelper::CreateUniqueSeriesIdentifier(File
*inFile) ";

%feature("docstring")
gdcm::SerieHelper::GetFirstSingleSerieUIDFileSet "FileList*
gdcm::SerieHelper::GetFirstSingleSerieUIDFileSet() ";

%feature("docstring")  gdcm::SerieHelper::GetNextSingleSerieUIDFileSet
"FileList* gdcm::SerieHelper::GetNextSingleSerieUIDFileSet() ";

%feature("docstring")  gdcm::SerieHelper::OrderFileList "void
gdcm::SerieHelper::OrderFileList(FileList *fileSet) ";

%feature("docstring")  gdcm::SerieHelper::SetDirectory "void
gdcm::SerieHelper::SetDirectory(std::string const &dir, bool
recursive=false) ";

%feature("docstring")  gdcm::SerieHelper::SetLoadMode "void
gdcm::SerieHelper::SetLoadMode(int) ";

%feature("docstring")  gdcm::SerieHelper::SetUseSeriesDetails "void
gdcm::SerieHelper::SetUseSeriesDetails(bool useSeriesDetails) ";


// File: classgdcm_1_1Series.xml
%feature("docstring") gdcm::Series "

Series.

C++ includes: gdcmSeries.h ";

%feature("docstring")  gdcm::Series::Series "gdcm::Series::Series()
";


// File: classgdcm_1_1network_1_1ServiceClassApplicationInformation.xml
%feature("docstring")
gdcm::network::ServiceClassApplicationInformation "

PS 3.4 Table B.3-1 SERVICE-CLASS-APPLICATION-INFORMATION (A-ASSOCIATE-
RQ)

C++ includes: gdcmServiceClassApplicationInformation.h ";

%feature("docstring")
gdcm::network::ServiceClassApplicationInformation::ServiceClassApplicationInformation
"gdcm::network::ServiceClassApplicationInformation::ServiceClassApplicationInformation()
";

%feature("docstring")
gdcm::network::ServiceClassApplicationInformation::Print "void
gdcm::network::ServiceClassApplicationInformation::Print(std::ostream
&os) const ";

%feature("docstring")
gdcm::network::ServiceClassApplicationInformation::Read "std::istream&
gdcm::network::ServiceClassApplicationInformation::Read(std::istream
&is) ";

%feature("docstring")
gdcm::network::ServiceClassApplicationInformation::SetTuple "void
gdcm::network::ServiceClassApplicationInformation::SetTuple(uint8_t
levelofsupport, uint8_t levelofdigitalsig, uint8_t elementcoercion) ";

%feature("docstring")
gdcm::network::ServiceClassApplicationInformation::Size "size_t
gdcm::network::ServiceClassApplicationInformation::Size() const ";

%feature("docstring")
gdcm::network::ServiceClassApplicationInformation::Write "const
std::ostream&
gdcm::network::ServiceClassApplicationInformation::Write(std::ostream
&os) const ";


// File: classgdcm_1_1ServiceClassUser.xml
%feature("docstring") gdcm::ServiceClassUser "

ServiceClassUser.

C++ includes: gdcmServiceClassUser.h ";

%feature("docstring")  gdcm::ServiceClassUser::ServiceClassUser "gdcm::ServiceClassUser::ServiceClassUser()

Construct a SCU with default: hostname = localhost

port = 104 ";

%feature("docstring")  gdcm::ServiceClassUser::~ServiceClassUser "gdcm::ServiceClassUser::~ServiceClassUser() ";

%feature("docstring")  gdcm::ServiceClassUser::GetAETitle "const
char* gdcm::ServiceClassUser::GetAETitle() const ";

%feature("docstring")  gdcm::ServiceClassUser::GetCalledAETitle "const char* gdcm::ServiceClassUser::GetCalledAETitle() const ";

%feature("docstring")  gdcm::ServiceClassUser::GetTimeout "double
gdcm::ServiceClassUser::GetTimeout() const ";

%feature("docstring")  gdcm::ServiceClassUser::InitializeConnection "bool gdcm::ServiceClassUser::InitializeConnection()

Will try to connect This will setup the actual timeout used during the
whole connection time. Need to call SetTimeout first ";

%feature("docstring")
gdcm::ServiceClassUser::IsPresentationContextAccepted "bool
gdcm::ServiceClassUser::IsPresentationContextAccepted(const
PresentationContext &pc) const

Return if the passed in presentation was accepted during association
negotiation. ";

%feature("docstring")  gdcm::ServiceClassUser::SendEcho "bool
gdcm::ServiceClassUser::SendEcho()

C-ECHO. ";

%feature("docstring")  gdcm::ServiceClassUser::SendFind "bool
gdcm::ServiceClassUser::SendFind(const BaseRootQuery *query,
std::vector< DataSet > &retDatasets)

C-FIND a query, return result are in retDatasets. ";

%feature("docstring")  gdcm::ServiceClassUser::SendMove "bool
gdcm::ServiceClassUser::SendMove(const BaseRootQuery *query, const
char *outputdir)

Execute a C-MOVE, based on query, return files are written in
outputdir. ";

%feature("docstring")  gdcm::ServiceClassUser::SendMove "bool
gdcm::ServiceClassUser::SendMove(const BaseRootQuery *query,
std::vector< DataSet > &retDatasets)

Execute a C-MOVE, based on query, returned dataset are Implicit. ";

%feature("docstring")  gdcm::ServiceClassUser::SendMove "bool
gdcm::ServiceClassUser::SendMove(const BaseRootQuery *query,
std::vector< File > &retFile)

Execute a C-MOVE, based on query, returned Files are stored in vector.
";

%feature("docstring")  gdcm::ServiceClassUser::SendStore "bool
gdcm::ServiceClassUser::SendStore(const char *filename)

Execute a C-STORE on file on disk, named filename. ";

%feature("docstring")  gdcm::ServiceClassUser::SendStore "bool
gdcm::ServiceClassUser::SendStore(File const &file)

Execute a C-STORE on a File, the transfer syntax used for the query is
based on the file. ";

%feature("docstring")  gdcm::ServiceClassUser::SendStore "bool
gdcm::ServiceClassUser::SendStore(DataSet const &ds)

Execute a C-STORE on a DataSet, the transfer syntax used will be
Implicit. ";

%feature("docstring")  gdcm::ServiceClassUser::SetAETitle "void
gdcm::ServiceClassUser::SetAETitle(const char *aetitle)

set calling ae title ";

%feature("docstring")  gdcm::ServiceClassUser::SetCalledAETitle "void
gdcm::ServiceClassUser::SetCalledAETitle(const char *aetitle)

set called ae title ";

%feature("docstring")  gdcm::ServiceClassUser::SetHostname "void
gdcm::ServiceClassUser::SetHostname(const char *hostname)

Set the name of the called hostname (hostname or IP address) ";

%feature("docstring")  gdcm::ServiceClassUser::SetPort "void
gdcm::ServiceClassUser::SetPort(uint16_t port)

Set port of remote host (called application) ";

%feature("docstring")  gdcm::ServiceClassUser::SetPortSCP "void
gdcm::ServiceClassUser::SetPortSCP(uint16_t portscp)

Set the port for any incoming C-STORE-SCP operation (typically in a
return of C-MOVE) ";

%feature("docstring")  gdcm::ServiceClassUser::SetPresentationContexts
"void gdcm::ServiceClassUser::SetPresentationContexts(std::vector<
PresentationContext > const &pcs)

Set the Presentation Context used for the Association. ";

%feature("docstring")  gdcm::ServiceClassUser::SetTimeout "void
gdcm::ServiceClassUser::SetTimeout(double t)

set/get Timeout ";

%feature("docstring")  gdcm::ServiceClassUser::StartAssociation "bool
gdcm::ServiceClassUser::StartAssociation()

Start the association. Need to call SetPresentationContexts before. ";

%feature("docstring")  gdcm::ServiceClassUser::StopAssociation "bool
gdcm::ServiceClassUser::StopAssociation()

Stop the running association. ";


// File: classstd_1_1set.xml
%feature("docstring") std::set "

STL class. ";


// File: classgdcm_1_1SHA1.xml
%feature("docstring") gdcm::SHA1 "

Class for SHA1.

WARNING:  this class is able to pick from one implementation:

the one from OpenSSL (when GDCM_USE_SYSTEM_OPENSSL is turned ON)

In all other cases it will return an error

C++ includes: gdcmSHA1.h ";

%feature("docstring")  gdcm::SHA1::SHA1 "gdcm::SHA1::SHA1() ";

%feature("docstring")  gdcm::SHA1::~SHA1 "gdcm::SHA1::~SHA1() ";


// File: classgdcm_1_1SimpleMemberCommand.xml
%feature("docstring") gdcm::SimpleMemberCommand "

Command subclass that calls a pointer to a member function.

SimpleMemberCommand calls a pointer to a member function with no
arguments.

C++ includes: gdcmCommand.h ";

%feature("docstring")  gdcm::SimpleMemberCommand::Execute "virtual
void gdcm::SimpleMemberCommand< T >::Execute(Subject *, const Event &)

Invoke the callback function. ";

%feature("docstring")  gdcm::SimpleMemberCommand::Execute "virtual
void gdcm::SimpleMemberCommand< T >::Execute(const Subject *, const
Event &)

Abstract method that defines the action to be taken by the command.
This variant is expected to be used when requests comes from a const
Object ";

%feature("docstring")  gdcm::SimpleMemberCommand::SetCallbackFunction
"void gdcm::SimpleMemberCommand< T >::SetCallbackFunction(T *object,
TMemberFunctionPointer memberFunction)

Specify the callback function. ";


// File: classgdcm_1_1SimpleSubjectWatcher.xml
%feature("docstring") gdcm::SimpleSubjectWatcher "

SimpleSubjectWatcher This is a typical Subject Watcher class. It will
observe all events.

C++ includes: gdcmSimpleSubjectWatcher.h ";

%feature("docstring")
gdcm::SimpleSubjectWatcher::SimpleSubjectWatcher "gdcm::SimpleSubjectWatcher::SimpleSubjectWatcher(Subject *s, const
char *comment=\"\") ";

%feature("docstring")
gdcm::SimpleSubjectWatcher::~SimpleSubjectWatcher "virtual
gdcm::SimpleSubjectWatcher::~SimpleSubjectWatcher() ";


// File: classstd_1_1smart__ptr.xml
%feature("docstring") std::smart_ptr "

STL class. ";


// File: classgdcm_1_1SmartPointer.xml
%feature("docstring") gdcm::SmartPointer "

Class for Smart Pointer.

Will only work for subclass of gdcm::Object See tr1/shared_ptr for a
more general approach (not invasive) #include <tr1/memory> {
shared_ptr<Bla> b(new Bla); } Class partly based on post by Bill
Hubauer:http://groups.google.com/group/comp.lang.c++/msg/173ddc38a827a930

See:  http://www.davethehat.com/articles/smartp.htm  and
itk::SmartPointer

C++ includes: gdcmSmartPointer.h ";

%feature("docstring")  gdcm::SmartPointer::SmartPointer "gdcm::SmartPointer< ObjectType >::SmartPointer() ";

%feature("docstring")  gdcm::SmartPointer::SmartPointer "gdcm::SmartPointer< ObjectType >::SmartPointer(const SmartPointer<
ObjectType > &p) ";

%feature("docstring")  gdcm::SmartPointer::SmartPointer "gdcm::SmartPointer< ObjectType >::SmartPointer(ObjectType *p) ";

%feature("docstring")  gdcm::SmartPointer::SmartPointer "gdcm::SmartPointer< ObjectType >::SmartPointer(ObjectType const &p) ";

%feature("docstring")  gdcm::SmartPointer::~SmartPointer "gdcm::SmartPointer< ObjectType >::~SmartPointer() ";

%feature("docstring")  gdcm::SmartPointer::GetPointer "ObjectType*
gdcm::SmartPointer< ObjectType >::GetPointer() const

Explicit function to retrieve the pointer. ";


// File: classgdcm_1_1network_1_1SOPClassExtendedNegociationSub.xml
%feature("docstring") gdcm::network::SOPClassExtendedNegociationSub "

SOPClassExtendedNegociationSub PS 3.7 Table D.3-11 SOP CLASS EXTENDED
NEGOTIATION SUB-ITEM FIELDS (A-ASSOCIATE-RQ and A-ASSOCIATE-AC)

C++ includes: gdcmSOPClassExtendedNegociationSub.h ";

%feature("docstring")
gdcm::network::SOPClassExtendedNegociationSub::SOPClassExtendedNegociationSub
"gdcm::network::SOPClassExtendedNegociationSub::SOPClassExtendedNegociationSub()
";

%feature("docstring")
gdcm::network::SOPClassExtendedNegociationSub::Print "void
gdcm::network::SOPClassExtendedNegociationSub::Print(std::ostream &os)
const ";

%feature("docstring")
gdcm::network::SOPClassExtendedNegociationSub::Read "std::istream&
gdcm::network::SOPClassExtendedNegociationSub::Read(std::istream &is)
";

%feature("docstring")
gdcm::network::SOPClassExtendedNegociationSub::SetTuple "void
gdcm::network::SOPClassExtendedNegociationSub::SetTuple(const char
*uid, uint8_t levelofsupport=3, uint8_t levelofdigitalsig=0, uint8_t
elementcoercion=2) ";

%feature("docstring")
gdcm::network::SOPClassExtendedNegociationSub::Size "size_t
gdcm::network::SOPClassExtendedNegociationSub::Size() const ";

%feature("docstring")
gdcm::network::SOPClassExtendedNegociationSub::Write "const
std::ostream&
gdcm::network::SOPClassExtendedNegociationSub::Write(std::ostream &os)
const ";


// File: classgdcm_1_1SOPClassUIDToIOD.xml
%feature("docstring") gdcm::SOPClassUIDToIOD "

Class convert a class SOP Class UID into IOD.

Reference PS 3.4 Table B.5-1 STANDARD SOP CLASSES

C++ includes: gdcmSOPClassUIDToIOD.h ";


// File: classgdcm_1_1Sorter.xml
%feature("docstring") gdcm::Sorter "

Sorter General class to do sorting using a custom function You simply
need to provide a function of type: Sorter::SortFunction.

WARNING:  implementation details. For now there is no cache mechanism.
Which means that everytime you call Sort, all files specified as input
paramater are read

See:   Scanner

C++ includes: gdcmSorter.h ";

%feature("docstring")  gdcm::Sorter::Sorter "gdcm::Sorter::Sorter()
";

%feature("docstring")  gdcm::Sorter::~Sorter "virtual
gdcm::Sorter::~Sorter() ";

%feature("docstring")  gdcm::Sorter::AddSelect "bool
gdcm::Sorter::AddSelect(Tag const &tag, const char *value)

UNSUPPORTED FOR NOW. ";

%feature("docstring")  gdcm::Sorter::GetFilenames "const
std::vector<std::string>& gdcm::Sorter::GetFilenames() const

Return the list of filenames as sorted by the specific algorithm used.
Empty by default (before Sort() is called) ";

%feature("docstring")  gdcm::Sorter::Print "void
gdcm::Sorter::Print(std::ostream &os) const

Print. ";

%feature("docstring")  gdcm::Sorter::SetSortFunction "void
gdcm::Sorter::SetSortFunction(SortFunction f) ";

%feature("docstring")  gdcm::Sorter::Sort "virtual bool
gdcm::Sorter::Sort(std::vector< std::string > const &filenames)

Typically the output of Directory::GetFilenames() ";

%feature("docstring")  gdcm::Sorter::StableSort "virtual bool
gdcm::Sorter::StableSort(std::vector< std::string > const &filenames)
";


// File: classgdcm_1_1Spacing.xml
%feature("docstring") gdcm::Spacing "

Class for Spacing.

It all began with a mail to WG6:

Subject: Imager Pixel Spacing vs Pixel Spacing Body: [Apologies for
the duplicate post, namely to David Clunie & OFFIS team]

I have been trying to understand CP-586 in the following two cases:

On the one hand: DISCIMG/IMAGES/CRIMAGE taken
fromhttp://dclunie.com/images/pixelspacingtestimages.zip

And on the other hand:
http://gdcm.sourceforge.net/thingies/cr_pixelspacing.dcm

If I understand correctly the CP, one is required to use Pixel Spacing
for measurement ('true size' print) instead of Imager Pixel Spacing,
since the two attributes are present and Pixel Spacing is different
from Imager Pixel Spacing.

If this is correct, then the test data DISCIMG/IMAGES/CRIMAGE is
incorrect. If this is incorrect (ie. I need to use Imager Pixel
Spacing), then the display of cr_pixelspacing.dcm for measurement will
be incorrect.

Could someone please let me know what am I missing here? I could not
find any information in any header that would allow me to
differentiate those.

Thank you for your time,

Ref:http://lists.nema.org/scripts/lyris.pl?sub=488573&id=400720477 See
PS 3.3-2008, Table C.7-11b IMAGE PIXEL MACRO ATTRIBUTES

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

Ref:http://gdcm.sourceforge.net/wiki/index.php/Imager_Pixel_Spacing

C++ includes: gdcmSpacing.h ";

%feature("docstring")  gdcm::Spacing::Spacing "gdcm::Spacing::Spacing() ";

%feature("docstring")  gdcm::Spacing::~Spacing "gdcm::Spacing::~Spacing() ";


// File: classgdcm_1_1Spectroscopy.xml
%feature("docstring") gdcm::Spectroscopy "

Spectroscopy class.

C++ includes: gdcmSpectroscopy.h ";

%feature("docstring")  gdcm::Spectroscopy::Spectroscopy "gdcm::Spectroscopy::Spectroscopy() ";


// File: classgdcm_1_1SplitMosaicFilter.xml
%feature("docstring") gdcm::SplitMosaicFilter "

SplitMosaicFilter class Class to reshuffle bytes for a SIEMENS Mosaic
image Siemens CSA Image Header CSA:= Common Siemens Architecture,
sometimes also known as Common syngo Architecture.

C++ includes: gdcmSplitMosaicFilter.h ";

%feature("docstring")  gdcm::SplitMosaicFilter::SplitMosaicFilter "gdcm::SplitMosaicFilter::SplitMosaicFilter() ";

%feature("docstring")  gdcm::SplitMosaicFilter::~SplitMosaicFilter "gdcm::SplitMosaicFilter::~SplitMosaicFilter() ";

%feature("docstring")
gdcm::SplitMosaicFilter::ComputeMOSAICDimensions "bool
gdcm::SplitMosaicFilter::ComputeMOSAICDimensions(unsigned int dims[3])

Compute the new dimensions according to private information stored in
the MOSAIC header. ";

%feature("docstring")  gdcm::SplitMosaicFilter::GetFile "File&
gdcm::SplitMosaicFilter::GetFile() ";

%feature("docstring")  gdcm::SplitMosaicFilter::GetFile "const File&
gdcm::SplitMosaicFilter::GetFile() const ";

%feature("docstring")  gdcm::SplitMosaicFilter::GetImage "const
Image& gdcm::SplitMosaicFilter::GetImage() const ";

%feature("docstring")  gdcm::SplitMosaicFilter::GetImage "Image&
gdcm::SplitMosaicFilter::GetImage() ";

%feature("docstring")  gdcm::SplitMosaicFilter::SetFile "void
gdcm::SplitMosaicFilter::SetFile(const File &f) ";

%feature("docstring")  gdcm::SplitMosaicFilter::SetImage "void
gdcm::SplitMosaicFilter::SetImage(const Image &image) ";

%feature("docstring")  gdcm::SplitMosaicFilter::Split "bool
gdcm::SplitMosaicFilter::Split()

Split the SIEMENS MOSAIC image. ";


// File: classstd_1_1stack.xml
%feature("docstring") std::stack "

STL class. ";


// File: classgdcm_1_1StartEvent.xml
%feature("docstring") gdcm::StartEvent "C++ includes: gdcmEvent.h ";


// File: structgdcm_1_1static__assert__test.xml
%feature("docstring") gdcm::static_assert_test "C++ includes:
gdcmStaticAssert.h ";


// File: structgdcm_1_1STATIC__ASSERTION__FAILURE_3_01true_01_4.xml
%feature("docstring") gdcm::STATIC_ASSERTION_FAILURE< true > " C++
includes: gdcmStaticAssert.h ";


// File: classgdcm_1_1StreamImageReader.xml
%feature("docstring") gdcm::StreamImageReader "

StreamImageReader.

its role is to convert the DICOM DataSet into a Image representation
via an ITK streaming (ie, multithreaded) interface Image is different
from Pixmap has it has a position and a direction in Space. Currently,
this class is thread safe in that it can read a single extent in a
single thread. Multiple versions can be used for multiple
extents/threads.

See:   Image

C++ includes: gdcmStreamImageReader.h ";

%feature("docstring")  gdcm::StreamImageReader::StreamImageReader "gdcm::StreamImageReader::StreamImageReader() ";

%feature("docstring")  gdcm::StreamImageReader::~StreamImageReader "virtual gdcm::StreamImageReader::~StreamImageReader() ";

%feature("docstring")  gdcm::StreamImageReader::CanReadImage "bool
gdcm::StreamImageReader::CanReadImage() const

Only RAW images are currently readable by the stream reader. As more
streaming codecs are added, then this function will be updated to
reflect those changes. Calling this function prior to reading will
ensure that only streamable files are streamed. Make sure to call
ReadImageInformation prior to calling this function. ";

%feature("docstring")  gdcm::StreamImageReader::DefinePixelExtent "void gdcm::StreamImageReader::DefinePixelExtent(uint16_t inXMin,
uint16_t inXMax, uint16_t inYMin, uint16_t inYMax, uint16_t inZMin=0,
uint16_t inZMax=1)

Defines an image extent for the Read function. DICOM states that an
image can have no more than 2^16 pixels per edge (as of 2009) In this
case, the pixel extents ignore the direction cosines entirely, and
assumes that the origin of the image is at location 0,0 (regardless of
the definition in space per the tags). So, if the first 100 pixels of
the first row are to be read in, this function should be called with
DefinePixelExtent(0, 100, 0, 1), regardless of pixel size or
orientation. ";

%feature("docstring")
gdcm::StreamImageReader::DefineProperBufferLength "uint32_t
gdcm::StreamImageReader::DefineProperBufferLength() const

Paying attention to the pixel format and so forth, define the proper
buffer length for the user. The return amount is in bytes. Call this
function to determine the size of the char* buffer that will need to
be passed in to ReadImageSubregion(). If the return is 0, then that
means that the pixel extent was not defined prior ";

%feature("docstring")
gdcm::StreamImageReader::GetDimensionsValueForResolution "std::vector<unsigned int>
gdcm::StreamImageReader::GetDimensionsValueForResolution(unsigned int)
";

%feature("docstring")  gdcm::StreamImageReader::GetFile "File const&
gdcm::StreamImageReader::GetFile() const

Returns the dataset read by ReadImageInformation Couple this with the
ImageHelper to get statistics about the image, like pixel extent, to
be able to initialize buffers for reading ";

%feature("docstring")  gdcm::StreamImageReader::Read "bool
gdcm::StreamImageReader::Read(char *inReadBuffer, const std::size_t
&inBufferLength)

Read the DICOM image. There are three reasons for failure: The extent
is not set

the conversion from char* to std::ostream (internally) fails

the given buffer isn't large enough to accommodate the desired pixel
extent. This method has been implemented to look similar to the
metaimageio in itk MUST have an extent defined, or else Read will
return false. If no particular extent is required, use ImageReader
instead. ";

%feature("docstring")  gdcm::StreamImageReader::ReadImageInformation "virtual bool gdcm::StreamImageReader::ReadImageInformation()

Set the spacing and dimension information for the set filename.
returns false if the file is not initialized or not an image, with the
pixel (7fe0,0010) tag. ";

%feature("docstring")  gdcm::StreamImageReader::SetFileName "void
gdcm::StreamImageReader::SetFileName(const char *inFileName)

One of either SetFileName or SetStream must be called prior to any
other functions. These initialize an internal Reader class to be able
to get non-pixel image information. ";

%feature("docstring")  gdcm::StreamImageReader::SetStream "void
gdcm::StreamImageReader::SetStream(std::istream &inStream) ";


// File: classgdcm_1_1StreamImageWriter.xml
%feature("docstring") gdcm::StreamImageWriter "

StreamImageReader.

its role is to convert the DICOM DataSet into a Image representation
via an ITK streaming (ie, multithreaded) interface Image is different
from Pixmap has it has a position and a direction in Space. Currently,
this class is threadsafe in that it can read a single extent in a
single thread. Multiple versions can be used for multiple
extents/threads.

See:   Image

C++ includes: gdcmStreamImageWriter.h ";

%feature("docstring")  gdcm::StreamImageWriter::StreamImageWriter "gdcm::StreamImageWriter::StreamImageWriter() ";

%feature("docstring")  gdcm::StreamImageWriter::~StreamImageWriter "virtual gdcm::StreamImageWriter::~StreamImageWriter() ";

%feature("docstring")  gdcm::StreamImageWriter::CanWriteFile "bool
gdcm::StreamImageWriter::CanWriteFile() const

This function determines if a file can even be written using the
streaming writer unlike the reader, can be called before
WriteImageInformation, but must be called after SetFile. ";

%feature("docstring")  gdcm::StreamImageWriter::DefinePixelExtent "void gdcm::StreamImageWriter::DefinePixelExtent(uint16_t inXMin,
uint16_t inXMax, uint16_t inYMin, uint16_t inYMax, uint16_t inZMin=0,
uint16_t inZMax=1)

Defines an image extent for the Read function. DICOM states that an
image can have no more than 2^16 pixels per edge (as of 2009) In this
case, the pixel extents ignore the direction cosines entirely, and
assumes that the origin of the image is at location 0,0 (regardless of
the definition in space per the tags). So, if the first 100 pixels of
the first row are to be read in, this function should be called with
DefinePixelExtent(0, 100, 0, 1), regardless of pixel size or
orientation. 15 nov 2010: added z dimension, defaults to being 1 plane
large ";

%feature("docstring")
gdcm::StreamImageWriter::DefineProperBufferLength "uint32_t
gdcm::StreamImageWriter::DefineProperBufferLength()

Paying attention to the pixel format and so forth, define the proper
buffer length for the user. The return amount is in bytes. If the
return is 0, then that means that the pixel extent was not defined
prior this return is for RAW inputs which are then encoded by the
writer, but are used to ensure that the writer gets the proper buffer
size ";

%feature("docstring")  gdcm::StreamImageWriter::SetFile "void
gdcm::StreamImageWriter::SetFile(const File &inFile)

Set the image information to be written to disk that is everything but
the pixel information: (7fe0,0010) PixelData ";

%feature("docstring")  gdcm::StreamImageWriter::SetFileName "void
gdcm::StreamImageWriter::SetFileName(const char *inFileName)

One of either SetFileName or SetStream must be called prior to any
other functions. These initialize an internal Reader class to be able
to get non-pixel image information. ";

%feature("docstring")  gdcm::StreamImageWriter::SetStream "void
gdcm::StreamImageWriter::SetStream(std::ostream &inStream) ";

%feature("docstring")  gdcm::StreamImageWriter::Write "bool
gdcm::StreamImageWriter::Write(void *inWriteBuffer, const std::size_t
&inBufferLength)

Read the DICOM image. There are three reasons for failure: The extent
is not set

the conversion from void* to std::ostream (internally) fails

the given buffer isn't large enough to accomodate the desired pixel
extent. This method has been implemented to look similar to the
metaimageio in itk MUST have an extent defined, or else Read will
return false. If no particular extent is required, use ImageReader
instead. ";

%feature("docstring")  gdcm::StreamImageWriter::WriteImageInformation
"virtual bool gdcm::StreamImageWriter::WriteImageInformation()

Write the header information to disk, and a bunch of zeros for the
actual pixel information Of course, if we're doing a non-compressed
format, that works but if it's compressed, we have to force the
ordering of chunks that are written. ";


// File: classgdcm_1_1String.xml
%feature("docstring") gdcm::String "

String.

TDelimiter template parameter is used to separate multiple String (VM1
>) TMaxLength is only a hint. Noone actually respect the max length
TPadChar is the string padding (0 or space)

C++ includes: gdcmString.h ";

%feature("docstring")  gdcm::String::String "gdcm::String<
TDelimiter, TMaxLength, TPadChar >::String()

String constructors. ";

%feature("docstring")  gdcm::String::String "gdcm::String<
TDelimiter, TMaxLength, TPadChar >::String(const value_type *s) ";

%feature("docstring")  gdcm::String::String "gdcm::String<
TDelimiter, TMaxLength, TPadChar >::String(const value_type *s,
size_type n) ";

%feature("docstring")  gdcm::String::String "gdcm::String<
TDelimiter, TMaxLength, TPadChar >::String(const std::string &s,
size_type pos=0, size_type n=npos) ";

%feature("docstring")  gdcm::String::IsValid "bool gdcm::String<
TDelimiter, TMaxLength, TPadChar >::IsValid() const

return if string is valid ";

%feature("docstring")  gdcm::String::Trim "std::string gdcm::String<
TDelimiter, TMaxLength, TPadChar >::Trim() const

Trim function is required to return a std::string object, otherwise we
could not create a gdcm::String object with an odd number of bytes...
";

%feature("docstring")  gdcm::String::Truncate "gdcm::String<TDelimiter, TMaxLength, TPadChar> gdcm::String<
TDelimiter, TMaxLength, TPadChar >::Truncate() const ";


// File: classstd_1_1string.xml
%feature("docstring") std::string "

STL class. ";


// File: classgdcm_1_1StringFilter.xml
%feature("docstring") gdcm::StringFilter "

StringFilter StringFilter is the class that make gdcm2.x looks more
like gdcm1 and transform the binary blob contained in a DataElement
into a string, typically this is a nice feature to have for wrapped
language.

C++ includes: gdcmStringFilter.h ";

%feature("docstring")  gdcm::StringFilter::StringFilter "gdcm::StringFilter::StringFilter() ";

%feature("docstring")  gdcm::StringFilter::~StringFilter "gdcm::StringFilter::~StringFilter() ";

%feature("docstring")  gdcm::StringFilter::ExecuteQuery "bool
gdcm::StringFilter::ExecuteQuery(std::string const &query, std::string
&value) const

Execute the XPATH query to find a value (as string) return false when
attribute is not found (or an error in the XPATH query) You need to
make sure that your XPATH query is syntatically correct ";

%feature("docstring")  gdcm::StringFilter::FromString "std::string
gdcm::StringFilter::FromString(const Tag &t, const char *value, VL
const &vl) ";

%feature("docstring")  gdcm::StringFilter::FromString "std::string
gdcm::StringFilter::FromString(const Tag &t, const char *value, size_t
len)

Convert to string the char array defined by the pair (value,len) ";

%feature("docstring")  gdcm::StringFilter::GetFile "File&
gdcm::StringFilter::GetFile() ";

%feature("docstring")  gdcm::StringFilter::GetFile "const File&
gdcm::StringFilter::GetFile() const ";

%feature("docstring")  gdcm::StringFilter::SetDicts "void
gdcm::StringFilter::SetDicts(const Dicts &dicts)

Allow user to pass in there own dicts. ";

%feature("docstring")  gdcm::StringFilter::SetFile "void
gdcm::StringFilter::SetFile(const File &f)

Set/Get File. ";

%feature("docstring")  gdcm::StringFilter::ToString "std::string
gdcm::StringFilter::ToString(const DataElement &de) const

Convert to string the ByteValue contained in a DataElement. The
DataElement must be coming from the actual DataSet associated with
File (see SetFile). ";

%feature("docstring")  gdcm::StringFilter::ToString "std::string
gdcm::StringFilter::ToString(const Tag &t) const

Directly from a Tag: ";

%feature("docstring")  gdcm::StringFilter::ToStringPair "std::pair<std::string, std::string>
gdcm::StringFilter::ToStringPair(const DataElement &de) const

Convert to string the ByteValue contained in a DataElement the
returned elements are: pair.first : the name as found in the
dictionary of DataElement pari.second : the value encoded into a
string (US,UL...) are properly converted ";

%feature("docstring")  gdcm::StringFilter::ToStringPair "std::pair<std::string, std::string>
gdcm::StringFilter::ToStringPair(const Tag &t) const

Directly from a Tag: ";

%feature("docstring")  gdcm::StringFilter::UseDictAlways "void
gdcm::StringFilter::UseDictAlways(bool) ";


// File: classstd_1_1stringstream.xml
%feature("docstring") std::stringstream "

STL class. ";


// File: classgdcm_1_1Study.xml
%feature("docstring") gdcm::Study "

Study.

C++ includes: gdcmStudy.h ";

%feature("docstring")  gdcm::Study::Study "gdcm::Study::Study() ";


// File: classgdcm_1_1Subject.xml
%feature("docstring") gdcm::Subject "

Subject.

See:   Command Event

C++ includes: gdcmSubject.h ";

%feature("docstring")  gdcm::Subject::Subject "gdcm::Subject::Subject() ";

%feature("docstring")  gdcm::Subject::~Subject "gdcm::Subject::~Subject() ";

%feature("docstring")  gdcm::Subject::AddObserver "unsigned long
gdcm::Subject::AddObserver(const Event &event, Command *)

Allow people to add/remove/invoke observers (callbacks) to any GDCM
object. This is an implementation of the subject/observer design
pattern. An observer is added by specifying an event to respond to and
an gdcm::Command to execute. It returns an unsigned long tag which can
be used later to remove the event or retrieve the command. The memory
for the Command becomes the responsibility of this object, so don't
pass the same instance of a command to two different objects ";

%feature("docstring")  gdcm::Subject::AddObserver "unsigned long
gdcm::Subject::AddObserver(const Event &event, Command *) const ";

%feature("docstring")  gdcm::Subject::GetCommand "Command*
gdcm::Subject::GetCommand(unsigned long tag)

Get the command associated with the given tag. NOTE: This returns a
pointer to a Command, but it is safe to asign this to a
Command::Pointer. Since Command inherits from LightObject, at this
point in the code, only a pointer or a reference to the Command can be
used. ";

%feature("docstring")  gdcm::Subject::HasObserver "bool
gdcm::Subject::HasObserver(const Event &event) const

Return true if an observer is registered for this event. ";

%feature("docstring")  gdcm::Subject::InvokeEvent "void
gdcm::Subject::InvokeEvent(const Event &)

Call Execute on all the Commands observing this event id. ";

%feature("docstring")  gdcm::Subject::InvokeEvent "void
gdcm::Subject::InvokeEvent(const Event &) const

Call Execute on all the Commands observing this event id. The actions
triggered by this call doesn't modify this object. ";

%feature("docstring")  gdcm::Subject::RemoveAllObservers "void
gdcm::Subject::RemoveAllObservers()

Remove all observers . ";

%feature("docstring")  gdcm::Subject::RemoveObserver "void
gdcm::Subject::RemoveObserver(unsigned long tag)

Remove the observer with this tag value. ";


// File: classgdcm_1_1Surface.xml
%feature("docstring") gdcm::Surface "

This class defines a SURFACE IE. This members are taken from required
surface mesh module attributes.

See:  PS 3.3 A.1.2.18 , A.57 and C.27

C++ includes: gdcmSurface.h ";

%feature("docstring")  gdcm::Surface::Surface "gdcm::Surface::Surface() ";

%feature("docstring")  gdcm::Surface::~Surface "virtual
gdcm::Surface::~Surface() ";

%feature("docstring")  gdcm::Surface::GetAlgorithmFamily "SegmentHelper::BasicCodedEntry const&
gdcm::Surface::GetAlgorithmFamily() const ";

%feature("docstring")  gdcm::Surface::GetAlgorithmFamily "SegmentHelper::BasicCodedEntry& gdcm::Surface::GetAlgorithmFamily() ";

%feature("docstring")  gdcm::Surface::GetAlgorithmName "const char*
gdcm::Surface::GetAlgorithmName() const ";

%feature("docstring")  gdcm::Surface::GetAlgorithmVersion "const
char* gdcm::Surface::GetAlgorithmVersion() const ";

%feature("docstring")  gdcm::Surface::GetAxisOfRotation "const float*
gdcm::Surface::GetAxisOfRotation() const

Pointer is null if undefined ";

%feature("docstring")  gdcm::Surface::GetCenterOfRotation "const
float* gdcm::Surface::GetCenterOfRotation() const

Pointer is null if undefined ";

%feature("docstring")  gdcm::Surface::GetFiniteVolume "STATES
gdcm::Surface::GetFiniteVolume() const ";

%feature("docstring")  gdcm::Surface::GetManifold "STATES
gdcm::Surface::GetManifold() const ";

%feature("docstring")  gdcm::Surface::GetMaximumPointDistance "float
gdcm::Surface::GetMaximumPointDistance() const ";

%feature("docstring")  gdcm::Surface::GetMeanPointDistance "float
gdcm::Surface::GetMeanPointDistance() const ";

%feature("docstring")  gdcm::Surface::GetMeshPrimitive "MeshPrimitive
const& gdcm::Surface::GetMeshPrimitive() const ";

%feature("docstring")  gdcm::Surface::GetMeshPrimitive "MeshPrimitive& gdcm::Surface::GetMeshPrimitive() ";

%feature("docstring")  gdcm::Surface::GetNumberOfSurfacePoints "unsigned long gdcm::Surface::GetNumberOfSurfacePoints() const ";

%feature("docstring")  gdcm::Surface::GetNumberOfVectors "unsigned
long gdcm::Surface::GetNumberOfVectors() const ";

%feature("docstring")  gdcm::Surface::GetPointCoordinatesData "const
DataElement& gdcm::Surface::GetPointCoordinatesData() const ";

%feature("docstring")  gdcm::Surface::GetPointCoordinatesData "DataElement& gdcm::Surface::GetPointCoordinatesData() ";

%feature("docstring")  gdcm::Surface::GetPointPositionAccuracy "const
float* gdcm::Surface::GetPointPositionAccuracy() const

Pointer is null if undefined ";

%feature("docstring")  gdcm::Surface::GetPointsBoundingBoxCoordinates
"const float* gdcm::Surface::GetPointsBoundingBoxCoordinates() const

Pointer is null if undefined ";

%feature("docstring")  gdcm::Surface::GetProcessingAlgorithm "SegmentHelper::BasicCodedEntry const&
gdcm::Surface::GetProcessingAlgorithm() const ";

%feature("docstring")  gdcm::Surface::GetProcessingAlgorithm "SegmentHelper::BasicCodedEntry&
gdcm::Surface::GetProcessingAlgorithm() ";

%feature("docstring")  gdcm::Surface::GetRecommendedDisplayCIELabValue
"const unsigned short*
gdcm::Surface::GetRecommendedDisplayCIELabValue() const ";

%feature("docstring")  gdcm::Surface::GetRecommendedDisplayCIELabValue
"unsigned short gdcm::Surface::GetRecommendedDisplayCIELabValue(const
unsigned int idx) const ";

%feature("docstring")
gdcm::Surface::GetRecommendedDisplayGrayscaleValue "unsigned short
gdcm::Surface::GetRecommendedDisplayGrayscaleValue() const ";

%feature("docstring")
gdcm::Surface::GetRecommendedPresentationOpacity "float
gdcm::Surface::GetRecommendedPresentationOpacity() const ";

%feature("docstring")  gdcm::Surface::GetRecommendedPresentationType "VIEWType gdcm::Surface::GetRecommendedPresentationType() const ";

%feature("docstring")  gdcm::Surface::GetSurfaceComments "const char*
gdcm::Surface::GetSurfaceComments() const ";

%feature("docstring")  gdcm::Surface::GetSurfaceNumber "unsigned long
gdcm::Surface::GetSurfaceNumber() const ";

%feature("docstring")  gdcm::Surface::GetSurfaceProcessing "bool
gdcm::Surface::GetSurfaceProcessing() const ";

%feature("docstring")  gdcm::Surface::GetSurfaceProcessingDescription
"const char* gdcm::Surface::GetSurfaceProcessingDescription() const
";

%feature("docstring")  gdcm::Surface::GetSurfaceProcessingRatio "float gdcm::Surface::GetSurfaceProcessingRatio() const ";

%feature("docstring")  gdcm::Surface::GetVectorAccuracy "const float*
gdcm::Surface::GetVectorAccuracy() const ";

%feature("docstring")  gdcm::Surface::GetVectorCoordinateData "const
DataElement& gdcm::Surface::GetVectorCoordinateData() const ";

%feature("docstring")  gdcm::Surface::GetVectorCoordinateData "DataElement& gdcm::Surface::GetVectorCoordinateData() ";

%feature("docstring")  gdcm::Surface::GetVectorDimensionality "unsigned short gdcm::Surface::GetVectorDimensionality() const ";

%feature("docstring")  gdcm::Surface::SetAlgorithmFamily "void
gdcm::Surface::SetAlgorithmFamily(SegmentHelper::BasicCodedEntry const
&BSE) ";

%feature("docstring")  gdcm::Surface::SetAlgorithmName "void
gdcm::Surface::SetAlgorithmName(const char *str) ";

%feature("docstring")  gdcm::Surface::SetAlgorithmVersion "void
gdcm::Surface::SetAlgorithmVersion(const char *str) ";

%feature("docstring")  gdcm::Surface::SetAxisOfRotation "void
gdcm::Surface::SetAxisOfRotation(const float *axis) ";

%feature("docstring")  gdcm::Surface::SetCenterOfRotation "void
gdcm::Surface::SetCenterOfRotation(const float *center) ";

%feature("docstring")  gdcm::Surface::SetFiniteVolume "void
gdcm::Surface::SetFiniteVolume(STATES state) ";

%feature("docstring")  gdcm::Surface::SetManifold "void
gdcm::Surface::SetManifold(STATES state) ";

%feature("docstring")  gdcm::Surface::SetMaximumPointDistance "void
gdcm::Surface::SetMaximumPointDistance(float maximum) ";

%feature("docstring")  gdcm::Surface::SetMeanPointDistance "void
gdcm::Surface::SetMeanPointDistance(float average) ";

%feature("docstring")  gdcm::Surface::SetMeshPrimitive "void
gdcm::Surface::SetMeshPrimitive(MeshPrimitive &mp) ";

%feature("docstring")  gdcm::Surface::SetNumberOfSurfacePoints "void
gdcm::Surface::SetNumberOfSurfacePoints(const unsigned long nb) ";

%feature("docstring")  gdcm::Surface::SetNumberOfVectors "void
gdcm::Surface::SetNumberOfVectors(const unsigned long nb) ";

%feature("docstring")  gdcm::Surface::SetPointCoordinatesData "void
gdcm::Surface::SetPointCoordinatesData(DataElement const &de) ";

%feature("docstring")  gdcm::Surface::SetPointPositionAccuracy "void
gdcm::Surface::SetPointPositionAccuracy(const float *accuracies) ";

%feature("docstring")  gdcm::Surface::SetPointsBoundingBoxCoordinates
"void gdcm::Surface::SetPointsBoundingBoxCoordinates(const float
*coordinates) ";

%feature("docstring")  gdcm::Surface::SetProcessingAlgorithm "void
gdcm::Surface::SetProcessingAlgorithm(SegmentHelper::BasicCodedEntry
const &BSE) ";

%feature("docstring")  gdcm::Surface::SetRecommendedDisplayCIELabValue
"void gdcm::Surface::SetRecommendedDisplayCIELabValue(const unsigned
short vl[3]) ";

%feature("docstring")  gdcm::Surface::SetRecommendedDisplayCIELabValue
"void gdcm::Surface::SetRecommendedDisplayCIELabValue(const unsigned
short vl, const unsigned int idx=0) ";

%feature("docstring")  gdcm::Surface::SetRecommendedDisplayCIELabValue
"void gdcm::Surface::SetRecommendedDisplayCIELabValue(const
std::vector< unsigned short > &vl) ";

%feature("docstring")
gdcm::Surface::SetRecommendedDisplayGrayscaleValue "void
gdcm::Surface::SetRecommendedDisplayGrayscaleValue(const unsigned
short vl) ";

%feature("docstring")
gdcm::Surface::SetRecommendedPresentationOpacity "void
gdcm::Surface::SetRecommendedPresentationOpacity(const float opacity)
";

%feature("docstring")  gdcm::Surface::SetRecommendedPresentationType "void gdcm::Surface::SetRecommendedPresentationType(VIEWType type) ";

%feature("docstring")  gdcm::Surface::SetSurfaceComments "void
gdcm::Surface::SetSurfaceComments(const char *comment) ";

%feature("docstring")  gdcm::Surface::SetSurfaceNumber "void
gdcm::Surface::SetSurfaceNumber(const unsigned long nb) ";

%feature("docstring")  gdcm::Surface::SetSurfaceProcessing "void
gdcm::Surface::SetSurfaceProcessing(bool b) ";

%feature("docstring")  gdcm::Surface::SetSurfaceProcessingDescription
"void gdcm::Surface::SetSurfaceProcessingDescription(const char
*description) ";

%feature("docstring")  gdcm::Surface::SetSurfaceProcessingRatio "void
gdcm::Surface::SetSurfaceProcessingRatio(const float ratio) ";

%feature("docstring")  gdcm::Surface::SetVectorAccuracy "void
gdcm::Surface::SetVectorAccuracy(const float *accuracy) ";

%feature("docstring")  gdcm::Surface::SetVectorCoordinateData "void
gdcm::Surface::SetVectorCoordinateData(DataElement const &de) ";

%feature("docstring")  gdcm::Surface::SetVectorDimensionality "void
gdcm::Surface::SetVectorDimensionality(const unsigned short dim) ";


// File: classgdcm_1_1SurfaceHelper.xml
%feature("docstring") gdcm::SurfaceHelper "

SurfaceHelper Helper class for Surface object.

C++ includes: gdcmSurfaceHelper.h ";


// File: classgdcm_1_1SurfaceReader.xml
%feature("docstring") gdcm::SurfaceReader "

This class defines a SURFACE IE reader. It reads surface mesh module
attributes.

See:  PS 3.3 A.1.2.18 , A.57 and C.27

C++ includes: gdcmSurfaceReader.h ";

%feature("docstring")  gdcm::SurfaceReader::SurfaceReader "gdcm::SurfaceReader::SurfaceReader() ";

%feature("docstring")  gdcm::SurfaceReader::~SurfaceReader "virtual
gdcm::SurfaceReader::~SurfaceReader() ";

%feature("docstring")  gdcm::SurfaceReader::GetNumberOfSurfaces "unsigned long gdcm::SurfaceReader::GetNumberOfSurfaces() const ";

%feature("docstring")  gdcm::SurfaceReader::Read "virtual bool
gdcm::SurfaceReader::Read()

Read. ";


// File: classgdcm_1_1SurfaceWriter.xml
%feature("docstring") gdcm::SurfaceWriter "

This class defines a SURFACE IE writer. It writes surface mesh module
attributes.

See:  PS 3.3 A.1.2.18 , A.57 and C.27

C++ includes: gdcmSurfaceWriter.h ";

%feature("docstring")  gdcm::SurfaceWriter::SurfaceWriter "gdcm::SurfaceWriter::SurfaceWriter() ";

%feature("docstring")  gdcm::SurfaceWriter::~SurfaceWriter "virtual
gdcm::SurfaceWriter::~SurfaceWriter() ";

%feature("docstring")  gdcm::SurfaceWriter::GetNumberOfSurfaces "unsigned long gdcm::SurfaceWriter::GetNumberOfSurfaces() ";

%feature("docstring")  gdcm::SurfaceWriter::SetNumberOfSurfaces "void
gdcm::SurfaceWriter::SetNumberOfSurfaces(const unsigned long nb) ";

%feature("docstring")  gdcm::SurfaceWriter::Write "bool
gdcm::SurfaceWriter::Write()

Write. ";


// File: classgdcm_1_1SwapCode.xml
%feature("docstring") gdcm::SwapCode "

SwapCode representation.

C++ includes: gdcmSwapCode.h ";

%feature("docstring")  gdcm::SwapCode::SwapCode "gdcm::SwapCode::SwapCode(SwapCodeType sc=Unknown) ";


// File: classgdcm_1_1SwapperDoOp.xml
%feature("docstring") gdcm::SwapperDoOp "C++ includes: gdcmSwapper.h
";


// File: classgdcm_1_1SwapperNoOp.xml
%feature("docstring") gdcm::SwapperNoOp "C++ includes: gdcmSwapper.h
";


// File: classgdcm_1_1System.xml
%feature("docstring") gdcm::System "

Class to do system operation.

OS independent functionalities

C++ includes: gdcmSystem.h ";


// File: classstd_1_1system__error.xml
%feature("docstring") std::system_error "

STL class. ";


// File: classgdcm_1_1Table.xml
%feature("docstring") gdcm::Table "

Table.

C++ includes: gdcmTable.h ";

%feature("docstring")  gdcm::Table::Table "gdcm::Table::Table() ";

%feature("docstring")  gdcm::Table::~Table "gdcm::Table::~Table() ";

%feature("docstring")  gdcm::Table::GetTableEntry "const TableEntry&
gdcm::Table::GetTableEntry(const Tag &tag) const ";

%feature("docstring")  gdcm::Table::InsertEntry "void
gdcm::Table::InsertEntry(Tag const &tag, TableEntry const &te) ";


// File: classgdcm_1_1TableEntry.xml
%feature("docstring") gdcm::TableEntry "

TableEntry.

C++ includes: gdcmTableEntry.h ";

%feature("docstring")  gdcm::TableEntry::TableEntry "gdcm::TableEntry::TableEntry(const char *attribute=0, Type const
&type=Type(), const char *des=0) ";

%feature("docstring")  gdcm::TableEntry::~TableEntry "gdcm::TableEntry::~TableEntry() ";


// File: classgdcm_1_1TableReader.xml
%feature("docstring") gdcm::TableReader "

Class for representing a TableReader.

This class is an empty shell meant to be derived

C++ includes: gdcmTableReader.h ";

%feature("docstring")  gdcm::TableReader::TableReader "gdcm::TableReader::TableReader(Defs &defs) ";

%feature("docstring")  gdcm::TableReader::~TableReader "virtual
gdcm::TableReader::~TableReader() ";

%feature("docstring")  gdcm::TableReader::CharacterDataHandler "virtual void gdcm::TableReader::CharacterDataHandler(const char *data,
int length) ";

%feature("docstring")  gdcm::TableReader::EndElement "virtual void
gdcm::TableReader::EndElement(const char *name) ";

%feature("docstring")  gdcm::TableReader::GetDefs "const Defs&
gdcm::TableReader::GetDefs() const ";

%feature("docstring")  gdcm::TableReader::GetFilename "const char*
gdcm::TableReader::GetFilename() ";

%feature("docstring")  gdcm::TableReader::HandleIOD "void
gdcm::TableReader::HandleIOD(const char **atts) ";

%feature("docstring")  gdcm::TableReader::HandleIODEntry "void
gdcm::TableReader::HandleIODEntry(const char **atts) ";

%feature("docstring")  gdcm::TableReader::HandleMacro "void
gdcm::TableReader::HandleMacro(const char **atts) ";

%feature("docstring")  gdcm::TableReader::HandleMacroEntry "void
gdcm::TableReader::HandleMacroEntry(const char **atts) ";

%feature("docstring")  gdcm::TableReader::HandleMacroEntryDescription
"void gdcm::TableReader::HandleMacroEntryDescription(const char
**atts) ";

%feature("docstring")  gdcm::TableReader::HandleModule "void
gdcm::TableReader::HandleModule(const char **atts) ";

%feature("docstring")  gdcm::TableReader::HandleModuleEntry "void
gdcm::TableReader::HandleModuleEntry(const char **atts) ";

%feature("docstring")  gdcm::TableReader::HandleModuleEntryDescription
"void gdcm::TableReader::HandleModuleEntryDescription(const char
**atts) ";

%feature("docstring")  gdcm::TableReader::HandleModuleInclude "void
gdcm::TableReader::HandleModuleInclude(const char **atts) ";

%feature("docstring")  gdcm::TableReader::Read "int
gdcm::TableReader::Read() ";

%feature("docstring")  gdcm::TableReader::SetFilename "void
gdcm::TableReader::SetFilename(const char *filename) ";

%feature("docstring")  gdcm::TableReader::StartElement "virtual void
gdcm::TableReader::StartElement(const char *name, const char **atts)
";


// File: classgdcm_1_1network_1_1TableRow.xml
%feature("docstring") gdcm::network::TableRow "C++ includes:
gdcmULTransitionTable.h ";

%feature("docstring")  gdcm::network::TableRow::TableRow "gdcm::network::TableRow::TableRow() ";

%feature("docstring")  gdcm::network::TableRow::~TableRow "gdcm::network::TableRow::~TableRow() ";


// File: classgdcm_1_1Tag.xml
%feature("docstring") gdcm::Tag "

Class to represent a DICOM Data Element ( Attribute) Tag (Group,
Element). Basically an uint32_t which can also be expressed as two
uint16_t (group and element)

DATA ELEMENT TAG: A unique identifier for a Data Element composed of
an ordered pair of numbers (a Group Number followed by an Element
Number). GROUP NUMBER: The first number in the ordered pair of numbers
that makes up a Data Element Tag. ELEMENT NUMBER: The second number in
the ordered pair of numbers that makes up a Data Element Tag.

C++ includes: gdcmTag.h ";

%feature("docstring")  gdcm::Tag::Tag "gdcm::Tag::Tag(uint16_t group,
uint16_t element)

Constructor with 2*uint16_t. ";

%feature("docstring")  gdcm::Tag::Tag "gdcm::Tag::Tag(uint32_t tag=0)

Constructor with 1*uint32_t Prefer the cstor that takes two uint16_t.
";

%feature("docstring")  gdcm::Tag::Tag "gdcm::Tag::Tag(const Tag
&_val) ";

%feature("docstring")  gdcm::Tag::GetElement "uint16_t
gdcm::Tag::GetElement() const

Returns the ' Element number' of the given Tag. ";

%feature("docstring")  gdcm::Tag::GetElementTag "uint32_t
gdcm::Tag::GetElementTag() const

Returns the full tag value of the given Tag. ";

%feature("docstring")  gdcm::Tag::GetGroup "uint16_t
gdcm::Tag::GetGroup() const

Returns the 'Group number' of the given Tag. ";

%feature("docstring")  gdcm::Tag::GetLength "uint32_t
gdcm::Tag::GetLength() const

return the length of tag (read: size on disk) ";

%feature("docstring")  gdcm::Tag::GetPrivateCreator "Tag
gdcm::Tag::GetPrivateCreator() const

Return the Private Creator Data Element tag of a private data element.
";

%feature("docstring")  gdcm::Tag::IsGroupLength "bool
gdcm::Tag::IsGroupLength() const

return whether the tag correspond to a group length tag: ";

%feature("docstring")  gdcm::Tag::IsGroupXX "bool
gdcm::Tag::IsGroupXX(const Tag &t) const

e.g 6002,3000 belong to groupXX: 6000,3000 ";

%feature("docstring")  gdcm::Tag::IsIllegal "bool
gdcm::Tag::IsIllegal() const

return if the tag is considered to be an illegal tag ";

%feature("docstring")  gdcm::Tag::IsPrivate "bool
gdcm::Tag::IsPrivate() const

PRIVATE DATA ELEMENT: Additional Data Element, defined by an
implementor, to communicate information that is not contained in
Standard Data Elements. Private Data elements have odd Group Numbers.
";

%feature("docstring")  gdcm::Tag::IsPrivateCreator "bool
gdcm::Tag::IsPrivateCreator() const

Returns if tag is a Private Creator (xxxx,00yy), where xxxx is odd
number and yy in [0x10,0xFF] ";

%feature("docstring")  gdcm::Tag::IsPublic "bool
gdcm::Tag::IsPublic() const

STANDARD DATA ELEMENT: A Data Element defined in the DICOM Standard,
and therefore listed in the DICOM Data Element Dictionary in PS 3.6.
Is the Tag from the Public dict...well the implementation is buggy it
does not prove the element is indeed in the dict... ";

%feature("docstring")  gdcm::Tag::PrintAsContinuousString "std::string gdcm::Tag::PrintAsContinuousString() const

Print tag value with no separating comma: eg. tag = \"12345678\" It
comes in useful when reading tag values from XML file(in
NativeDICOMModel) ";

%feature("docstring")  gdcm::Tag::PrintAsContinuousUpperCaseString "std::string gdcm::Tag::PrintAsContinuousUpperCaseString() const

Same as PrintAsContinuousString, but hexadecimal [a-f] are printed
using upper case. ";

%feature("docstring")  gdcm::Tag::PrintAsPipeSeparatedString "std::string gdcm::Tag::PrintAsPipeSeparatedString() const

Print as a pipe separated string (GDCM 1.x compat only). Do not use in
newer code See:   ReadFromPipeSeparatedString ";

%feature("docstring")  gdcm::Tag::Read "std::istream&
gdcm::Tag::Read(std::istream &is)

Read a tag from binary representation. ";

%feature("docstring")  gdcm::Tag::ReadFromCommaSeparatedString "bool
gdcm::Tag::ReadFromCommaSeparatedString(const char *str)

Read from a comma separated string. This is a highly user oriented
function, the string should be formated as: 1234,5678 to specify the
tag (0x1234,0x5678) The notation comes from the DICOM standard, and is
handy to use from a command line program ";

%feature("docstring")  gdcm::Tag::ReadFromContinuousString "bool
gdcm::Tag::ReadFromContinuousString(const char *str)

Read From XML formatted tag value eg. tag = \"12345678\" It comes in
useful when reading tag values from XML file(in NativeDICOMModel) ";

%feature("docstring")  gdcm::Tag::ReadFromPipeSeparatedString "bool
gdcm::Tag::ReadFromPipeSeparatedString(const char *str)

Read from a pipe separated string (GDCM 1.x compat only). Do not use
in newer code See:   ReadFromCommaSeparatedString ";

%feature("docstring")  gdcm::Tag::SetElement "void
gdcm::Tag::SetElement(uint16_t element)

Sets the ' Element number' of the given Tag. ";

%feature("docstring")  gdcm::Tag::SetElementTag "void
gdcm::Tag::SetElementTag(uint16_t group, uint16_t element)

Sets the 'Group number' & ' Element number' of the given Tag. ";

%feature("docstring")  gdcm::Tag::SetElementTag "void
gdcm::Tag::SetElementTag(uint32_t tag)

Sets the full tag value of the given Tag. ";

%feature("docstring")  gdcm::Tag::SetGroup "void
gdcm::Tag::SetGroup(uint16_t group)

Sets the 'Group number' of the given Tag. ";

%feature("docstring")  gdcm::Tag::SetPrivateCreator "void
gdcm::Tag::SetPrivateCreator(Tag const &t)

Set private creator: ";

%feature("docstring")  gdcm::Tag::Write "const std::ostream&
gdcm::Tag::Write(std::ostream &os) const

Write a tag in binary rep. ";


// File: classgdcm_1_1TagPath.xml
%feature("docstring") gdcm::TagPath "

class to handle a path of tag.

Any Resemblance to Existing XPath is Purely
Coincidentalftp://medical.nema.org/medical/dicom/supps/sup118_pc.pdf

C++ includes: gdcmTagPath.h ";

%feature("docstring")  gdcm::TagPath::TagPath "gdcm::TagPath::TagPath() ";

%feature("docstring")  gdcm::TagPath::~TagPath "gdcm::TagPath::~TagPath() ";

%feature("docstring")  gdcm::TagPath::ConstructFromString "bool
gdcm::TagPath::ConstructFromString(const char *path)

\"/0018,0018/\"... No space allowed, comma is use to separate tag
group from tag element and slash is used to separate tag return false
if invalid ";

%feature("docstring")  gdcm::TagPath::ConstructFromTagList "bool
gdcm::TagPath::ConstructFromTagList(Tag const *l, unsigned int n)

Construct from a list of tags. ";

%feature("docstring")  gdcm::TagPath::Print "void
gdcm::TagPath::Print(std::ostream &) const ";

%feature("docstring")  gdcm::TagPath::Push "bool
gdcm::TagPath::Push(Tag const &t) ";

%feature("docstring")  gdcm::TagPath::Push "bool
gdcm::TagPath::Push(unsigned int itemnum) ";


// File: classgdcm_1_1Testing.xml
%feature("docstring") gdcm::Testing "

class for testing

this class is used for the nightly regression system for GDCM It makes
heavily use of md5 computation

See:   gdcm::MD5 class for md5 computation

C++ includes: gdcmTesting.h ";

%feature("docstring")  gdcm::Testing::Testing "gdcm::Testing::Testing() ";

%feature("docstring")  gdcm::Testing::~Testing "gdcm::Testing::~Testing() ";

%feature("docstring")  gdcm::Testing::Print "void
gdcm::Testing::Print(std::ostream &os=std::cout)

Print. ";


// File: classstd_1_1thread.xml
%feature("docstring") std::thread "

STL class. ";


// File: classgdcm_1_1Trace.xml
%feature("docstring") gdcm::Trace "

Trace.

Debug / Warning and Error are encapsulated in this class by default
the Trace class will redirect any debug/warning/error to std::cerr.
Unless SetStream was specified with another (open) stream or
SetStreamToFile was specified to a writable file on the system.

WARNING:  All string messages are removed during compilation time when
compiled with CMAKE_BUILD_TYPE being set to either: Release

MinSizeRel It is recommended to compile with RelWithDebInfo and/or
Debug during prototyping of applications.

C++ includes: gdcmTrace.h ";

%feature("docstring")  gdcm::Trace::Trace "gdcm::Trace::Trace() ";

%feature("docstring")  gdcm::Trace::~Trace "gdcm::Trace::~Trace() ";


// File: classgdcm_1_1TransferSyntax.xml
%feature("docstring") gdcm::TransferSyntax "

Class to manipulate Transfer Syntax.

TRANSFER SYNTAX (Standard and Private): A set of encoding rules that
allow Application Entities to unambiguously negotiate the encoding
techniques (e.g., Data Element structure, byte ordering, compression)
they are able to support, thereby allowing these Application Entities
to communicate. Todo : The implementation is completely retarded ->
see gdcm::UIDs for a replacement We need: IsSupported We need
preprocess of raw/xml file We need GetFullName()

Need a notion of Private Syntax. As defined in PS 3.5. Section 9.2

See:   UIDs

C++ includes: gdcmTransferSyntax.h ";

%feature("docstring")  gdcm::TransferSyntax::TransferSyntax "gdcm::TransferSyntax::TransferSyntax(TSType
type=ImplicitVRLittleEndian) ";

%feature("docstring")  gdcm::TransferSyntax::CanStoreLossy "bool
gdcm::TransferSyntax::CanStoreLossy() const

return true if TransFer Syntax Allow storing of Lossy Pixel Data ";

%feature("docstring")  gdcm::TransferSyntax::GetNegociatedType "NegociatedType gdcm::TransferSyntax::GetNegociatedType() const ";

%feature("docstring")  gdcm::TransferSyntax::GetString "const char*
gdcm::TransferSyntax::GetString() const ";

%feature("docstring")  gdcm::TransferSyntax::GetSwapCode "SwapCode
gdcm::TransferSyntax::GetSwapCode() const

Deprecated Return the SwapCode associated with the Transfer Syntax. Be
careful with the special GE private syntax the DataSet is written in
little endian but the Pixel Data is in Big Endian. ";

%feature("docstring")  gdcm::TransferSyntax::IsEncapsulated "bool
gdcm::TransferSyntax::IsEncapsulated() const ";

%feature("docstring")  gdcm::TransferSyntax::IsEncoded "bool
gdcm::TransferSyntax::IsEncoded() const ";

%feature("docstring")  gdcm::TransferSyntax::IsExplicit "bool
gdcm::TransferSyntax::IsExplicit() const ";

%feature("docstring")  gdcm::TransferSyntax::IsImplicit "bool
gdcm::TransferSyntax::IsImplicit() const ";

%feature("docstring")  gdcm::TransferSyntax::IsLossless "bool
gdcm::TransferSyntax::IsLossless() const

Return true if the transfer syntax algorithm is a lossless algorithm
";

%feature("docstring")  gdcm::TransferSyntax::IsLossy "bool
gdcm::TransferSyntax::IsLossy() const

Return true if the transfer syntax algorithm is a lossy algorithm ";

%feature("docstring")  gdcm::TransferSyntax::IsValid "bool
gdcm::TransferSyntax::IsValid() const ";


// File: classgdcm_1_1network_1_1TransferSyntaxSub.xml
%feature("docstring") gdcm::network::TransferSyntaxSub "

TransferSyntaxSub Table 9-15 TRANSFER SYNTAX SUB-ITEM FIELDS.

TODO what is the goal of :

Table 9-19 TRANSFER SYNTAX SUB-ITEM FIELDS

C++ includes: gdcmTransferSyntaxSub.h ";

%feature("docstring")
gdcm::network::TransferSyntaxSub::TransferSyntaxSub "gdcm::network::TransferSyntaxSub::TransferSyntaxSub() ";

%feature("docstring")  gdcm::network::TransferSyntaxSub::GetName "const char* gdcm::network::TransferSyntaxSub::GetName() const ";

%feature("docstring")  gdcm::network::TransferSyntaxSub::Print "void
gdcm::network::TransferSyntaxSub::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::network::TransferSyntaxSub::Read "std::istream& gdcm::network::TransferSyntaxSub::Read(std::istream &is)
";

%feature("docstring")  gdcm::network::TransferSyntaxSub::SetName "void gdcm::network::TransferSyntaxSub::SetName(const char *name) ";

%feature("docstring")
gdcm::network::TransferSyntaxSub::SetNameFromUID "void
gdcm::network::TransferSyntaxSub::SetNameFromUID(UIDs::TSName tsname)
";

%feature("docstring")  gdcm::network::TransferSyntaxSub::Size "size_t
gdcm::network::TransferSyntaxSub::Size() const ";

%feature("docstring")  gdcm::network::TransferSyntaxSub::Write "const
std::ostream& gdcm::network::TransferSyntaxSub::Write(std::ostream
&os) const ";


// File: structgdcm_1_1network_1_1Transition.xml
%feature("docstring") gdcm::network::Transition "C++ includes:
gdcmULTransitionTable.h ";

%feature("docstring")  gdcm::network::Transition::Transition "gdcm::network::Transition::Transition() ";

%feature("docstring")  gdcm::network::Transition::Transition "gdcm::network::Transition::Transition(int inEndState, ULAction
*inAction) ";

%feature("docstring")  gdcm::network::Transition::~Transition "gdcm::network::Transition::~Transition() ";


// File: classgdcm_1_1Type.xml
%feature("docstring") gdcm::Type "

Type.

PS 3.5 7.4 DATA ELEMENT TYPE 7.4.1 TYPE 1 REQUIRED DATA ELEMENTS 7.4.2
TYPE 1C CONDITIONAL DATA ELEMENTS 7.4.3 TYPE 2 REQUIRED DATA ELEMENTS
7.4.4 TYPE 2C CONDITIONAL DATA ELEMENTS 7.4.5 TYPE 3 OPTIONAL DATA
ELEMENTS  The intent of Type 2 Data Elements is to allow a zero length
to be conveyed when the operator or application does not know its
value or has a specific reason for not specifying its value. It is the
intent that the device should support these Data Elements.

C++ includes: gdcmType.h ";

%feature("docstring")  gdcm::Type::Type "gdcm::Type::Type(TypeType
type=UNKNOWN) ";


// File: structgdcm_1_1UI.xml
%feature("docstring") gdcm::UI "C++ includes: gdcmVR.h ";


// File: classgdcm_1_1UIDGenerator.xml
%feature("docstring") gdcm::UIDGenerator "

Class for generating unique UID.

bla Usage: When constructing a Series or Study UID, user has to keep
around the UID, otherwise the UID Generator will simply forget the
value and create a new UID.

C++ includes: gdcmUIDGenerator.h ";

%feature("docstring")  gdcm::UIDGenerator::UIDGenerator "gdcm::UIDGenerator::UIDGenerator()

By default the root of a UID is a GDCM Root... ";

%feature("docstring")  gdcm::UIDGenerator::Generate "const char*
gdcm::UIDGenerator::Generate()

Internally uses a std::string, so two calls have the same pointer !
save into a std::string In summary do not write code like that: const
char *uid1 = uid.Generate(); const char *uid2 = uid.Generate(); since
uid1 == uid2 ";


// File: classgdcm_1_1UIDs.xml
%feature("docstring") gdcm::UIDs "

all known uids

C++ includes: gdcmUIDs.h ";

%feature("docstring")  gdcm::UIDs::GetName "const char*
gdcm::UIDs::GetName() const

When object is Initialize function return the well known name
associated with uid return NULL when not initialized ";

%feature("docstring")  gdcm::UIDs::GetString "const char*
gdcm::UIDs::GetString() const

When object is Initialize function return the uid return NULL when not
initialized ";

%feature("docstring")  gdcm::UIDs::SetFromUID "bool
gdcm::UIDs::SetFromUID(const char *str)

Initialize object from a string (a uid number) return false on error,
and internal state is set to 0 ";


// File: classgdcm_1_1network_1_1ULAction.xml
%feature("docstring") gdcm::network::ULAction "

ULAction A ULConnection in a given ULState can perform certain
ULActions. This base class provides the interface for running those
ULActions on a given ULConnection.

Essentially, the ULConnectionManager will take this object, determined
from the current ULState of the ULConnection, and pass the
ULConnection object to the ULAction. The ULAction will then invoke
whatever necessary commands are required by a given action.

The result of a ULAction is a ULEvent (ie, what happened as a result
of the action).

This ULEvent is passed to the ULState, so that the transition to the
next state can occur.

Actions are associated with Payloads be thos filestreams, AETitles to
establish connections, whatever. The actual parameters that the user
will pass via an action will come through a Payload object, which
should, in itself, be some gdcm-based object (but not all objects can
be payloads; sending a single dataelement as a payload isn't
meaningful). As such, each action has its own particular payload.

For the sake of keeping files together, both the particular payload
class and the action class will be defined in the same header file.
Payloads should JUST be data (or streams), NO METHODS.

Some actions perform changes that should raise events on the local
system, and some actions perform changes that will require waiting for
events from the remote system.

Therefore, this base action has been modified so that those events are
set by each action. When the event loop runs an action, it will then
test to see if a local event was raised by the action, and if so,
perform the appropriate subsequent action. If the action requires
waiting for a response from the remote system, then the event loop
will sit there (presumably with the ARTIM timer running) and wait for
a response from the remote system. Once a response is obtained, then
the the rest of the state transitions can happen.

C++ includes: gdcmULAction.h ";

%feature("docstring")  gdcm::network::ULAction::ULAction "gdcm::network::ULAction::ULAction() ";

%feature("docstring")  gdcm::network::ULAction::~ULAction "virtual
gdcm::network::ULAction::~ULAction() ";

%feature("docstring")  gdcm::network::ULAction::PerformAction "virtual EStateID gdcm::network::ULAction::PerformAction(Subject *s,
ULEvent &inEvent, ULConnection &inConnection, bool
&outWaitingForEvent, EEventID &outRaisedEvent)=0 ";


// File: classgdcm_1_1network_1_1ULActionAA1.xml
%feature("docstring") gdcm::network::ULActionAA1 "C++ includes:
gdcmULActionAA.h ";

%feature("docstring")  gdcm::network::ULActionAA1::PerformAction "EStateID gdcm::network::ULActionAA1::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAA2.xml
%feature("docstring") gdcm::network::ULActionAA2 "C++ includes:
gdcmULActionAA.h ";

%feature("docstring")  gdcm::network::ULActionAA2::PerformAction "EStateID gdcm::network::ULActionAA2::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAA3.xml
%feature("docstring") gdcm::network::ULActionAA3 "C++ includes:
gdcmULActionAA.h ";

%feature("docstring")  gdcm::network::ULActionAA3::PerformAction "EStateID gdcm::network::ULActionAA3::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAA4.xml
%feature("docstring") gdcm::network::ULActionAA4 "C++ includes:
gdcmULActionAA.h ";

%feature("docstring")  gdcm::network::ULActionAA4::PerformAction "EStateID gdcm::network::ULActionAA4::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAA5.xml
%feature("docstring") gdcm::network::ULActionAA5 "C++ includes:
gdcmULActionAA.h ";

%feature("docstring")  gdcm::network::ULActionAA5::PerformAction "EStateID gdcm::network::ULActionAA5::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAA6.xml
%feature("docstring") gdcm::network::ULActionAA6 "C++ includes:
gdcmULActionAA.h ";

%feature("docstring")  gdcm::network::ULActionAA6::PerformAction "EStateID gdcm::network::ULActionAA6::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAA7.xml
%feature("docstring") gdcm::network::ULActionAA7 "C++ includes:
gdcmULActionAA.h ";

%feature("docstring")  gdcm::network::ULActionAA7::PerformAction "EStateID gdcm::network::ULActionAA7::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAA8.xml
%feature("docstring") gdcm::network::ULActionAA8 "C++ includes:
gdcmULActionAA.h ";

%feature("docstring")  gdcm::network::ULActionAA8::PerformAction "EStateID gdcm::network::ULActionAA8::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAE1.xml
%feature("docstring") gdcm::network::ULActionAE1 "C++ includes:
gdcmULActionAE.h ";

%feature("docstring")  gdcm::network::ULActionAE1::PerformAction "EStateID gdcm::network::ULActionAE1::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAE2.xml
%feature("docstring") gdcm::network::ULActionAE2 "C++ includes:
gdcmULActionAE.h ";

%feature("docstring")  gdcm::network::ULActionAE2::PerformAction "EStateID gdcm::network::ULActionAE2::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAE3.xml
%feature("docstring") gdcm::network::ULActionAE3 "C++ includes:
gdcmULActionAE.h ";

%feature("docstring")  gdcm::network::ULActionAE3::PerformAction "EStateID gdcm::network::ULActionAE3::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAE4.xml
%feature("docstring") gdcm::network::ULActionAE4 "C++ includes:
gdcmULActionAE.h ";

%feature("docstring")  gdcm::network::ULActionAE4::PerformAction "EStateID gdcm::network::ULActionAE4::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAE5.xml
%feature("docstring") gdcm::network::ULActionAE5 "C++ includes:
gdcmULActionAE.h ";

%feature("docstring")  gdcm::network::ULActionAE5::PerformAction "EStateID gdcm::network::ULActionAE5::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAE6.xml
%feature("docstring") gdcm::network::ULActionAE6 "C++ includes:
gdcmULActionAE.h ";

%feature("docstring")  gdcm::network::ULActionAE6::PerformAction "EStateID gdcm::network::ULActionAE6::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAE7.xml
%feature("docstring") gdcm::network::ULActionAE7 "C++ includes:
gdcmULActionAE.h ";

%feature("docstring")  gdcm::network::ULActionAE7::PerformAction "EStateID gdcm::network::ULActionAE7::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAE8.xml
%feature("docstring") gdcm::network::ULActionAE8 "C++ includes:
gdcmULActionAE.h ";

%feature("docstring")  gdcm::network::ULActionAE8::PerformAction "EStateID gdcm::network::ULActionAE8::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAR1.xml
%feature("docstring") gdcm::network::ULActionAR1 "C++ includes:
gdcmULActionAR.h ";

%feature("docstring")  gdcm::network::ULActionAR1::PerformAction "EStateID gdcm::network::ULActionAR1::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAR10.xml
%feature("docstring") gdcm::network::ULActionAR10 "C++ includes:
gdcmULActionAR.h ";

%feature("docstring")  gdcm::network::ULActionAR10::PerformAction "EStateID gdcm::network::ULActionAR10::PerformAction(Subject *s,
ULEvent &inEvent, ULConnection &inConnection, bool
&outWaitingForEvent, EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAR2.xml
%feature("docstring") gdcm::network::ULActionAR2 "C++ includes:
gdcmULActionAR.h ";

%feature("docstring")  gdcm::network::ULActionAR2::PerformAction "EStateID gdcm::network::ULActionAR2::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAR3.xml
%feature("docstring") gdcm::network::ULActionAR3 "C++ includes:
gdcmULActionAR.h ";

%feature("docstring")  gdcm::network::ULActionAR3::PerformAction "EStateID gdcm::network::ULActionAR3::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAR4.xml
%feature("docstring") gdcm::network::ULActionAR4 "C++ includes:
gdcmULActionAR.h ";

%feature("docstring")  gdcm::network::ULActionAR4::PerformAction "EStateID gdcm::network::ULActionAR4::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAR5.xml
%feature("docstring") gdcm::network::ULActionAR5 "C++ includes:
gdcmULActionAR.h ";

%feature("docstring")  gdcm::network::ULActionAR5::PerformAction "EStateID gdcm::network::ULActionAR5::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAR6.xml
%feature("docstring") gdcm::network::ULActionAR6 "C++ includes:
gdcmULActionAR.h ";

%feature("docstring")  gdcm::network::ULActionAR6::PerformAction "EStateID gdcm::network::ULActionAR6::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAR7.xml
%feature("docstring") gdcm::network::ULActionAR7 "C++ includes:
gdcmULActionAR.h ";

%feature("docstring")  gdcm::network::ULActionAR7::PerformAction "EStateID gdcm::network::ULActionAR7::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAR8.xml
%feature("docstring") gdcm::network::ULActionAR8 "C++ includes:
gdcmULActionAR.h ";

%feature("docstring")  gdcm::network::ULActionAR8::PerformAction "EStateID gdcm::network::ULActionAR8::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionAR9.xml
%feature("docstring") gdcm::network::ULActionAR9 "C++ includes:
gdcmULActionAR.h ";

%feature("docstring")  gdcm::network::ULActionAR9::PerformAction "EStateID gdcm::network::ULActionAR9::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionDT1.xml
%feature("docstring") gdcm::network::ULActionDT1 "C++ includes:
gdcmULActionDT.h ";

%feature("docstring")  gdcm::network::ULActionDT1::PerformAction "EStateID gdcm::network::ULActionDT1::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULActionDT2.xml
%feature("docstring") gdcm::network::ULActionDT2 "C++ includes:
gdcmULActionDT.h ";

%feature("docstring")  gdcm::network::ULActionDT2::PerformAction "EStateID gdcm::network::ULActionDT2::PerformAction(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) ";


// File: classgdcm_1_1network_1_1ULBasicCallback.xml
%feature("docstring") gdcm::network::ULBasicCallback "

ULBasicCallback This is the most basic of callbacks for how the
ULConnectionManager handles incoming datasets. DataSets are just
concatenated to the mDataSets vector, and the result can be pulled out
of the vector by later code. Alternatives to this method include
progress updates, saving to disk, etc. This class is NOT THREAD SAFE.
Access the dataset vector after the entire set of datasets has been
returned by the ULConnectionManager.

C++ includes: gdcmULBasicCallback.h ";

%feature("docstring")  gdcm::network::ULBasicCallback::ULBasicCallback
"gdcm::network::ULBasicCallback::ULBasicCallback() ";

%feature("docstring")
gdcm::network::ULBasicCallback::~ULBasicCallback "virtual
gdcm::network::ULBasicCallback::~ULBasicCallback() ";

%feature("docstring")  gdcm::network::ULBasicCallback::GetDataSets "std::vector<DataSet> const&
gdcm::network::ULBasicCallback::GetDataSets() const ";

%feature("docstring")  gdcm::network::ULBasicCallback::GetResponses "std::vector<DataSet> const&
gdcm::network::ULBasicCallback::GetResponses() const ";

%feature("docstring")  gdcm::network::ULBasicCallback::HandleDataSet "virtual void gdcm::network::ULBasicCallback::HandleDataSet(const
DataSet &inDataSet) ";

%feature("docstring")  gdcm::network::ULBasicCallback::HandleResponse
"virtual void gdcm::network::ULBasicCallback::HandleResponse(const
DataSet &inDataSet) ";


// File: classgdcm_1_1network_1_1ULConnection.xml
%feature("docstring") gdcm::network::ULConnection "

ULConnection This is the class that contains the socket to another
machine, and passes data through itself, as well as maintaining a
sense of state.

The ULConnectionManager tells the ULConnection what data can actually
be sent.

This class is done this way so that it can be eventually be replaced
with a ULSecureConnection, if such a protocol is warranted, so that
all data that passes through can be managed through a secure
connection. For now, this class provides a simple pass-through
mechanism to the socket itself.

So, for instance, a gdcm object will be passes to this object, and it
will then get passed along the connection, if that connection is in
the proper state to do so.

For right now, this class is not directly intended to be inherited
from, but the potential for future ULSecureConnection warrants the
addition, rather than having everything be managed from within the
ULConnectionManager (or this class) without a wrapper.

C++ includes: gdcmULConnection.h ";

%feature("docstring")  gdcm::network::ULConnection::ULConnection "gdcm::network::ULConnection::ULConnection(const ULConnectionInfo
&inUserInformation) ";

%feature("docstring")  gdcm::network::ULConnection::~ULConnection "virtual gdcm::network::ULConnection::~ULConnection() ";

%feature("docstring")
gdcm::network::ULConnection::AddAcceptedPresentationContext "void
gdcm::network::ULConnection::AddAcceptedPresentationContext(const
PresentationContextAC &inPC) ";

%feature("docstring")  gdcm::network::ULConnection::FindContext "PresentationContextRQ gdcm::network::ULConnection::FindContext(const
DataElement &de) const ";

%feature("docstring")
gdcm::network::ULConnection::GetAcceptedPresentationContexts "std::vector<PresentationContextAC> const&
gdcm::network::ULConnection::GetAcceptedPresentationContexts() const
";

%feature("docstring")
gdcm::network::ULConnection::GetAcceptedPresentationContexts "std::vector<PresentationContextAC>&
gdcm::network::ULConnection::GetAcceptedPresentationContexts() ";

%feature("docstring")  gdcm::network::ULConnection::GetConnectionInfo
"const ULConnectionInfo&
gdcm::network::ULConnection::GetConnectionInfo() const ";

%feature("docstring")  gdcm::network::ULConnection::GetMaxPDUSize "uint32_t gdcm::network::ULConnection::GetMaxPDUSize() const ";

%feature("docstring")
gdcm::network::ULConnection::GetPresentationContextACByID "const
PresentationContextAC*
gdcm::network::ULConnection::GetPresentationContextACByID(uint8_t id)
const ";

%feature("docstring")
gdcm::network::ULConnection::GetPresentationContextIDFromPresentationContext
"uint8_t
gdcm::network::ULConnection::GetPresentationContextIDFromPresentationContext(PresentationContextRQ
const &pc) const

return 0 upon error ";

%feature("docstring")
gdcm::network::ULConnection::GetPresentationContextRQByID "const
PresentationContextRQ*
gdcm::network::ULConnection::GetPresentationContextRQByID(uint8_t id)
const ";

%feature("docstring")
gdcm::network::ULConnection::GetPresentationContexts "std::vector<PresentationContextRQ> const&
gdcm::network::ULConnection::GetPresentationContexts() const ";

%feature("docstring")  gdcm::network::ULConnection::GetProtocol "std::iostream* gdcm::network::ULConnection::GetProtocol() ";

%feature("docstring")  gdcm::network::ULConnection::GetState "EStateID gdcm::network::ULConnection::GetState() const ";

%feature("docstring")  gdcm::network::ULConnection::GetTimer "ARTIMTimer& gdcm::network::ULConnection::GetTimer() ";

%feature("docstring")
gdcm::network::ULConnection::InitializeConnection "bool
gdcm::network::ULConnection::InitializeConnection()

used to establish scu connections ";

%feature("docstring")
gdcm::network::ULConnection::InitializeIncomingConnection "bool
gdcm::network::ULConnection::InitializeIncomingConnection()

used to establish scp connections ";

%feature("docstring")  gdcm::network::ULConnection::SetMaxPDUSize "void gdcm::network::ULConnection::SetMaxPDUSize(uint32_t inSize) ";

%feature("docstring")
gdcm::network::ULConnection::SetPresentationContexts "void
gdcm::network::ULConnection::SetPresentationContexts(const
std::vector< PresentationContextRQ > &inContexts) ";

%feature("docstring")
gdcm::network::ULConnection::SetPresentationContexts "void
gdcm::network::ULConnection::SetPresentationContexts(const
std::vector< PresentationContext > &inContexts) ";

%feature("docstring")  gdcm::network::ULConnection::SetState "void
gdcm::network::ULConnection::SetState(const EStateID &inState) ";

%feature("docstring")  gdcm::network::ULConnection::StopProtocol "void gdcm::network::ULConnection::StopProtocol() ";


// File: classgdcm_1_1network_1_1ULConnectionCallback.xml
%feature("docstring") gdcm::network::ULConnectionCallback "

When a dataset comes back from a query/move/etc, the result can either
be stored entirely in memory, or could be stored on disk. This class
provides a mechanism to indicate what the ULConnectionManager should
do with datasets that are produced through query results. The
ULConnectionManager will call the HandleDataSet function during the
course of receiving datasets. Particular implementations should fill
in what that function does, including updating progress, etc. NOTE:
since cmove requires that multiple event loops be employed, the
callback function MUST set mHandledDataSet to true. otherwise, the
cmove event loop handler will not know data was received, and proceed
to end the loop prematurely.

C++ includes: gdcmULConnectionCallback.h ";

%feature("docstring")
gdcm::network::ULConnectionCallback::ULConnectionCallback "gdcm::network::ULConnectionCallback::ULConnectionCallback() ";

%feature("docstring")
gdcm::network::ULConnectionCallback::~ULConnectionCallback "virtual
gdcm::network::ULConnectionCallback::~ULConnectionCallback() ";

%feature("docstring")
gdcm::network::ULConnectionCallback::DataSetHandles "bool
gdcm::network::ULConnectionCallback::DataSetHandles() const ";

%feature("docstring")
gdcm::network::ULConnectionCallback::HandleDataSet "virtual void
gdcm::network::ULConnectionCallback::HandleDataSet(const DataSet
&inDataSet)=0 ";

%feature("docstring")
gdcm::network::ULConnectionCallback::HandleResponse "virtual void
gdcm::network::ULConnectionCallback::HandleResponse(const DataSet
&inDataSet)=0 ";

%feature("docstring")
gdcm::network::ULConnectionCallback::ResetHandledDataSet "void
gdcm::network::ULConnectionCallback::ResetHandledDataSet() ";

%feature("docstring")
gdcm::network::ULConnectionCallback::SetImplicitFlag "void
gdcm::network::ULConnectionCallback::SetImplicitFlag(const bool imp)
";


// File: classgdcm_1_1network_1_1ULConnectionInfo.xml
%feature("docstring") gdcm::network::ULConnectionInfo "

ULConnectionInfo this class contains all the information about a
particular connection as established by the user. That is, it's: User
Information Calling AE Title Called AE Title IP address/computer name
IP Port A connection must be established with this information, that's
subsequently placed into various primitives for actual communication.

C++ includes: gdcmULConnectionInfo.h ";

%feature("docstring")
gdcm::network::ULConnectionInfo::ULConnectionInfo "gdcm::network::ULConnectionInfo::ULConnectionInfo() ";

%feature("docstring")
gdcm::network::ULConnectionInfo::GetCalledAETitle "const char*
gdcm::network::ULConnectionInfo::GetCalledAETitle() const ";

%feature("docstring")
gdcm::network::ULConnectionInfo::GetCalledComputerName "std::string
gdcm::network::ULConnectionInfo::GetCalledComputerName() const ";

%feature("docstring")
gdcm::network::ULConnectionInfo::GetCalledIPAddress "unsigned long
gdcm::network::ULConnectionInfo::GetCalledIPAddress() const ";

%feature("docstring")
gdcm::network::ULConnectionInfo::GetCalledIPPort "int
gdcm::network::ULConnectionInfo::GetCalledIPPort() const ";

%feature("docstring")
gdcm::network::ULConnectionInfo::GetCallingAETitle "const char*
gdcm::network::ULConnectionInfo::GetCallingAETitle() const ";

%feature("docstring")
gdcm::network::ULConnectionInfo::GetMaxPDULength "unsigned long
gdcm::network::ULConnectionInfo::GetMaxPDULength() const ";

%feature("docstring")  gdcm::network::ULConnectionInfo::Initialize "bool gdcm::network::ULConnectionInfo::Initialize(UserInformation const
&inUserInformation, const char *inCalledAETitle, const char
*inCallingAETitle, unsigned long inCalledIPAddress, int
inCalledIPPort, std::string inCalledComputerName) ";

%feature("docstring")
gdcm::network::ULConnectionInfo::SetMaxPDULength "void
gdcm::network::ULConnectionInfo::SetMaxPDULength(unsigned long
inMaxPDULength) ";


// File: classgdcm_1_1network_1_1ULConnectionManager.xml
%feature("docstring") gdcm::network::ULConnectionManager "

ULConnectionManager The ULConnectionManager performs actions on the
ULConnection given inputs from the user and from the state of what's
going on around the connection (ie, timeouts of the ARTIM timer,
responses from the peer across the connection, etc).

Its inputs are ULEvents, and it performs ULActions.

C++ includes: gdcmULConnectionManager.h ";

%feature("docstring")
gdcm::network::ULConnectionManager::ULConnectionManager "gdcm::network::ULConnectionManager::ULConnectionManager() ";

%feature("docstring")
gdcm::network::ULConnectionManager::~ULConnectionManager "gdcm::network::ULConnectionManager::~ULConnectionManager() ";

%feature("docstring")
gdcm::network::ULConnectionManager::BreakConnection "bool
gdcm::network::ULConnectionManager::BreakConnection(const double
&inTimeout) ";

%feature("docstring")
gdcm::network::ULConnectionManager::BreakConnectionNow "void
gdcm::network::ULConnectionManager::BreakConnectionNow() ";

%feature("docstring")
gdcm::network::ULConnectionManager::EstablishConnection "bool
gdcm::network::ULConnectionManager::EstablishConnection(const
std::string &inAETitle, const std::string &inConnectAETitle, const
std::string &inComputerName, long inIPAddress, uint16_t inConnectPort,
double inTimeout, std::vector< PresentationContext > const &pcVector)

returns true if a connection of the given AETitle (ie, 'this' program)
is able to connect to the given AETitle and Port in a certain amount
of time providing the connection type will establish the proper
exchange syntax with a server; if a different functionality is
required, a different connection should be established. returns false
if the connection type is 'move' have to give a return port for move
to work as specified. ";

%feature("docstring")
gdcm::network::ULConnectionManager::EstablishConnectionMove "bool
gdcm::network::ULConnectionManager::EstablishConnectionMove(const
std::string &inAETitle, const std::string &inConnectAETitle, const
std::string &inComputerName, long inIPAddress, uint16_t inConnectPort,
double inTimeout, uint16_t inReturnPort, std::vector<
PresentationContext > const &pcVector)

returns true for above reasons, but contains the special 'move' port
";

%feature("docstring")  gdcm::network::ULConnectionManager::SendEcho "std::vector<PresentationDataValue>
gdcm::network::ULConnectionManager::SendEcho() ";

%feature("docstring")  gdcm::network::ULConnectionManager::SendFind "std::vector<DataSet>
gdcm::network::ULConnectionManager::SendFind(const BaseRootQuery
*inRootQuery) ";

%feature("docstring")  gdcm::network::ULConnectionManager::SendFind "void gdcm::network::ULConnectionManager::SendFind(const BaseRootQuery
*inRootQuery, ULConnectionCallback *inCallback) ";

%feature("docstring")  gdcm::network::ULConnectionManager::SendMove "std::vector<DataSet>
gdcm::network::ULConnectionManager::SendMove(const BaseRootQuery
*inRootQuery) ";

%feature("docstring")  gdcm::network::ULConnectionManager::SendMove "bool gdcm::network::ULConnectionManager::SendMove(const BaseRootQuery
*inRootQuery, ULConnectionCallback *inCallback)

return false upon error ";

%feature("docstring")  gdcm::network::ULConnectionManager::SendStore "std::vector<DataSet>
gdcm::network::ULConnectionManager::SendStore(const File &file) ";

%feature("docstring")  gdcm::network::ULConnectionManager::SendStore "void gdcm::network::ULConnectionManager::SendStore(const File &file,
ULConnectionCallback *inCallback)

callback based API ";


// File: classgdcm_1_1network_1_1ULEvent.xml
%feature("docstring") gdcm::network::ULEvent "

ULEvent base class for network events.

An event consists of the event ID and the data associated with that
event.

Note that once a PDU is created, it is now the responsibility of the
associated event to destroy it!

C++ includes: gdcmULEvent.h ";

%feature("docstring")  gdcm::network::ULEvent::ULEvent "gdcm::network::ULEvent::ULEvent(const EEventID &inEventID,
std::vector< BasePDU * > const &inBasePDU) ";

%feature("docstring")  gdcm::network::ULEvent::ULEvent "gdcm::network::ULEvent::ULEvent(const EEventID &inEventID, BasePDU
*inBasePDU) ";

%feature("docstring")  gdcm::network::ULEvent::~ULEvent "gdcm::network::ULEvent::~ULEvent() ";

%feature("docstring")  gdcm::network::ULEvent::GetEvent "EEventID
gdcm::network::ULEvent::GetEvent() const ";

%feature("docstring")  gdcm::network::ULEvent::GetPDUs "std::vector<BasePDU*> const& gdcm::network::ULEvent::GetPDUs() const
";

%feature("docstring")  gdcm::network::ULEvent::SetEvent "void
gdcm::network::ULEvent::SetEvent(const EEventID &inEvent) ";

%feature("docstring")  gdcm::network::ULEvent::SetPDU "void
gdcm::network::ULEvent::SetPDU(std::vector< BasePDU * > const &inPDU)
";


// File: classgdcm_1_1network_1_1ULTransitionTable.xml
%feature("docstring") gdcm::network::ULTransitionTable "

ULTransitionTable The transition table of all the ULEvents, new
ULActions, and ULStates.

Based roughly on the solutions in player2.cpp in the boost examples
and this so question:http://stackoverflow.com/questions/1647631/c
-state-machine-design

The transition table is constructed of TableRows. Each row is based on
an event, and an event handler in the TransitionTable object takes a
given event, and then finds the given row.

Then, given the current state of the connection, determines the
appropriate action to take and then the state to transition to next.

C++ includes: gdcmULTransitionTable.h ";

%feature("docstring")
gdcm::network::ULTransitionTable::ULTransitionTable "gdcm::network::ULTransitionTable::ULTransitionTable() ";

%feature("docstring")  gdcm::network::ULTransitionTable::HandleEvent "void gdcm::network::ULTransitionTable::HandleEvent(Subject *s, ULEvent
&inEvent, ULConnection &inConnection, bool &outWaitingForEvent,
EEventID &outRaisedEvent) const ";

%feature("docstring")  gdcm::network::ULTransitionTable::PrintTable "void gdcm::network::ULTransitionTable::PrintTable() const ";


// File: classgdcm_1_1network_1_1ULWritingCallback.xml
%feature("docstring") gdcm::network::ULWritingCallback "C++ includes:
gdcmULWritingCallback.h ";

%feature("docstring")
gdcm::network::ULWritingCallback::ULWritingCallback "gdcm::network::ULWritingCallback::ULWritingCallback() ";

%feature("docstring")
gdcm::network::ULWritingCallback::~ULWritingCallback "virtual
gdcm::network::ULWritingCallback::~ULWritingCallback() ";

%feature("docstring")  gdcm::network::ULWritingCallback::HandleDataSet
"virtual void gdcm::network::ULWritingCallback::HandleDataSet(const
DataSet &inDataSet) ";

%feature("docstring")
gdcm::network::ULWritingCallback::HandleResponse "virtual void
gdcm::network::ULWritingCallback::HandleResponse(const DataSet
&inDataSet) ";

%feature("docstring")  gdcm::network::ULWritingCallback::SetDirectory
"void gdcm::network::ULWritingCallback::SetDirectory(const
std::string &inDirectoryName)

provide the directory into which all files are written. ";


// File: classstd_1_1underflow__error.xml
%feature("docstring") std::underflow_error "

STL class. ";


// File: classgdcm_1_1UNExplicitDataElement.xml
%feature("docstring") gdcm::UNExplicitDataElement "

Class to read/write a DataElement as UNExplicit Data Element.

bla

C++ includes: gdcmUNExplicitDataElement.h ";

%feature("docstring")  gdcm::UNExplicitDataElement::GetLength "VL
gdcm::UNExplicitDataElement::GetLength() const ";

%feature("docstring")  gdcm::UNExplicitDataElement::Read "std::istream& gdcm::UNExplicitDataElement::Read(std::istream &is) ";

%feature("docstring")  gdcm::UNExplicitDataElement::ReadPreValue "std::istream& gdcm::UNExplicitDataElement::ReadPreValue(std::istream
&is) ";

%feature("docstring")  gdcm::UNExplicitDataElement::ReadValue "std::istream& gdcm::UNExplicitDataElement::ReadValue(std::istream &is,
bool readvalues=true) ";

%feature("docstring")  gdcm::UNExplicitDataElement::ReadWithLength "std::istream& gdcm::UNExplicitDataElement::ReadWithLength(std::istream
&is, VL &length) ";


// File: classgdcm_1_1UNExplicitImplicitDataElement.xml
%feature("docstring") gdcm::UNExplicitImplicitDataElement "

Class to read/write a DataElement as ExplicitImplicit Data Element
This class gather two known bugs:

GDCM 1.2.0 would rewrite VR=UN Value Length on 2 bytes instead of 4
bytes

GDCM 1.2.0 would also rewrite DataElement as Implicit when the VR
would not be known this would only happen in some very rare cases.
gdcm 2.X design could handle bug #1 or #2 exclusively, this class can
now handle file which have both issues. See:
gdcmData/TheralysGDCM120Bug.dcm

C++ includes: gdcmUNExplicitImplicitDataElement.h ";

%feature("docstring")  gdcm::UNExplicitImplicitDataElement::GetLength
"VL gdcm::UNExplicitImplicitDataElement::GetLength() const ";

%feature("docstring")  gdcm::UNExplicitImplicitDataElement::Read "std::istream& gdcm::UNExplicitImplicitDataElement::Read(std::istream
&is) ";

%feature("docstring")
gdcm::UNExplicitImplicitDataElement::ReadPreValue "std::istream&
gdcm::UNExplicitImplicitDataElement::ReadPreValue(std::istream &is) ";

%feature("docstring")  gdcm::UNExplicitImplicitDataElement::ReadValue
"std::istream&
gdcm::UNExplicitImplicitDataElement::ReadValue(std::istream &is) ";


// File: classstd_1_1unique__ptr.xml
%feature("docstring") std::unique_ptr "

STL class. ";


// File: classstd_1_1unordered__map.xml
%feature("docstring") std::unordered_map "

STL class. ";


// File: classstd_1_1unordered__multimap.xml
%feature("docstring") std::unordered_multimap "

STL class. ";


// File: classstd_1_1unordered__multiset.xml
%feature("docstring") std::unordered_multiset "

STL class. ";


// File: classstd_1_1unordered__set.xml
%feature("docstring") std::unordered_set "

STL class. ";


// File: classgdcm_1_1Unpacker12Bits.xml
%feature("docstring") gdcm::Unpacker12Bits "

Pack/Unpack 12 bits pixel into 16bits.

You can only pack an even number of 16bits, which means a multiple of
4 (expressed in bytes)

You can only unpack a multiple of 3 bytes  This class has no purpose
in general purpose DICOM implementation. However to be able to cope
with some early ACR-NEMA file generated by a well-known private
vendor, one would need to unpack 12bits Stored Pixel Value into a more
standard 16bits Stored Pixel Value.

See:   Rescaler

C++ includes: gdcmUnpacker12Bits.h ";


// File: classgdcm_1_1Usage.xml
%feature("docstring") gdcm::Usage "

Usage.

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

C++ includes: gdcmUsage.h ";

%feature("docstring")  gdcm::Usage::Usage "gdcm::Usage::Usage(UsageType type=Invalid) ";


// File: classgdcm_1_1UserEvent.xml
%feature("docstring") gdcm::UserEvent "C++ includes: gdcmEvent.h ";


// File: classgdcm_1_1network_1_1UserInformation.xml
%feature("docstring") gdcm::network::UserInformation "

UserInformation Table 9-16 USER INFORMATION ITEM FIELDS.

TODO what is the goal of :

Table 9-20 USER INFORMATION ITEM FIELDS

C++ includes: gdcmUserInformation.h ";

%feature("docstring")  gdcm::network::UserInformation::UserInformation
"gdcm::network::UserInformation::UserInformation() ";

%feature("docstring")
gdcm::network::UserInformation::~UserInformation "gdcm::network::UserInformation::~UserInformation() ";

%feature("docstring")
gdcm::network::UserInformation::AddRoleSelectionSub "void
gdcm::network::UserInformation::AddRoleSelectionSub(RoleSelectionSub
const &r) ";

%feature("docstring")
gdcm::network::UserInformation::AddSOPClassExtendedNegociationSub "void
gdcm::network::UserInformation::AddSOPClassExtendedNegociationSub(SOPClassExtendedNegociationSub
const &s) ";

%feature("docstring")
gdcm::network::UserInformation::GetMaximumLengthSub "const
MaximumLengthSub&
gdcm::network::UserInformation::GetMaximumLengthSub() const ";

%feature("docstring")
gdcm::network::UserInformation::GetMaximumLengthSub "MaximumLengthSub&
gdcm::network::UserInformation::GetMaximumLengthSub() ";

%feature("docstring")  gdcm::network::UserInformation::Print "void
gdcm::network::UserInformation::Print(std::ostream &os) const ";

%feature("docstring")  gdcm::network::UserInformation::Read "std::istream& gdcm::network::UserInformation::Read(std::istream &is)
";

%feature("docstring")  gdcm::network::UserInformation::Size "size_t
gdcm::network::UserInformation::Size() const ";

%feature("docstring")  gdcm::network::UserInformation::Write "const
std::ostream& gdcm::network::UserInformation::Write(std::ostream &os)
const ";


// File: classgdcm_1_1UUIDGenerator.xml
%feature("docstring") gdcm::UUIDGenerator "

Class for generating unique UUID generate DCE 1.1 uid.

C++ includes: gdcmUUIDGenerator.h ";

%feature("docstring")  gdcm::UUIDGenerator::Generate "const char*
gdcm::UUIDGenerator::Generate()

Return the generated uuid NOT THREAD SAFE ";


// File: classstd_1_1valarray.xml
%feature("docstring") std::valarray "

STL class. ";


// File: classgdcm_1_1Validate.xml
%feature("docstring") gdcm::Validate "

Validate class.

C++ includes: gdcmValidate.h ";

%feature("docstring")  gdcm::Validate::Validate "gdcm::Validate::Validate() ";

%feature("docstring")  gdcm::Validate::~Validate "gdcm::Validate::~Validate() ";

%feature("docstring")  gdcm::Validate::GetValidatedFile "const File&
gdcm::Validate::GetValidatedFile() ";

%feature("docstring")  gdcm::Validate::SetFile "void
gdcm::Validate::SetFile(File const &f) ";

%feature("docstring")  gdcm::Validate::Validation "void
gdcm::Validate::Validation() ";


// File: classgdcm_1_1Value.xml
%feature("docstring") gdcm::Value "

Class to represent the value of a Data Element.

VALUE: A component of a Value Field. A Value Field may consist of one
or more of these components.

C++ includes: gdcmValue.h ";

%feature("docstring")  gdcm::Value::Value "gdcm::Value::Value() ";

%feature("docstring")  gdcm::Value::~Value "gdcm::Value::~Value() ";

%feature("docstring")  gdcm::Value::Clear "virtual void
gdcm::Value::Clear()=0 ";

%feature("docstring")  gdcm::Value::GetLength "virtual VL
gdcm::Value::GetLength() const =0 ";

%feature("docstring")  gdcm::Value::SetLength "virtual void
gdcm::Value::SetLength(VL l)=0 ";


// File: classgdcm_1_1ValueIO.xml
%feature("docstring") gdcm::ValueIO "

Class to dispatch template calls.

C++ includes: gdcmValueIO.h ";


// File: classstd_1_1vector.xml
%feature("docstring") std::vector "

STL class. ";


// File: classgdcm_1_1Version.xml
%feature("docstring") gdcm::Version "

major/minor and build version

C++ includes: gdcmVersion.h ";

%feature("docstring")  gdcm::Version::Version "gdcm::Version::Version() ";

%feature("docstring")  gdcm::Version::~Version "gdcm::Version::~Version() ";

%feature("docstring")  gdcm::Version::Print "void
gdcm::Version::Print(std::ostream &os=std::cout) const ";


// File: classgdcm_1_1VL.xml
%feature("docstring") gdcm::VL "

Value Length.

WARNING:  this is a 4bytes value ! Do not try to use it for 2bytes
value length

C++ includes: gdcmVL.h ";

%feature("docstring")  gdcm::VL::VL "gdcm::VL::VL(uint32_t vl=0) ";

%feature("docstring")  gdcm::VL::GetLength "VL gdcm::VL::GetLength()
const ";

%feature("docstring")  gdcm::VL::IsOdd "bool gdcm::VL::IsOdd() const

Return whether or not the VL is odd or not. ";

%feature("docstring")  gdcm::VL::IsUndefined "bool
gdcm::VL::IsUndefined() const ";

%feature("docstring")  gdcm::VL::Read "std::istream&
gdcm::VL::Read(std::istream &is) ";

%feature("docstring")  gdcm::VL::Read16 "std::istream&
gdcm::VL::Read16(std::istream &is) ";

%feature("docstring")  gdcm::VL::SetToUndefined "void
gdcm::VL::SetToUndefined() ";

%feature("docstring")  gdcm::VL::Write "const std::ostream&
gdcm::VL::Write(std::ostream &os) const ";

%feature("docstring")  gdcm::VL::Write16 "const std::ostream&
gdcm::VL::Write16(std::ostream &os) const ";


// File: classgdcm_1_1VM.xml
%feature("docstring") gdcm::VM "

Value Multiplicity Looking at the DICOMV3 dict only there is very few
cases: 1 2 3 4 5 6 8 16 24 1-2 1-3 1-8 1-32 1-99 1-n 2-2n 2-n 3-3n
3-n.

Some private dict define some more: 4-4n 1-4 1-5 256 9 3-4

even more:

7-7n 10 18 12 35 47_47n 30_30n 28

6-6n

C++ includes: gdcmVM.h ";

%feature("docstring")  gdcm::VM::VM "gdcm::VM::VM(VMType type=VM0) ";

%feature("docstring")  gdcm::VM::Compatible "bool
gdcm::VM::Compatible(VM const &vm) const

WARNING: Implementation deficiency The Compatible function is poorly
implemented, the reference vm should be coming from the dictionary,
while the passed in value is the value guess from the file. ";

%feature("docstring")  gdcm::VM::GetLength "unsigned int
gdcm::VM::GetLength() const ";


// File: classgdcm_1_1VR.xml
%feature("docstring") gdcm::VR "

VR class This is adapted from DICOM standard The biggest difference is
the INVALID VR and the composite one that differ from standard (more
like an addition) This allow us to represent all the possible case
express in the DICOMV3 dict.

VALUE REPRESENTATION ( VR) Specifies the data type and format of the
Value(s) contained in the Value Field of a Data Element. VALUE
REPRESENTATION FIELD: The field where the Value Representation of a
Data Element is stored in the encoding of a Data Element structure
with explicit VR.

C++ includes: gdcmVR.h ";

%feature("docstring")  gdcm::VR::VR "gdcm::VR::VR(VRType vr=INVALID)
";

%feature("docstring")  gdcm::VR::Compatible "bool
gdcm::VR::Compatible(VR const &vr) const ";

%feature("docstring")  gdcm::VR::GetLength "int gdcm::VR::GetLength()
const ";

%feature("docstring")  gdcm::VR::GetSize "unsigned int
gdcm::VR::GetSize() const ";

%feature("docstring")  gdcm::VR::GetSizeof "unsigned int
gdcm::VR::GetSizeof() const ";

%feature("docstring")  gdcm::VR::IsDual "bool gdcm::VR::IsDual()
const ";

%feature("docstring")  gdcm::VR::IsVRFile "bool gdcm::VR::IsVRFile()
const ";

%feature("docstring")  gdcm::VR::Read "std::istream&
gdcm::VR::Read(std::istream &is) ";

%feature("docstring")  gdcm::VR::Write "const std::ostream&
gdcm::VR::Write(std::ostream &os) const ";


// File: classgdcm_1_1VR16ExplicitDataElement.xml
%feature("docstring") gdcm::VR16ExplicitDataElement "

Class to read/write a DataElement as Explicit Data Element.

This class support 16 bits when finding an unkown VR: For instance:
Siemens_CT_Sensation64_has_VR_RT.dcm

C++ includes: gdcmVR16ExplicitDataElement.h ";

%feature("docstring")  gdcm::VR16ExplicitDataElement::GetLength "VL
gdcm::VR16ExplicitDataElement::GetLength() const ";

%feature("docstring")  gdcm::VR16ExplicitDataElement::Read "std::istream& gdcm::VR16ExplicitDataElement::Read(std::istream &is) ";

%feature("docstring")  gdcm::VR16ExplicitDataElement::ReadPreValue "std::istream& gdcm::VR16ExplicitDataElement::ReadPreValue(std::istream
&is) ";

%feature("docstring")  gdcm::VR16ExplicitDataElement::ReadValue "std::istream& gdcm::VR16ExplicitDataElement::ReadValue(std::istream
&is, bool readvalues=true) ";

%feature("docstring")  gdcm::VR16ExplicitDataElement::ReadWithLength "std::istream&
gdcm::VR16ExplicitDataElement::ReadWithLength(std::istream &is, VL
&length) ";


// File: classgdcm_1_1VRVLSize_3_010_01_4.xml
%feature("docstring") gdcm::VRVLSize< 0 > " C++ includes:
gdcmAttribute.h ";


// File: classgdcm_1_1VRVLSize_3_011_01_4.xml
%feature("docstring") gdcm::VRVLSize< 1 > " C++ includes:
gdcmAttribute.h ";


// File: classvtkGDCMImageReader.xml
%feature("docstring") vtkGDCMImageReader "C++ includes:
vtkGDCMImageReader.h ";

%feature("docstring")  vtkGDCMImageReader::CanReadFile "virtual int
vtkGDCMImageReader::CanReadFile(const char *fname) ";

%feature("docstring")  vtkGDCMImageReader::GetDescriptiveName "virtual const char* vtkGDCMImageReader::GetDescriptiveName() ";

%feature("docstring")  vtkGDCMImageReader::GetFileExtensions "virtual
const char* vtkGDCMImageReader::GetFileExtensions() ";

%feature("docstring")  vtkGDCMImageReader::GetIconImage "vtkImageData* vtkGDCMImageReader::GetIconImage() ";

%feature("docstring")  vtkGDCMImageReader::GetOverlay "vtkImageData*
vtkGDCMImageReader::GetOverlay(int i) ";

%feature("docstring")  vtkGDCMImageReader::PrintSelf "virtual void
vtkGDCMImageReader::PrintSelf(ostream &os, vtkIndent indent) ";

%feature("docstring")  vtkGDCMImageReader::SetCurve "virtual void
vtkGDCMImageReader::SetCurve(vtkPolyData *pd) ";

%feature("docstring")  vtkGDCMImageReader::SetFileNames "virtual void
vtkGDCMImageReader::SetFileNames(vtkStringArray *) ";

%feature("docstring")  vtkGDCMImageReader::SetMedicalImageProperties "virtual void
vtkGDCMImageReader::SetMedicalImageProperties(vtkMedicalImageProperties
*pd) ";

%feature("docstring")  vtkGDCMImageReader::vtkBooleanMacro "vtkGDCMImageReader::vtkBooleanMacro(LoadOverlays, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkBooleanMacro "vtkGDCMImageReader::vtkBooleanMacro(LoadIconImage, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkBooleanMacro "vtkGDCMImageReader::vtkBooleanMacro(LossyFlag, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkBooleanMacro "vtkGDCMImageReader::vtkBooleanMacro(ApplyLookupTable, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkBooleanMacro "int
vtkGDCMImageReader::vtkBooleanMacro(ApplyYBRToRGB, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkGetMacro "vtkGDCMImageReader::vtkGetMacro(LoadOverlays, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkGetMacro "vtkGDCMImageReader::vtkGetMacro(LoadIconImage, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkGetMacro "vtkGDCMImageReader::vtkGetMacro(LossyFlag, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkGetMacro "vtkGDCMImageReader::vtkGetMacro(NumberOfOverlays, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkGetMacro "vtkGDCMImageReader::vtkGetMacro(NumberOfIconImages, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkGetMacro "vtkGDCMImageReader::vtkGetMacro(ApplyLookupTable, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkGetMacro "vtkGDCMImageReader::vtkGetMacro(ApplyYBRToRGB, int)
vtkSetMacro(ApplyYBRToRGB ";

%feature("docstring")  vtkGDCMImageReader::vtkGetMacro "vtkGDCMImageReader::vtkGetMacro(ImageFormat, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkGetMacro "vtkGDCMImageReader::vtkGetMacro(PlanarConfiguration, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkGetMacro "vtkGDCMImageReader::vtkGetMacro(Shift, double) ";

%feature("docstring")  vtkGDCMImageReader::vtkGetMacro "vtkGDCMImageReader::vtkGetMacro(Scale, double) ";

%feature("docstring")  vtkGDCMImageReader::vtkGetObjectMacro "vtkGDCMImageReader::vtkGetObjectMacro(DirectionCosines, vtkMatrix4x4)
";

%feature("docstring")  vtkGDCMImageReader::vtkGetObjectMacro "vtkGDCMImageReader::vtkGetObjectMacro(MedicalImageProperties,
vtkMedicalImageProperties) ";

%feature("docstring")  vtkGDCMImageReader::vtkGetObjectMacro "vtkGDCMImageReader::vtkGetObjectMacro(FileNames, vtkStringArray) ";

%feature("docstring")  vtkGDCMImageReader::vtkGetObjectMacro "vtkGDCMImageReader::vtkGetObjectMacro(Curve, vtkPolyData) ";

%feature("docstring")  vtkGDCMImageReader::vtkGetVector3Macro "vtkGDCMImageReader::vtkGetVector3Macro(ImagePositionPatient, double)
";

%feature("docstring")  vtkGDCMImageReader::vtkGetVector6Macro "vtkGDCMImageReader::vtkGetVector6Macro(ImageOrientationPatient,
double) ";

%feature("docstring")  vtkGDCMImageReader::vtkSetMacro "vtkGDCMImageReader::vtkSetMacro(LoadOverlays, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkSetMacro "vtkGDCMImageReader::vtkSetMacro(LoadIconImage, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkSetMacro "vtkGDCMImageReader::vtkSetMacro(LossyFlag, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkSetMacro "vtkGDCMImageReader::vtkSetMacro(ApplyLookupTable, int) ";

%feature("docstring")  vtkGDCMImageReader::vtkTypeRevisionMacro "vtkGDCMImageReader::vtkTypeRevisionMacro(vtkGDCMImageReader,
vtkMedicalImageReader2) ";


// File: classvtkGDCMImageReader2.xml
%feature("docstring") vtkGDCMImageReader2 "C++ includes:
vtkGDCMImageReader2.h ";

%feature("docstring")  vtkGDCMImageReader2::CanReadFile "virtual int
vtkGDCMImageReader2::CanReadFile(const char *fname) ";

%feature("docstring")  vtkGDCMImageReader2::GetDescriptiveName "virtual const char* vtkGDCMImageReader2::GetDescriptiveName() ";

%feature("docstring")  vtkGDCMImageReader2::GetFileExtensions "virtual const char* vtkGDCMImageReader2::GetFileExtensions() ";

%feature("docstring")  vtkGDCMImageReader2::GetIconImage "vtkImageData* vtkGDCMImageReader2::GetIconImage() ";

%feature("docstring")  vtkGDCMImageReader2::GetIconImagePort "vtkAlgorithmOutput* vtkGDCMImageReader2::GetIconImagePort() ";

%feature("docstring")  vtkGDCMImageReader2::GetOverlay "vtkImageData*
vtkGDCMImageReader2::GetOverlay(int i) ";

%feature("docstring")  vtkGDCMImageReader2::GetOverlayPort "vtkAlgorithmOutput* vtkGDCMImageReader2::GetOverlayPort(int index) ";

%feature("docstring")  vtkGDCMImageReader2::PrintSelf "virtual void
vtkGDCMImageReader2::PrintSelf(ostream &os, vtkIndent indent) ";

%feature("docstring")  vtkGDCMImageReader2::SetCurve "virtual void
vtkGDCMImageReader2::SetCurve(vtkPolyData *pd) ";

%feature("docstring")  vtkGDCMImageReader2::SetMedicalImageProperties
"virtual void
vtkGDCMImageReader2::SetMedicalImageProperties(vtkMedicalImageProperties
*pd) ";

%feature("docstring")  vtkGDCMImageReader2::vtkBooleanMacro "vtkGDCMImageReader2::vtkBooleanMacro(LoadOverlays, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkBooleanMacro "vtkGDCMImageReader2::vtkBooleanMacro(LoadIconImage, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkBooleanMacro "vtkGDCMImageReader2::vtkBooleanMacro(LossyFlag, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkBooleanMacro "vtkGDCMImageReader2::vtkBooleanMacro(ApplyLookupTable, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkBooleanMacro "int
vtkGDCMImageReader2::vtkBooleanMacro(ApplyYBRToRGB, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkGetMacro "vtkGDCMImageReader2::vtkGetMacro(LoadOverlays, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkGetMacro "vtkGDCMImageReader2::vtkGetMacro(LoadIconImage, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkGetMacro "vtkGDCMImageReader2::vtkGetMacro(LossyFlag, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkGetMacro "vtkGDCMImageReader2::vtkGetMacro(NumberOfOverlays, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkGetMacro "vtkGDCMImageReader2::vtkGetMacro(NumberOfIconImages, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkGetMacro "vtkGDCMImageReader2::vtkGetMacro(ApplyLookupTable, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkGetMacro "vtkGDCMImageReader2::vtkGetMacro(ApplyYBRToRGB, int)
vtkSetMacro(ApplyYBRToRGB ";

%feature("docstring")  vtkGDCMImageReader2::vtkGetMacro "vtkGDCMImageReader2::vtkGetMacro(ImageFormat, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkGetMacro "vtkGDCMImageReader2::vtkGetMacro(PlanarConfiguration, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkGetMacro "vtkGDCMImageReader2::vtkGetMacro(Shift, double) ";

%feature("docstring")  vtkGDCMImageReader2::vtkGetMacro "vtkGDCMImageReader2::vtkGetMacro(Scale, double) ";

%feature("docstring")  vtkGDCMImageReader2::vtkGetObjectMacro "vtkGDCMImageReader2::vtkGetObjectMacro(DirectionCosines, vtkMatrix4x4)
";

%feature("docstring")  vtkGDCMImageReader2::vtkGetObjectMacro "vtkGDCMImageReader2::vtkGetObjectMacro(Curve, vtkPolyData) ";

%feature("docstring")  vtkGDCMImageReader2::vtkGetVector3Macro "vtkGDCMImageReader2::vtkGetVector3Macro(ImagePositionPatient, double)
";

%feature("docstring")  vtkGDCMImageReader2::vtkGetVector6Macro "vtkGDCMImageReader2::vtkGetVector6Macro(ImageOrientationPatient,
double) ";

%feature("docstring")  vtkGDCMImageReader2::vtkSetMacro "vtkGDCMImageReader2::vtkSetMacro(LoadOverlays, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkSetMacro "vtkGDCMImageReader2::vtkSetMacro(LoadIconImage, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkSetMacro "vtkGDCMImageReader2::vtkSetMacro(LossyFlag, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkSetMacro "vtkGDCMImageReader2::vtkSetMacro(ApplyLookupTable, int) ";

%feature("docstring")  vtkGDCMImageReader2::vtkTypeRevisionMacro "vtkGDCMImageReader2::vtkTypeRevisionMacro(vtkGDCMImageReader2,
vtkMedicalImageReader2) ";


// File: classvtkGDCMImageWriter.xml
%feature("docstring") vtkGDCMImageWriter "C++ includes:
vtkGDCMImageWriter.h ";

%feature("docstring")  vtkGDCMImageWriter::GetDescriptiveName "virtual const char* vtkGDCMImageWriter::GetDescriptiveName() ";

%feature("docstring")  vtkGDCMImageWriter::GetFileExtensions "virtual
const char* vtkGDCMImageWriter::GetFileExtensions() ";

%feature("docstring")  vtkGDCMImageWriter::PrintSelf "virtual void
vtkGDCMImageWriter::PrintSelf(ostream &os, vtkIndent indent) ";

%feature("docstring")  vtkGDCMImageWriter::SetDirectionCosines "virtual void vtkGDCMImageWriter::SetDirectionCosines(vtkMatrix4x4
*matrix) ";

%feature("docstring")
vtkGDCMImageWriter::SetDirectionCosinesFromImageOrientationPatient "virtual void
vtkGDCMImageWriter::SetDirectionCosinesFromImageOrientationPatient(const
double dircos[6]) ";

%feature("docstring")  vtkGDCMImageWriter::SetFileNames "virtual void
vtkGDCMImageWriter::SetFileNames(vtkStringArray *) ";

%feature("docstring")  vtkGDCMImageWriter::SetMedicalImageProperties "virtual void
vtkGDCMImageWriter::SetMedicalImageProperties(vtkMedicalImageProperties
*) ";

%feature("docstring")  vtkGDCMImageWriter::vtkBooleanMacro "vtkGDCMImageWriter::vtkBooleanMacro(LossyFlag, int) ";

%feature("docstring")  vtkGDCMImageWriter::vtkBooleanMacro "vtkGDCMImageWriter::vtkBooleanMacro(FileLowerLeft, int) ";

%feature("docstring")  vtkGDCMImageWriter::vtkGetMacro "vtkGDCMImageWriter::vtkGetMacro(LossyFlag, int) ";

%feature("docstring")  vtkGDCMImageWriter::vtkGetMacro "vtkGDCMImageWriter::vtkGetMacro(Shift, double) ";

%feature("docstring")  vtkGDCMImageWriter::vtkGetMacro "vtkGDCMImageWriter::vtkGetMacro(Scale, double) ";

%feature("docstring")  vtkGDCMImageWriter::vtkGetMacro "vtkGDCMImageWriter::vtkGetMacro(ImageFormat, int) ";

%feature("docstring")  vtkGDCMImageWriter::vtkGetMacro "vtkGDCMImageWriter::vtkGetMacro(FileLowerLeft, int) ";

%feature("docstring")  vtkGDCMImageWriter::vtkGetMacro "vtkGDCMImageWriter::vtkGetMacro(PlanarConfiguration, int) ";

%feature("docstring")  vtkGDCMImageWriter::vtkGetMacro "vtkGDCMImageWriter::vtkGetMacro(CompressionType, int) ";

%feature("docstring")  vtkGDCMImageWriter::vtkGetObjectMacro "vtkGDCMImageWriter::vtkGetObjectMacro(MedicalImageProperties,
vtkMedicalImageProperties) ";

%feature("docstring")  vtkGDCMImageWriter::vtkGetObjectMacro "vtkGDCMImageWriter::vtkGetObjectMacro(FileNames, vtkStringArray) ";

%feature("docstring")  vtkGDCMImageWriter::vtkGetObjectMacro "vtkGDCMImageWriter::vtkGetObjectMacro(DirectionCosines, vtkMatrix4x4)
";

%feature("docstring")  vtkGDCMImageWriter::vtkGetStringMacro "vtkGDCMImageWriter::vtkGetStringMacro(StudyUID) ";

%feature("docstring")  vtkGDCMImageWriter::vtkGetStringMacro "vtkGDCMImageWriter::vtkGetStringMacro(SeriesUID) ";

%feature("docstring")  vtkGDCMImageWriter::vtkSetMacro "vtkGDCMImageWriter::vtkSetMacro(LossyFlag, int) ";

%feature("docstring")  vtkGDCMImageWriter::vtkSetMacro "vtkGDCMImageWriter::vtkSetMacro(Shift, double) ";

%feature("docstring")  vtkGDCMImageWriter::vtkSetMacro "vtkGDCMImageWriter::vtkSetMacro(Scale, double) ";

%feature("docstring")  vtkGDCMImageWriter::vtkSetMacro "vtkGDCMImageWriter::vtkSetMacro(ImageFormat, int) ";

%feature("docstring")  vtkGDCMImageWriter::vtkSetMacro "vtkGDCMImageWriter::vtkSetMacro(FileLowerLeft, int) ";

%feature("docstring")  vtkGDCMImageWriter::vtkSetMacro "vtkGDCMImageWriter::vtkSetMacro(PlanarConfiguration, int) ";

%feature("docstring")  vtkGDCMImageWriter::vtkSetMacro "vtkGDCMImageWriter::vtkSetMacro(CompressionType, int) ";

%feature("docstring")  vtkGDCMImageWriter::vtkSetStringMacro "vtkGDCMImageWriter::vtkSetStringMacro(StudyUID) ";

%feature("docstring")  vtkGDCMImageWriter::vtkSetStringMacro "vtkGDCMImageWriter::vtkSetStringMacro(SeriesUID) ";

%feature("docstring")  vtkGDCMImageWriter::vtkTypeRevisionMacro "vtkGDCMImageWriter::vtkTypeRevisionMacro(vtkGDCMImageWriter,
vtkImageWriter) ";

%feature("docstring")  vtkGDCMImageWriter::Write "virtual void
vtkGDCMImageWriter::Write() ";


// File: classvtkGDCMMedicalImageProperties.xml
%feature("docstring") vtkGDCMMedicalImageProperties "C++ includes:
vtkGDCMMedicalImageProperties.h ";

%feature("docstring")  vtkGDCMMedicalImageProperties::Clear "virtual
void vtkGDCMMedicalImageProperties::Clear() ";

%feature("docstring")  vtkGDCMMedicalImageProperties::PrintSelf "void
vtkGDCMMedicalImageProperties::PrintSelf(ostream &os, vtkIndent
indent) ";

%feature("docstring")
vtkGDCMMedicalImageProperties::vtkTypeRevisionMacro "vtkGDCMMedicalImageProperties::vtkTypeRevisionMacro(vtkGDCMMedicalImageProperties,
vtkMedicalImageProperties) ";


// File: classvtkGDCMPolyDataReader.xml
%feature("docstring") vtkGDCMPolyDataReader "C++ includes:
vtkGDCMPolyDataReader.h ";

%feature("docstring")  vtkGDCMPolyDataReader::PrintSelf "virtual void
vtkGDCMPolyDataReader::PrintSelf(ostream &os, vtkIndent indent) ";

%feature("docstring")  vtkGDCMPolyDataReader::vtkGetObjectMacro "vtkGDCMPolyDataReader::vtkGetObjectMacro(MedicalImageProperties,
vtkMedicalImageProperties) ";

%feature("docstring")  vtkGDCMPolyDataReader::vtkGetObjectMacro "vtkGDCMPolyDataReader::vtkGetObjectMacro(RTStructSetProperties,
vtkRTStructSetProperties) ";

%feature("docstring")  vtkGDCMPolyDataReader::vtkGetStringMacro "vtkGDCMPolyDataReader::vtkGetStringMacro(FileName) ";

%feature("docstring")  vtkGDCMPolyDataReader::vtkSetStringMacro "vtkGDCMPolyDataReader::vtkSetStringMacro(FileName) ";

%feature("docstring")  vtkGDCMPolyDataReader::vtkTypeRevisionMacro "vtkGDCMPolyDataReader::vtkTypeRevisionMacro(vtkGDCMPolyDataReader,
vtkPolyDataAlgorithm) ";


// File: classvtkGDCMPolyDataWriter.xml
%feature("docstring") vtkGDCMPolyDataWriter "C++ includes:
vtkGDCMPolyDataWriter.h ";

%feature("docstring")  vtkGDCMPolyDataWriter::InitializeRTStructSet "void vtkGDCMPolyDataWriter::InitializeRTStructSet(vtkStdString
inDirectory, vtkStdString inStructLabel, vtkStdString inStructName,
vtkStringArray *inROINames, vtkStringArray *inROIAlgorithmName,
vtkStringArray *inROIType) ";

%feature("docstring")  vtkGDCMPolyDataWriter::PrintSelf "virtual void
vtkGDCMPolyDataWriter::PrintSelf(ostream &os, vtkIndent indent) ";

%feature("docstring")
vtkGDCMPolyDataWriter::SetMedicalImageProperties "virtual void
vtkGDCMPolyDataWriter::SetMedicalImageProperties(vtkMedicalImageProperties
*pd) ";

%feature("docstring")  vtkGDCMPolyDataWriter::SetNumberOfInputPorts "void vtkGDCMPolyDataWriter::SetNumberOfInputPorts(int n) ";

%feature("docstring")  vtkGDCMPolyDataWriter::SetRTStructSetProperties
"virtual void
vtkGDCMPolyDataWriter::SetRTStructSetProperties(vtkRTStructSetProperties
*pd) ";

%feature("docstring")  vtkGDCMPolyDataWriter::vtkTypeRevisionMacro "vtkGDCMPolyDataWriter::vtkTypeRevisionMacro(vtkGDCMPolyDataWriter,
vtkPolyDataWriter) ";


// File: classvtkGDCMTesting.xml
%feature("docstring") vtkGDCMTesting "C++ includes: vtkGDCMTesting.h
";

%feature("docstring")  vtkGDCMTesting::PrintSelf "void
vtkGDCMTesting::PrintSelf(ostream &os, vtkIndent indent) ";

%feature("docstring")  vtkGDCMTesting::vtkTypeRevisionMacro "vtkGDCMTesting::vtkTypeRevisionMacro(vtkGDCMTesting, vtkObject) ";


// File: classvtkGDCMThreadedImageReader.xml
%feature("docstring") vtkGDCMThreadedImageReader "C++ includes:
vtkGDCMThreadedImageReader.h ";

%feature("docstring")  vtkGDCMThreadedImageReader::PrintSelf "virtual
void vtkGDCMThreadedImageReader::PrintSelf(ostream &os, vtkIndent
indent) ";

%feature("docstring")  vtkGDCMThreadedImageReader::vtkBooleanMacro "vtkGDCMThreadedImageReader::vtkBooleanMacro(UseShiftScale, int) ";

%feature("docstring")  vtkGDCMThreadedImageReader::vtkGetMacro "vtkGDCMThreadedImageReader::vtkGetMacro(UseShiftScale, int) ";

%feature("docstring")  vtkGDCMThreadedImageReader::vtkSetMacro "vtkGDCMThreadedImageReader::vtkSetMacro(Shift, double) ";

%feature("docstring")  vtkGDCMThreadedImageReader::vtkSetMacro "vtkGDCMThreadedImageReader::vtkSetMacro(Scale, double) ";

%feature("docstring")  vtkGDCMThreadedImageReader::vtkSetMacro "vtkGDCMThreadedImageReader::vtkSetMacro(UseShiftScale, int) ";

%feature("docstring")
vtkGDCMThreadedImageReader::vtkTypeRevisionMacro "vtkGDCMThreadedImageReader::vtkTypeRevisionMacro(vtkGDCMThreadedImageReader,
vtkGDCMImageReader) ";


// File: classvtkGDCMThreadedImageReader2.xml
%feature("docstring") vtkGDCMThreadedImageReader2 "C++ includes:
vtkGDCMThreadedImageReader2.h ";

%feature("docstring")  vtkGDCMThreadedImageReader2::GetFileName "virtual const char* vtkGDCMThreadedImageReader2::GetFileName(int i=0)
";

%feature("docstring")  vtkGDCMThreadedImageReader2::PrintSelf "virtual void vtkGDCMThreadedImageReader2::PrintSelf(ostream &os,
vtkIndent indent) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::SetFileName "virtual void vtkGDCMThreadedImageReader2::SetFileName(const char
*filename) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::SetFileNames "virtual void vtkGDCMThreadedImageReader2::SetFileNames(vtkStringArray
*) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::SplitExtent "int
vtkGDCMThreadedImageReader2::SplitExtent(int splitExt[6], int
startExt[6], int num, int total) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkBooleanMacro "vtkGDCMThreadedImageReader2::vtkBooleanMacro(FileLowerLeft, int) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkBooleanMacro "vtkGDCMThreadedImageReader2::vtkBooleanMacro(LoadOverlays, int) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkBooleanMacro "vtkGDCMThreadedImageReader2::vtkBooleanMacro(UseShiftScale, int) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkGetMacro "vtkGDCMThreadedImageReader2::vtkGetMacro(FileLowerLeft, int) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkGetMacro "vtkGDCMThreadedImageReader2::vtkGetMacro(NumberOfOverlays, int) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkGetMacro "vtkGDCMThreadedImageReader2::vtkGetMacro(DataScalarType, int) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkGetMacro "vtkGDCMThreadedImageReader2::vtkGetMacro(NumberOfScalarComponents,
int) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkGetMacro "vtkGDCMThreadedImageReader2::vtkGetMacro(LoadOverlays, int) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkGetMacro "vtkGDCMThreadedImageReader2::vtkGetMacro(Shift, double) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkGetMacro "vtkGDCMThreadedImageReader2::vtkGetMacro(Scale, double) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkGetMacro "vtkGDCMThreadedImageReader2::vtkGetMacro(UseShiftScale, int) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkGetObjectMacro
"vtkGDCMThreadedImageReader2::vtkGetObjectMacro(FileNames,
vtkStringArray) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkGetVector3Macro
"vtkGDCMThreadedImageReader2::vtkGetVector3Macro(DataOrigin, double)
";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkGetVector3Macro
"vtkGDCMThreadedImageReader2::vtkGetVector3Macro(DataSpacing, double)
";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkGetVector6Macro
"vtkGDCMThreadedImageReader2::vtkGetVector6Macro(DataExtent, int) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkSetMacro "vtkGDCMThreadedImageReader2::vtkSetMacro(FileLowerLeft, int) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkSetMacro "vtkGDCMThreadedImageReader2::vtkSetMacro(DataScalarType, int) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkSetMacro "vtkGDCMThreadedImageReader2::vtkSetMacro(NumberOfScalarComponents,
int) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkSetMacro "vtkGDCMThreadedImageReader2::vtkSetMacro(LoadOverlays, int) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkSetMacro "vtkGDCMThreadedImageReader2::vtkSetMacro(Shift, double) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkSetMacro "vtkGDCMThreadedImageReader2::vtkSetMacro(Scale, double) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkSetMacro "vtkGDCMThreadedImageReader2::vtkSetMacro(UseShiftScale, int) ";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkSetVector3Macro
"vtkGDCMThreadedImageReader2::vtkSetVector3Macro(DataOrigin, double)
";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkSetVector3Macro
"vtkGDCMThreadedImageReader2::vtkSetVector3Macro(DataSpacing, double)
";

%feature("docstring")  vtkGDCMThreadedImageReader2::vtkSetVector6Macro
"vtkGDCMThreadedImageReader2::vtkSetVector6Macro(DataExtent, int) ";

%feature("docstring")
vtkGDCMThreadedImageReader2::vtkTypeRevisionMacro "vtkGDCMThreadedImageReader2::vtkTypeRevisionMacro(vtkGDCMThreadedImageReader2,
vtkThreadedImageAlgorithm) ";


// File: classvtkImageColorViewer.xml
%feature("docstring") vtkImageColorViewer "C++ includes:
vtkImageColorViewer.h ";

%feature("docstring")  vtkImageColorViewer::AddInput "virtual void
vtkImageColorViewer::AddInput(vtkImageData *input) ";

%feature("docstring")  vtkImageColorViewer::AddInputConnection "virtual void
vtkImageColorViewer::AddInputConnection(vtkAlgorithmOutput *input) ";

%feature("docstring")  vtkImageColorViewer::GetColorLevel "virtual
double vtkImageColorViewer::GetColorLevel() ";

%feature("docstring")  vtkImageColorViewer::GetColorWindow "virtual
double vtkImageColorViewer::GetColorWindow() ";

%feature("docstring")  vtkImageColorViewer::GetInput "virtual
vtkImageData* vtkImageColorViewer::GetInput() ";

%feature("docstring")  vtkImageColorViewer::GetOffScreenRendering "virtual int vtkImageColorViewer::GetOffScreenRendering() ";

%feature("docstring")  vtkImageColorViewer::GetOverlayVisibility "double vtkImageColorViewer::GetOverlayVisibility() ";

%feature("docstring")  vtkImageColorViewer::GetPosition "virtual int*
vtkImageColorViewer::GetPosition() ";

%feature("docstring")  vtkImageColorViewer::GetSize "virtual int*
vtkImageColorViewer::GetSize() ";

%feature("docstring")  vtkImageColorViewer::GetSliceMax "virtual int
vtkImageColorViewer::GetSliceMax() ";

%feature("docstring")  vtkImageColorViewer::GetSliceMin "virtual int
vtkImageColorViewer::GetSliceMin() ";

%feature("docstring")  vtkImageColorViewer::GetSliceRange "virtual
void vtkImageColorViewer::GetSliceRange(int range[2]) ";

%feature("docstring")  vtkImageColorViewer::GetSliceRange "virtual
void vtkImageColorViewer::GetSliceRange(int &min, int &max) ";

%feature("docstring")  vtkImageColorViewer::GetSliceRange "virtual
int* vtkImageColorViewer::GetSliceRange() ";

%feature("docstring")  vtkImageColorViewer::GetWindowName "virtual
const char* vtkImageColorViewer::GetWindowName() ";

%feature("docstring")  vtkImageColorViewer::PrintSelf "void
vtkImageColorViewer::PrintSelf(ostream &os, vtkIndent indent) ";

%feature("docstring")  vtkImageColorViewer::Render "virtual void
vtkImageColorViewer::Render(void) ";

%feature("docstring")  vtkImageColorViewer::SetColorLevel "virtual
void vtkImageColorViewer::SetColorLevel(double s) ";

%feature("docstring")  vtkImageColorViewer::SetColorWindow "virtual
void vtkImageColorViewer::SetColorWindow(double s) ";

%feature("docstring")  vtkImageColorViewer::SetDisplayId "virtual
void vtkImageColorViewer::SetDisplayId(void *a) ";

%feature("docstring")  vtkImageColorViewer::SetInput "virtual void
vtkImageColorViewer::SetInput(vtkImageData *in) ";

%feature("docstring")  vtkImageColorViewer::SetInputConnection "virtual void
vtkImageColorViewer::SetInputConnection(vtkAlgorithmOutput *input) ";

%feature("docstring")  vtkImageColorViewer::SetOffScreenRendering "virtual void vtkImageColorViewer::SetOffScreenRendering(int) ";

%feature("docstring")  vtkImageColorViewer::SetOverlayVisibility "void vtkImageColorViewer::SetOverlayVisibility(double vis) ";

%feature("docstring")  vtkImageColorViewer::SetParentId "virtual void
vtkImageColorViewer::SetParentId(void *a) ";

%feature("docstring")  vtkImageColorViewer::SetPosition "virtual void
vtkImageColorViewer::SetPosition(int a, int b) ";

%feature("docstring")  vtkImageColorViewer::SetPosition "virtual void
vtkImageColorViewer::SetPosition(int a[2]) ";

%feature("docstring")  vtkImageColorViewer::SetRenderer "virtual void
vtkImageColorViewer::SetRenderer(vtkRenderer *arg) ";

%feature("docstring")  vtkImageColorViewer::SetRenderWindow "virtual
void vtkImageColorViewer::SetRenderWindow(vtkRenderWindow *arg) ";

%feature("docstring")  vtkImageColorViewer::SetSize "virtual void
vtkImageColorViewer::SetSize(int a, int b) ";

%feature("docstring")  vtkImageColorViewer::SetSize "virtual void
vtkImageColorViewer::SetSize(int a[2]) ";

%feature("docstring")  vtkImageColorViewer::SetSlice "virtual void
vtkImageColorViewer::SetSlice(int s) ";

%feature("docstring")  vtkImageColorViewer::SetSliceOrientation "virtual void vtkImageColorViewer::SetSliceOrientation(int orientation)
";

%feature("docstring")  vtkImageColorViewer::SetSliceOrientationToXY "virtual void vtkImageColorViewer::SetSliceOrientationToXY() ";

%feature("docstring")  vtkImageColorViewer::SetSliceOrientationToXZ "virtual void vtkImageColorViewer::SetSliceOrientationToXZ() ";

%feature("docstring")  vtkImageColorViewer::SetSliceOrientationToYZ "virtual void vtkImageColorViewer::SetSliceOrientationToYZ() ";

%feature("docstring")  vtkImageColorViewer::SetupInteractor "virtual
void vtkImageColorViewer::SetupInteractor(vtkRenderWindowInteractor *)
";

%feature("docstring")  vtkImageColorViewer::SetWindowId "virtual void
vtkImageColorViewer::SetWindowId(void *a) ";

%feature("docstring")  vtkImageColorViewer::UpdateDisplayExtent "virtual void vtkImageColorViewer::UpdateDisplayExtent() ";

%feature("docstring")  vtkImageColorViewer::VTK_LEGACY "vtkImageColorViewer::VTK_LEGACY(int GetWholeZMin()) ";

%feature("docstring")  vtkImageColorViewer::VTK_LEGACY "vtkImageColorViewer::VTK_LEGACY(int GetWholeZMax()) ";

%feature("docstring")  vtkImageColorViewer::VTK_LEGACY "vtkImageColorViewer::VTK_LEGACY(int GetZSlice()) ";

%feature("docstring")  vtkImageColorViewer::VTK_LEGACY "vtkImageColorViewer::VTK_LEGACY(void SetZSlice(int)) ";

%feature("docstring")  vtkImageColorViewer::vtkBooleanMacro "vtkImageColorViewer::vtkBooleanMacro(OffScreenRendering, int) ";

%feature("docstring")  vtkImageColorViewer::vtkGetMacro "vtkImageColorViewer::vtkGetMacro(SliceOrientation, int) ";

%feature("docstring")  vtkImageColorViewer::vtkGetMacro "vtkImageColorViewer::vtkGetMacro(Slice, int) ";

%feature("docstring")  vtkImageColorViewer::vtkGetObjectMacro "vtkImageColorViewer::vtkGetObjectMacro(RenderWindow, vtkRenderWindow)
";

%feature("docstring")  vtkImageColorViewer::vtkGetObjectMacro "vtkImageColorViewer::vtkGetObjectMacro(Renderer, vtkRenderer) ";

%feature("docstring")  vtkImageColorViewer::vtkGetObjectMacro "vtkImageColorViewer::vtkGetObjectMacro(ImageActor, vtkImageActor) ";

%feature("docstring")  vtkImageColorViewer::vtkGetObjectMacro "vtkImageColorViewer::vtkGetObjectMacro(WindowLevel,
vtkImageMapToWindowLevelColors2) ";

%feature("docstring")  vtkImageColorViewer::vtkGetObjectMacro "vtkImageColorViewer::vtkGetObjectMacro(InteractorStyle,
vtkInteractorStyleImage) ";

%feature("docstring")  vtkImageColorViewer::vtkTypeRevisionMacro "vtkImageColorViewer::vtkTypeRevisionMacro(vtkImageColorViewer,
vtkObject) ";


// File: classvtkImageMapToColors16.xml
%feature("docstring") vtkImageMapToColors16 "C++ includes:
vtkImageMapToColors16.h ";

%feature("docstring")  vtkImageMapToColors16::GetMTime "virtual
unsigned long vtkImageMapToColors16::GetMTime() ";

%feature("docstring")  vtkImageMapToColors16::PrintSelf "void
vtkImageMapToColors16::PrintSelf(ostream &os, vtkIndent indent) ";

%feature("docstring")  vtkImageMapToColors16::SetLookupTable "virtual
void vtkImageMapToColors16::SetLookupTable(vtkScalarsToColors *) ";

%feature("docstring")
vtkImageMapToColors16::SetOutputFormatToLuminance "void
vtkImageMapToColors16::SetOutputFormatToLuminance() ";

%feature("docstring")
vtkImageMapToColors16::SetOutputFormatToLuminanceAlpha "void
vtkImageMapToColors16::SetOutputFormatToLuminanceAlpha() ";

%feature("docstring")  vtkImageMapToColors16::SetOutputFormatToRGB "void vtkImageMapToColors16::SetOutputFormatToRGB() ";

%feature("docstring")  vtkImageMapToColors16::SetOutputFormatToRGBA "void vtkImageMapToColors16::SetOutputFormatToRGBA() ";

%feature("docstring")  vtkImageMapToColors16::vtkBooleanMacro "vtkImageMapToColors16::vtkBooleanMacro(PassAlphaToOutput, int) ";

%feature("docstring")  vtkImageMapToColors16::vtkGetMacro "vtkImageMapToColors16::vtkGetMacro(OutputFormat, int) ";

%feature("docstring")  vtkImageMapToColors16::vtkGetMacro "vtkImageMapToColors16::vtkGetMacro(ActiveComponent, int) ";

%feature("docstring")  vtkImageMapToColors16::vtkGetMacro "vtkImageMapToColors16::vtkGetMacro(PassAlphaToOutput, int) ";

%feature("docstring")  vtkImageMapToColors16::vtkGetObjectMacro "vtkImageMapToColors16::vtkGetObjectMacro(LookupTable,
vtkScalarsToColors) ";

%feature("docstring")  vtkImageMapToColors16::vtkSetMacro "vtkImageMapToColors16::vtkSetMacro(OutputFormat, int) ";

%feature("docstring")  vtkImageMapToColors16::vtkSetMacro "vtkImageMapToColors16::vtkSetMacro(ActiveComponent, int) ";

%feature("docstring")  vtkImageMapToColors16::vtkSetMacro "vtkImageMapToColors16::vtkSetMacro(PassAlphaToOutput, int) ";

%feature("docstring")  vtkImageMapToColors16::vtkTypeRevisionMacro "vtkImageMapToColors16::vtkTypeRevisionMacro(vtkImageMapToColors16,
vtkThreadedImageAlgorithm) ";


// File: classvtkImageMapToWindowLevelColors2.xml
%feature("docstring") vtkImageMapToWindowLevelColors2 "C++ includes:
vtkImageMapToWindowLevelColors2.h ";

%feature("docstring")  vtkImageMapToWindowLevelColors2::PrintSelf "void vtkImageMapToWindowLevelColors2::PrintSelf(ostream &os, vtkIndent
indent) ";

%feature("docstring")  vtkImageMapToWindowLevelColors2::vtkGetMacro "vtkImageMapToWindowLevelColors2::vtkGetMacro(Window, double) ";

%feature("docstring")  vtkImageMapToWindowLevelColors2::vtkGetMacro "vtkImageMapToWindowLevelColors2::vtkGetMacro(Level, double) ";

%feature("docstring")  vtkImageMapToWindowLevelColors2::vtkSetMacro "vtkImageMapToWindowLevelColors2::vtkSetMacro(Window, double) ";

%feature("docstring")  vtkImageMapToWindowLevelColors2::vtkSetMacro "vtkImageMapToWindowLevelColors2::vtkSetMacro(Level, double) ";

%feature("docstring")
vtkImageMapToWindowLevelColors2::vtkTypeRevisionMacro "vtkImageMapToWindowLevelColors2::vtkTypeRevisionMacro(vtkImageMapToWindowLevelColors2,
vtkImageMapToColors) ";


// File: classvtkImagePlanarComponentsToComponents.xml
%feature("docstring") vtkImagePlanarComponentsToComponents "C++
includes: vtkImagePlanarComponentsToComponents.h ";

%feature("docstring")  vtkImagePlanarComponentsToComponents::PrintSelf
"void vtkImagePlanarComponentsToComponents::PrintSelf(ostream &os,
vtkIndent indent) ";

%feature("docstring")
vtkImagePlanarComponentsToComponents::vtkTypeRevisionMacro "vtkImagePlanarComponentsToComponents::vtkTypeRevisionMacro(vtkImagePlanarComponentsToComponents,
vtkImageAlgorithm) ";


// File: classvtkImageRGBToYBR.xml
%feature("docstring") vtkImageRGBToYBR "C++ includes:
vtkImageRGBToYBR.h ";

%feature("docstring")  vtkImageRGBToYBR::PrintSelf "void
vtkImageRGBToYBR::PrintSelf(ostream &os, vtkIndent indent) ";

%feature("docstring")  vtkImageRGBToYBR::vtkTypeRevisionMacro "vtkImageRGBToYBR::vtkTypeRevisionMacro(vtkImageRGBToYBR,
vtkThreadedImageAlgorithm) ";


// File: classvtkImageYBRToRGB.xml
%feature("docstring") vtkImageYBRToRGB "C++ includes:
vtkImageYBRToRGB.h ";

%feature("docstring")  vtkImageYBRToRGB::PrintSelf "void
vtkImageYBRToRGB::PrintSelf(ostream &os, vtkIndent indent) ";

%feature("docstring")  vtkImageYBRToRGB::vtkTypeRevisionMacro "vtkImageYBRToRGB::vtkTypeRevisionMacro(vtkImageYBRToRGB,
vtkThreadedImageAlgorithm) ";


// File: classvtkLookupTable16.xml
%feature("docstring") vtkLookupTable16 "C++ includes:
vtkLookupTable16.h ";

%feature("docstring")  vtkLookupTable16::Build "void
vtkLookupTable16::Build() ";

%feature("docstring")  vtkLookupTable16::GetPointer "unsigned short*
vtkLookupTable16::GetPointer(const vtkIdType id) ";

%feature("docstring")  vtkLookupTable16::PrintSelf "void
vtkLookupTable16::PrintSelf(ostream &os, vtkIndent indent) ";

%feature("docstring")  vtkLookupTable16::SetNumberOfTableValues "void
vtkLookupTable16::SetNumberOfTableValues(vtkIdType number) ";

%feature("docstring")  vtkLookupTable16::vtkTypeRevisionMacro "vtkLookupTable16::vtkTypeRevisionMacro(vtkLookupTable16,
vtkLookupTable) ";

%feature("docstring")  vtkLookupTable16::WritePointer "unsigned char
* vtkLookupTable16::WritePointer(const vtkIdType id, const int number)
";


// File: classvtkRTStructSetProperties.xml
%feature("docstring") vtkRTStructSetProperties "C++ includes:
vtkRTStructSetProperties.h ";

%feature("docstring")
vtkRTStructSetProperties::AddContourReferencedFrameOfReference "void
vtkRTStructSetProperties::AddContourReferencedFrameOfReference(vtkIdType
pdnum, const char *classuid, const char *instanceuid) ";

%feature("docstring")
vtkRTStructSetProperties::AddReferencedFrameOfReference "void
vtkRTStructSetProperties::AddReferencedFrameOfReference(const char
*classuid, const char *instanceuid) ";

%feature("docstring")  vtkRTStructSetProperties::AddStructureSetROI "void vtkRTStructSetProperties::AddStructureSetROI(int roinumber, const
char *refframerefuid, const char *roiname, const char
*ROIGenerationAlgorithm, const char *ROIDescription=0) ";

%feature("docstring")
vtkRTStructSetProperties::AddStructureSetROIObservation "void
vtkRTStructSetProperties::AddStructureSetROIObservation(int refnumber,
int observationnumber, const char *rtroiinterpretedtype, const char
*roiinterpreter, const char *roiobservationlabel=0) ";

%feature("docstring")  vtkRTStructSetProperties::Clear "virtual void
vtkRTStructSetProperties::Clear() ";

%feature("docstring")  vtkRTStructSetProperties::DeepCopy "virtual
void vtkRTStructSetProperties::DeepCopy(vtkRTStructSetProperties *p)
";

%feature("docstring")
vtkRTStructSetProperties::GetContourReferencedFrameOfReferenceClassUID
"const char*
vtkRTStructSetProperties::GetContourReferencedFrameOfReferenceClassUID(vtkIdType
pdnum, vtkIdType id) ";

%feature("docstring")
vtkRTStructSetProperties::GetContourReferencedFrameOfReferenceInstanceUID
"const char*
vtkRTStructSetProperties::GetContourReferencedFrameOfReferenceInstanceUID(vtkIdType
pdnum, vtkIdType id) ";

%feature("docstring")
vtkRTStructSetProperties::GetNumberOfContourReferencedFrameOfReferences
"vtkIdType
vtkRTStructSetProperties::GetNumberOfContourReferencedFrameOfReferences()
";

%feature("docstring")
vtkRTStructSetProperties::GetNumberOfContourReferencedFrameOfReferences
"vtkIdType
vtkRTStructSetProperties::GetNumberOfContourReferencedFrameOfReferences(vtkIdType
pdnum) ";

%feature("docstring")
vtkRTStructSetProperties::GetNumberOfReferencedFrameOfReferences "vtkIdType
vtkRTStructSetProperties::GetNumberOfReferencedFrameOfReferences() ";

%feature("docstring")
vtkRTStructSetProperties::GetNumberOfStructureSetROIs "vtkIdType
vtkRTStructSetProperties::GetNumberOfStructureSetROIs() ";

%feature("docstring")
vtkRTStructSetProperties::GetReferencedFrameOfReferenceClassUID "const char*
vtkRTStructSetProperties::GetReferencedFrameOfReferenceClassUID(vtkIdType
id) ";

%feature("docstring")
vtkRTStructSetProperties::GetReferencedFrameOfReferenceInstanceUID "const char*
vtkRTStructSetProperties::GetReferencedFrameOfReferenceInstanceUID(vtkIdType
id) ";

%feature("docstring")
vtkRTStructSetProperties::GetStructureSetObservationNumber "int
vtkRTStructSetProperties::GetStructureSetObservationNumber(vtkIdType
id) ";

%feature("docstring")
vtkRTStructSetProperties::GetStructureSetROIDescription "const char*
vtkRTStructSetProperties::GetStructureSetROIDescription(vtkIdType id)
";

%feature("docstring")
vtkRTStructSetProperties::GetStructureSetROIGenerationAlgorithm "const char*
vtkRTStructSetProperties::GetStructureSetROIGenerationAlgorithm(vtkIdType)
";

%feature("docstring")
vtkRTStructSetProperties::GetStructureSetROIName "const char*
vtkRTStructSetProperties::GetStructureSetROIName(vtkIdType) ";

%feature("docstring")
vtkRTStructSetProperties::GetStructureSetROINumber "int
vtkRTStructSetProperties::GetStructureSetROINumber(vtkIdType id) ";

%feature("docstring")
vtkRTStructSetProperties::GetStructureSetROIObservationLabel "const
char*
vtkRTStructSetProperties::GetStructureSetROIObservationLabel(vtkIdType
id) ";

%feature("docstring")
vtkRTStructSetProperties::GetStructureSetROIRefFrameRefUID "const
char*
vtkRTStructSetProperties::GetStructureSetROIRefFrameRefUID(vtkIdType)
";

%feature("docstring")
vtkRTStructSetProperties::GetStructureSetRTROIInterpretedType "const
char*
vtkRTStructSetProperties::GetStructureSetRTROIInterpretedType(vtkIdType
id) ";

%feature("docstring")  vtkRTStructSetProperties::PrintSelf "void
vtkRTStructSetProperties::PrintSelf(ostream &os, vtkIndent indent) ";

%feature("docstring")  vtkRTStructSetProperties::vtkGetStringMacro "vtkRTStructSetProperties::vtkGetStringMacro(StructureSetLabel) ";

%feature("docstring")  vtkRTStructSetProperties::vtkGetStringMacro "vtkRTStructSetProperties::vtkGetStringMacro(StructureSetName) ";

%feature("docstring")  vtkRTStructSetProperties::vtkGetStringMacro "vtkRTStructSetProperties::vtkGetStringMacro(StructureSetDate) ";

%feature("docstring")  vtkRTStructSetProperties::vtkGetStringMacro "vtkRTStructSetProperties::vtkGetStringMacro(StructureSetTime) ";

%feature("docstring")  vtkRTStructSetProperties::vtkGetStringMacro "vtkRTStructSetProperties::vtkGetStringMacro(SOPInstanceUID) ";

%feature("docstring")  vtkRTStructSetProperties::vtkGetStringMacro "vtkRTStructSetProperties::vtkGetStringMacro(StudyInstanceUID) ";

%feature("docstring")  vtkRTStructSetProperties::vtkGetStringMacro "vtkRTStructSetProperties::vtkGetStringMacro(SeriesInstanceUID) ";

%feature("docstring")  vtkRTStructSetProperties::vtkGetStringMacro "vtkRTStructSetProperties::vtkGetStringMacro(ReferenceSeriesInstanceUID)
";

%feature("docstring")  vtkRTStructSetProperties::vtkGetStringMacro "vtkRTStructSetProperties::vtkGetStringMacro(ReferenceFrameOfReferenceUID)
";

%feature("docstring")  vtkRTStructSetProperties::vtkSetStringMacro "vtkRTStructSetProperties::vtkSetStringMacro(StructureSetLabel) ";

%feature("docstring")  vtkRTStructSetProperties::vtkSetStringMacro "vtkRTStructSetProperties::vtkSetStringMacro(StructureSetName) ";

%feature("docstring")  vtkRTStructSetProperties::vtkSetStringMacro "vtkRTStructSetProperties::vtkSetStringMacro(StructureSetDate) ";

%feature("docstring")  vtkRTStructSetProperties::vtkSetStringMacro "vtkRTStructSetProperties::vtkSetStringMacro(StructureSetTime) ";

%feature("docstring")  vtkRTStructSetProperties::vtkSetStringMacro "vtkRTStructSetProperties::vtkSetStringMacro(SOPInstanceUID) ";

%feature("docstring")  vtkRTStructSetProperties::vtkSetStringMacro "vtkRTStructSetProperties::vtkSetStringMacro(StudyInstanceUID) ";

%feature("docstring")  vtkRTStructSetProperties::vtkSetStringMacro "vtkRTStructSetProperties::vtkSetStringMacro(SeriesInstanceUID) ";

%feature("docstring")  vtkRTStructSetProperties::vtkSetStringMacro "vtkRTStructSetProperties::vtkSetStringMacro(ReferenceSeriesInstanceUID)
";

%feature("docstring")  vtkRTStructSetProperties::vtkSetStringMacro "vtkRTStructSetProperties::vtkSetStringMacro(ReferenceFrameOfReferenceUID)
";

%feature("docstring")  vtkRTStructSetProperties::vtkTypeRevisionMacro
"vtkRTStructSetProperties::vtkTypeRevisionMacro(vtkRTStructSetProperties,
vtkObject) ";


// File: classgdcm_1_1Waveform.xml
%feature("docstring") gdcm::Waveform "

Waveform class.

C++ includes: gdcmWaveform.h ";

%feature("docstring")  gdcm::Waveform::Waveform "gdcm::Waveform::Waveform() ";


// File: classstd_1_1weak__ptr.xml
%feature("docstring") std::weak_ptr "

STL class. ";


// File: classstd_1_1wfstream.xml
%feature("docstring") std::wfstream "

STL class. ";


// File: classstd_1_1wifstream.xml
%feature("docstring") std::wifstream "

STL class. ";


// File: classstd_1_1wios.xml
%feature("docstring") std::wios "

STL class. ";


// File: classstd_1_1wistream.xml
%feature("docstring") std::wistream "

STL class. ";


// File: classstd_1_1wistringstream.xml
%feature("docstring") std::wistringstream "

STL class. ";


// File: classstd_1_1wofstream.xml
%feature("docstring") std::wofstream "

STL class. ";


// File: classstd_1_1wostream.xml
%feature("docstring") std::wostream "

STL class. ";


// File: classstd_1_1wostringstream.xml
%feature("docstring") std::wostringstream "

STL class. ";


// File: classgdcm_1_1Writer.xml
%feature("docstring") gdcm::Writer "

Writer ala DOM (Document Object Model) This class is a non-validating
writer, it will only performs well- formedness check only.

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

WARNING:   gdcm::Writer cannot write a DataSet if no SOP Instance UID
(0008,0018) is found, unless a DICOMDIR is being written out

See:   Reader DataSet File

C++ includes: gdcmWriter.h ";

%feature("docstring")  gdcm::Writer::Writer "gdcm::Writer::Writer()
";

%feature("docstring")  gdcm::Writer::~Writer "virtual
gdcm::Writer::~Writer() ";

%feature("docstring")  gdcm::Writer::CheckFileMetaInformationOff "void gdcm::Writer::CheckFileMetaInformationOff() ";

%feature("docstring")  gdcm::Writer::CheckFileMetaInformationOn "void
gdcm::Writer::CheckFileMetaInformationOn() ";

%feature("docstring")  gdcm::Writer::GetFile "File&
gdcm::Writer::GetFile() ";

%feature("docstring")  gdcm::Writer::SetCheckFileMetaInformation "void gdcm::Writer::SetCheckFileMetaInformation(bool b)

Undocumented function, do not use (= leave default) ";

%feature("docstring")  gdcm::Writer::SetFile "void
gdcm::Writer::SetFile(const File &f)

Set/Get the DICOM file ( DataSet + Header) ";

%feature("docstring")  gdcm::Writer::SetFileName "void
gdcm::Writer::SetFileName(const char *filename_native)

Set the filename of DICOM file to write: ";

%feature("docstring")  gdcm::Writer::SetStream "void
gdcm::Writer::SetStream(std::ostream &output_stream)

Set user ostream buffer. ";

%feature("docstring")  gdcm::Writer::Write "virtual bool
gdcm::Writer::Write()

Main function to tell the writer to write. ";


// File: classstd_1_1wstring.xml
%feature("docstring") std::wstring "

STL class. ";


// File: classstd_1_1wstringstream.xml
%feature("docstring") std::wstringstream "

STL class. ";


// File: classgdcm_1_1XMLDictReader.xml
%feature("docstring") gdcm::XMLDictReader "

Class for representing a XMLDictReader.

bla Will read the DICOMV3.xml file

C++ includes: gdcmXMLDictReader.h ";

%feature("docstring")  gdcm::XMLDictReader::XMLDictReader "gdcm::XMLDictReader::XMLDictReader() ";

%feature("docstring")  gdcm::XMLDictReader::~XMLDictReader "gdcm::XMLDictReader::~XMLDictReader() ";

%feature("docstring")  gdcm::XMLDictReader::CharacterDataHandler "void gdcm::XMLDictReader::CharacterDataHandler(const char *data, int
length) ";

%feature("docstring")  gdcm::XMLDictReader::EndElement "void
gdcm::XMLDictReader::EndElement(const char *name) ";

%feature("docstring")  gdcm::XMLDictReader::GetDict "const Dict&
gdcm::XMLDictReader::GetDict() ";

%feature("docstring")  gdcm::XMLDictReader::StartElement "void
gdcm::XMLDictReader::StartElement(const char *name, const char **atts)
";


// File: classgdcm_1_1XMLPrinter.xml
%feature("docstring") gdcm::XMLPrinter "C++ includes:
gdcmXMLPrinter.h ";

%feature("docstring")  gdcm::XMLPrinter::XMLPrinter "gdcm::XMLPrinter::XMLPrinter() ";

%feature("docstring")  gdcm::XMLPrinter::~XMLPrinter "virtual
gdcm::XMLPrinter::~XMLPrinter() ";

%feature("docstring")  gdcm::XMLPrinter::GetPrintStyle "PrintStyles
gdcm::XMLPrinter::GetPrintStyle() const ";

%feature("docstring")  gdcm::XMLPrinter::HandleBulkData "virtual void
gdcm::XMLPrinter::HandleBulkData(const char *uuid, const
TransferSyntax &ts, const char *bulkdata, size_t bulklen)

Virtual function mecanism to allow application programmer to override
the default mecanism for BulkData handling. By default GDCM will
simply discard the BulkData and only write the UUID ";

%feature("docstring")  gdcm::XMLPrinter::Print "void
gdcm::XMLPrinter::Print(std::ostream &os) ";

%feature("docstring")  gdcm::XMLPrinter::PrintDataSet "void
gdcm::XMLPrinter::PrintDataSet(const DataSet &ds, const TransferSyntax
&ts, std::ostream &os) ";

%feature("docstring")  gdcm::XMLPrinter::SetFile "void
gdcm::XMLPrinter::SetFile(File const &f) ";

%feature("docstring")  gdcm::XMLPrinter::SetStyle "void
gdcm::XMLPrinter::SetStyle(PrintStyles ps) ";


// File: classgdcm_1_1XMLPrivateDictReader.xml
%feature("docstring") gdcm::XMLPrivateDictReader "

Class for representing a XMLPrivateDictReader.

bla Will read the Private.xml file

C++ includes: gdcmXMLPrivateDictReader.h ";

%feature("docstring")
gdcm::XMLPrivateDictReader::XMLPrivateDictReader "gdcm::XMLPrivateDictReader::XMLPrivateDictReader() ";

%feature("docstring")
gdcm::XMLPrivateDictReader::~XMLPrivateDictReader "gdcm::XMLPrivateDictReader::~XMLPrivateDictReader() ";

%feature("docstring")
gdcm::XMLPrivateDictReader::CharacterDataHandler "void
gdcm::XMLPrivateDictReader::CharacterDataHandler(const char *data, int
length) ";

%feature("docstring")  gdcm::XMLPrivateDictReader::EndElement "void
gdcm::XMLPrivateDictReader::EndElement(const char *name) ";

%feature("docstring")  gdcm::XMLPrivateDictReader::GetPrivateDict "const PrivateDict& gdcm::XMLPrivateDictReader::GetPrivateDict() ";

%feature("docstring")  gdcm::XMLPrivateDictReader::StartElement "void
gdcm::XMLPrivateDictReader::StartElement(const char *name, const char
**atts) ";


// File: namespacegdcm.xml
%feature("docstring")  gdcm::network::backslash "ignore_char const
gdcm::backslash('\\\\\\\\') ";

%feature("docstring")  gdcm::network::GetVRFromTag "VR::VRType
gdcm::GetVRFromTag(Tag const &tag) ";

%feature("docstring")  gdcm::network::to_string "std::string
gdcm::to_string(Float data) ";

%feature("docstring")  gdcm::network::TYPETOENCODING "gdcm::TYPETOENCODING(SQ, VRBINARY, unsigned char) TYPETOENCODING(UN ";


// File: namespacegdcm_1_1network.xml
%feature("docstring")  gdcm::network::GetStateIndex "int
gdcm::network::GetStateIndex(EStateID inState) ";


// File: namespacegdcm_1_1SegmentHelper.xml


// File: namespacegdcm_1_1terminal.xml
%feature("docstring")  gdcm::terminal::setattribute "GDCM_EXPORT
std::string gdcm::terminal::setattribute(Attribute att) ";

%feature("docstring")  gdcm::terminal::setbgcolor "GDCM_EXPORT
std::string gdcm::terminal::setbgcolor(Color c) ";

%feature("docstring")  gdcm::terminal::setfgcolor "GDCM_EXPORT
std::string gdcm::terminal::setfgcolor(Color c) ";

%feature("docstring")  gdcm::terminal::setmode "GDCM_EXPORT void
gdcm::terminal::setmode(Mode m) ";


// File: namespacestd.xml


// File: gdcm2pnm_8dox.xml


// File: gdcm2vtk_8dox.xml


// File: gdcmAAbortPDU_8h.xml


// File: gdcmAAssociateACPDU_8h.xml


// File: gdcmAAssociateRJPDU_8h.xml


// File: gdcmAAssociateRQPDU_8h.xml


// File: gdcmAbstractSyntax_8h.xml


// File: gdcmanon_8dox.xml


// File: gdcmAnonymizeEvent_8h.xml


// File: gdcmAnonymizer_8h.xml


// File: gdcmApplicationContext_8h.xml


// File: gdcmApplicationEntity_8h.xml


// File: gdcmAReleaseRPPDU_8h.xml


// File: gdcmAReleaseRQPDU_8h.xml


// File: gdcmARTIMTimer_8h.xml


// File: gdcmASN1_8h.xml


// File: gdcmAsynchronousOperationsWindowSub_8h.xml


// File: gdcmAttribute_8h.xml


// File: gdcmAudioCodec_8h.xml


// File: gdcmBase64_8h.xml


// File: gdcmBaseCompositeMessage_8h.xml


// File: gdcmBasePDU_8h.xml


// File: gdcmBaseRootQuery_8h.xml


// File: gdcmBasicOffsetTable_8h.xml


// File: gdcmBitmap_8h.xml


// File: gdcmBitmapToBitmapFilter_8h.xml


// File: gdcmBoxRegion_8h.xml


// File: gdcmByteBuffer_8h.xml


// File: gdcmByteSwap_8h.xml


// File: gdcmByteSwapFilter_8h.xml


// File: gdcmByteValue_8h.xml


// File: gdcmCAPICryptoFactory_8h.xml


// File: gdcmCAPICryptographicMessageSyntax_8h.xml


// File: gdcmCEchoMessages_8h.xml


// File: gdcmCFindMessages_8h.xml


// File: gdcmCMoveMessages_8h.xml


// File: gdcmCodec_8h.xml


// File: gdcmCoder_8h.xml


// File: gdcmCodeString_8h.xml


// File: gdcmCommand_8h.xml


// File: gdcmCommandDataSet_8h.xml


// File: gdcmCompositeMessageFactory_8h.xml


// File: gdcmCompositeNetworkFunctions_8h.xml


// File: gdcmConstCharWrapper_8h.xml


// File: gdcmconv_8dox.xml


// File: gdcmCP246ExplicitDataElement_8h.xml


// File: gdcmCryptoFactory_8h.xml


// File: gdcmCryptographicMessageSyntax_8h.xml


// File: gdcmCSAElement_8h.xml


// File: gdcmCSAHeader_8h.xml


// File: gdcmCSAHeaderDict_8h.xml


// File: gdcmCSAHeaderDictEntry_8h.xml


// File: gdcmCStoreMessages_8h.xml


// File: gdcmCurve_8h.xml


// File: gdcmDataElement_8h.xml


// File: gdcmDataEvent_8h.xml


// File: gdcmDataSet_8h.xml


// File: gdcmDataSetEvent_8h.xml


// File: gdcmDataSetHelper_8h.xml


// File: gdcmDecoder_8h.xml


// File: gdcmDefinedTerms_8h.xml


// File: gdcmDeflateStream_8h.xml


// File: gdcmDefs_8h.xml


// File: gdcmDeltaEncodingCodec_8h.xml


// File: gdcmDICOMDIR_8h.xml


// File: gdcmDICOMDIRGenerator_8h.xml


// File: gdcmDict_8h.xml


// File: gdcmDictConverter_8h.xml


// File: gdcmDictEntry_8h.xml


// File: gdcmDictPrinter_8h.xml


// File: gdcmDicts_8h.xml


// File: gdcmdiff_8dox.xml


// File: gdcmDIMSE_8h.xml


// File: gdcmDirectionCosines_8h.xml


// File: gdcmDirectory_8h.xml


// File: gdcmDirectoryHelper_8h.xml


// File: gdcmDummyValueGenerator_8h.xml


// File: gdcmdump_8dox.xml


// File: gdcmDumper_8h.xml


// File: gdcmElement_8h.xml


// File: gdcmEncapsulatedDocument_8h.xml


// File: gdcmEnumeratedValues_8h.xml


// File: gdcmEvent_8h.xml


// File: gdcmException_8h.xml


// File: gdcmExplicitDataElement_8h.xml


// File: gdcmExplicitImplicitDataElement_8h.xml


// File: gdcmFiducials_8h.xml


// File: gdcmFile_8h.xml


// File: gdcmFileAnonymizer_8h.xml


// File: gdcmFileChangeTransferSyntax_8h.xml


// File: gdcmFileDerivation_8h.xml


// File: gdcmFileExplicitFilter_8h.xml


// File: gdcmFileMetaInformation_8h.xml


// File: gdcmFilename_8h.xml


// File: gdcmFileNameEvent_8h.xml


// File: gdcmFilenameGenerator_8h.xml


// File: gdcmFileSet_8h.xml


// File: gdcmFileStreamer_8h.xml


// File: gdcmFindPatientRootQuery_8h.xml


// File: gdcmFindStudyRootQuery_8h.xml


// File: gdcmFragment_8h.xml


// File: gdcmgendir_8dox.xml


// File: gdcmGlobal_8h.xml


// File: gdcmGroupDict_8h.xml


// File: gdcmIconImage_8h.xml


// File: gdcmIconImageFilter_8h.xml


// File: gdcmIconImageGenerator_8h.xml


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


// File: gdcmImageRegionReader_8h.xml


// File: gdcmImageToImageFilter_8h.xml


// File: gdcmImageWriter_8h.xml


// File: gdcmimg_8dox.xml


// File: gdcmImplementationClassUIDSub_8h.xml


// File: gdcmImplementationUIDSub_8h.xml


// File: gdcmImplementationVersionNameSub_8h.xml


// File: gdcmImplicitDataElement_8h.xml


// File: gdcminfo_8dox.xml


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


// File: gdcmJSON_8h.xml


// File: gdcmKAKADUCodec_8h.xml


// File: gdcmLegacyMacro_8h.xml


// File: gdcmLO_8h.xml


// File: gdcmLookupTable_8h.xml


// File: gdcmMacro_8h.xml


// File: gdcmMacroEntry_8h.xml


// File: gdcmMacros_8h.xml


// File: gdcmMaximumLengthSub_8h.xml


// File: gdcmMD5_8h.xml


// File: gdcmMediaStorage_8h.xml


// File: gdcmMeshPrimitive_8h.xml


// File: gdcmModule_8h.xml


// File: gdcmModuleEntry_8h.xml


// File: gdcmModules_8h.xml


// File: gdcmMovePatientRootQuery_8h.xml


// File: gdcmMoveStudyRootQuery_8h.xml


// File: gdcmNestedModuleEntries_8h.xml


// File: gdcmNetworkEvents_8h.xml


// File: gdcmNetworkStateID_8h.xml


// File: gdcmObject_8h.xml


// File: gdcmOpenSSLCryptoFactory_8h.xml


// File: gdcmOpenSSLCryptographicMessageSyntax_8h.xml


// File: gdcmOpenSSLP7CryptoFactory_8h.xml


// File: gdcmOpenSSLP7CryptographicMessageSyntax_8h.xml


// File: gdcmOrientation_8h.xml


// File: gdcmOverlay_8h.xml


// File: gdcmpap3_8dox.xml


// File: gdcmParseException_8h.xml


// File: gdcmParser_8h.xml


// File: gdcmPatient_8h.xml


// File: gdcmPDataTFPDU_8h.xml


// File: gdcmPDBElement_8h.xml


// File: gdcmPDBHeader_8h.xml


// File: gdcmpdf_8dox.xml


// File: gdcmPDFCodec_8h.xml


// File: gdcmPDUFactory_8h.xml


// File: gdcmPersonName_8h.xml


// File: gdcmPGXCodec_8h.xml


// File: gdcmPhotometricInterpretation_8h.xml


// File: gdcmPixelFormat_8h.xml


// File: gdcmPixmap_8h.xml


// File: gdcmPixmapReader_8h.xml


// File: gdcmPixmapToPixmapFilter_8h.xml


// File: gdcmPixmapWriter_8h.xml


// File: gdcmPNMCodec_8h.xml


// File: gdcmPreamble_8h.xml


// File: gdcmPresentationContext_8h.xml


// File: gdcmPresentationContextAC_8h.xml


// File: gdcmPresentationContextGenerator_8h.xml


// File: gdcmPresentationContextRQ_8h.xml


// File: gdcmPresentationDataValue_8h.xml


// File: gdcmPrinter_8h.xml


// File: gdcmPrivateTag_8h.xml


// File: gdcmProgressEvent_8h.xml


// File: gdcmPVRGCodec_8h.xml


// File: gdcmPythonFilter_8h.xml


// File: gdcmQueryBase_8h.xml


// File: gdcmQueryFactory_8h.xml


// File: gdcmQueryImage_8h.xml


// File: gdcmQueryPatient_8h.xml


// File: gdcmQuerySeries_8h.xml


// File: gdcmQueryStudy_8h.xml


// File: gdcmraw_8dox.xml


// File: gdcmRAWCodec_8h.xml


// File: gdcmReader_8h.xml


// File: gdcmRegion_8h.xml


// File: gdcmRescaler_8h.xml


// File: gdcmRLECodec_8h.xml


// File: gdcmRoleSelectionSub_8h.xml


// File: gdcmscanner_8dox.xml


// File: gdcmScanner_8h.xml


// File: gdcmscu_8dox.xml


// File: gdcmSegment_8h.xml


// File: gdcmSegmentedPaletteColorLookupTable_8h.xml


// File: gdcmSegmentHelper_8h.xml


// File: gdcmSegmentReader_8h.xml


// File: gdcmSegmentWriter_8h.xml


// File: gdcmSequenceOfFragments_8h.xml


// File: gdcmSequenceOfItems_8h.xml


// File: gdcmSerieHelper_8h.xml


// File: gdcmSeries_8h.xml


// File: gdcmServiceClassApplicationInformation_8h.xml


// File: gdcmServiceClassUser_8h.xml


// File: gdcmSHA1_8h.xml


// File: gdcmSimpleSubjectWatcher_8h.xml


// File: gdcmSmartPointer_8h.xml


// File: gdcmSOPClassExtendedNegociationSub_8h.xml


// File: gdcmSOPClassUIDToIOD_8h.xml


// File: gdcmSorter_8h.xml


// File: gdcmSpacing_8h.xml


// File: gdcmSpectroscopy_8h.xml


// File: gdcmSplitMosaicFilter_8h.xml


// File: gdcmStaticAssert_8h.xml


// File: gdcmStreamImageReader_8h.xml


// File: gdcmStreamImageWriter_8h.xml


// File: gdcmString_8h.xml


// File: gdcmStringFilter_8h.xml


// File: gdcmStudy_8h.xml


// File: gdcmSubject_8h.xml


// File: gdcmSurface_8h.xml


// File: gdcmSurfaceHelper_8h.xml


// File: gdcmSurfaceReader_8h.xml


// File: gdcmSurfaceWriter_8h.xml


// File: gdcmSwapCode_8h.xml


// File: gdcmSwapper_8h.xml


// File: gdcmSystem_8h.xml


// File: gdcmTable_8h.xml


// File: gdcmTableEntry_8h.xml


// File: gdcmTableReader_8h.xml


// File: gdcmTag_8h.xml


// File: gdcmTagPath_8h.xml


// File: gdcmTagToVR_8h.xml


// File: gdcmtar_8dox.xml


// File: gdcmTerminal_8h.xml


// File: gdcmTestDriver_8h.xml


// File: gdcmTesting_8h.xml


// File: gdcmTrace_8h.xml


// File: gdcmTransferSyntax_8h.xml


// File: gdcmTransferSyntaxSub_8h.xml


// File: gdcmType_8h.xml


// File: gdcmTypes_8h.xml


// File: gdcmUIDGenerator_8h.xml


// File: gdcmUIDs_8h.xml


// File: gdcmULAction_8h.xml


// File: gdcmULActionAA_8h.xml


// File: gdcmULActionAE_8h.xml


// File: gdcmULActionAR_8h.xml


// File: gdcmULActionDT_8h.xml


// File: gdcmULBasicCallback_8h.xml


// File: gdcmULConnection_8h.xml


// File: gdcmULConnectionCallback_8h.xml


// File: gdcmULConnectionInfo_8h.xml


// File: gdcmULConnectionManager_8h.xml


// File: gdcmULEvent_8h.xml


// File: gdcmULTransitionTable_8h.xml


// File: gdcmULWritingCallback_8h.xml


// File: gdcmUNExplicitDataElement_8h.xml


// File: gdcmUNExplicitImplicitDataElement_8h.xml


// File: gdcmUnpacker12Bits_8h.xml


// File: gdcmUsage_8h.xml


// File: gdcmUserInformation_8h.xml


// File: gdcmUUIDGenerator_8h.xml


// File: gdcmValidate_8h.xml


// File: gdcmValue_8h.xml


// File: gdcmValueIO_8h.xml


// File: gdcmVersion_8h.xml


// File: gdcmviewer_8dox.xml


// File: gdcmVL_8h.xml


// File: gdcmVM_8h.xml


// File: gdcmVR_8h.xml


// File: gdcmVR16ExplicitDataElement_8h.xml


// File: gdcmWaveform_8h.xml


// File: gdcmWin32_8h.xml


// File: gdcmWriter_8h.xml


// File: gdcmxml_8dox.xml


// File: gdcmXMLDictReader_8h.xml


// File: gdcmXMLPrinter_8h.xml


// File: gdcmXMLPrivateDictReader_8h.xml


// File: README_8txt.xml


// File: TestsList_8txt.xml


// File: vtkGDCMImageReader_8h.xml


// File: vtkGDCMImageReader2_8h.xml


// File: vtkGDCMImageWriter_8h.xml


// File: vtkGDCMMedicalImageProperties_8h.xml


// File: vtkGDCMPolyDataReader_8h.xml


// File: vtkGDCMPolyDataWriter_8h.xml


// File: vtkGDCMTesting_8h.xml


// File: vtkGDCMThreadedImageReader_8h.xml


// File: vtkGDCMThreadedImageReader2_8h.xml


// File: vtkImageColorViewer_8h.xml


// File: vtkImageMapToColors16_8h.xml


// File: vtkImageMapToWindowLevelColors2_8h.xml


// File: vtkImagePlanarComponentsToComponents_8h.xml


// File: vtkImageRGBToYBR_8h.xml


// File: vtkImageYBRToRGB_8h.xml


// File: vtkLookupTable16_8h.xml


// File: vtkRTStructSetProperties_8h.xml


// File: gdcm2pnm.xml


// File: gdcm2vtk.xml


// File: gdcmanon.xml


// File: gdcmconv.xml


// File: gdcmdiff.xml


// File: gdcmdump.xml


// File: gdcmgendir.xml


// File: gdcmimg.xml


// File: gdcminfo.xml


// File: gdcmpap3.xml


// File: gdcmpdf.xml


// File: gdcmraw.xml


// File: gdcmscanner.xml


// File: gdcmscu.xml


// File: gdcmtar.xml


// File: gdcmviewer.xml


// File: gdcmxml.xml


// File: todo.xml


// File: deprecated.xml


// File: bug.xml


// File: dir_48be02fb937e08881437e02515417ab2.xml


// File: dir_dbdfee04788ce02e68d05e06d5e6d98f.xml


// File: dir_422e8974cbd0b7203ed9c70ede735192.xml


// File: dir_fc2dbd93ff698b14d78f486017ee822b.xml


// File: dir_63e84970519399936bea68aa0151439e.xml


// File: dir_a3a231e2bd7f702d85036607d7d87964.xml


// File: dir_bfc3201f3b82d7ccf14c524caa3c389b.xml


// File: dir_9a6580727919559370fc2250dcaca6b8.xml


// File: dir_087222ad62d2f517f4e0198672951648.xml


// File: dir_2a74275ceded0a5f3b0fb2e9bd792825.xml


// File: dir_acafdc7d686494cf0735517ddc7a7669.xml


// File: dir_d2ab22b73e3ee89be3a207288d7a9056.xml


// File: AWTMedical3_8java-example.xml


// File: BasicAnonymizer_8cs-example.xml


// File: BasicImageAnonymizer_8cs-example.xml


// File: CastConvertPhilips_8py-example.xml


// File: ChangePrivateTags_8cxx-example.xml


// File: ChangeSequenceUltrasound_8cxx-example.xml


// File: CheckBigEndianBug_8cxx-example.xml


// File: ClinicalTrialAnnotate_8cxx-example.xml


// File: ClinicalTrialIdentificationWorkflow_8cs-example.xml


// File: CompressImage_8cxx-example.xml


// File: CompressLossyJPEG_8cs-example.xml


// File: Compute3DSpacing_8cxx-example.xml


// File: Convert16BitsTo8Bits_8cxx-example.xml


// File: ConvertMPL_8py-example.xml


// File: ConvertMultiFrameToSingleFrame_8cxx-example.xml


// File: ConvertNumpy_8py-example.xml


// File: ConvertPIL_8py-example.xml


// File: ConvertRGBToLuminance_8cxx-example.xml


// File: ConvertSingleBitTo8Bits_8cxx-example.xml


// File: ConvertToQImage_8cxx-example.xml


// File: CreateARGBImage_8cxx-example.xml


// File: CreateCMYKImage_8cxx-example.xml


// File: CreateFakeRTDOSE_8cxx-example.xml


// File: CreateJPIPDataSet_8cxx-example.xml


// File: CreateRAWStorage_8py-example.xml


// File: csa2img_8cxx-example.xml


// File: CStoreQtProgress_8cxx-example.xml


// File: DecompressImage_8cs-example.xml


// File: DecompressImage_8java-example.xml


// File: DecompressImage_8py-example.xml


// File: DecompressImageMultiframe_8cs-example.xml


// File: DecompressJPEGFile_8cs-example.xml


// File: DecompressPixmap_8java-example.xml


// File: DiffFile_8cxx-example.xml


// File: DiscriminateVolume_8cxx-example.xml


// File: DumbAnonymizer_8py-example.xml


// File: DumpADAC_8cxx-example.xml


// File: DumpExamCard_8cxx-example.xml


// File: DumpGEMSMovieGroup_8cxx-example.xml


// File: DumpImageHeaderInfo_8cxx-example.xml


// File: DumpPhilipsECHO_8cxx-example.xml


// File: DumpToSQLITE3_8cxx-example.xml


// File: DuplicatePCDE_8cxx-example.xml


// File: ELSCINT1WaveToText_8cxx-example.xml


// File: EncapsulateFileInRawData_8cxx-example.xml


// File: ExtractEncapsulatedFile_8cs-example.xml


// File: ExtractEncryptedContent_8cxx-example.xml


// File: ExtractIconFromFile_8cxx-example.xml


// File: ExtractImageRegion_8cs-example.xml


// File: ExtractImageRegion_8java-example.xml


// File: ExtractImageRegionWithLUT_8cs-example.xml


// File: Extracting_All_Resolution_8cxx-example.xml


// File: ExtractOneFrame_8cs-example.xml


// File: Fake_Image_Using_Stream_Image_Writer_8cxx-example.xml


// File: FileAnonymize_8cs-example.xml


// File: FileAnonymize_8java-example.xml


// File: FileChangeTS_8cs-example.xml


// File: FileChangeTSLossy_8cs-example.xml


// File: FileStreaming_8cs-example.xml


// File: FindAllPatientName_8py-example.xml


// File: FixBrokenJ2K_8cxx-example.xml


// File: FixCommaBug_8py-example.xml


// File: FixJAIBugJPEGLS_8cxx-example.xml


// File: gdcmorthoplanes_8cxx-example.xml


// File: gdcmreslice_8cxx-example.xml


// File: gdcmrtionplan_8cxx-example.xml


// File: gdcmrtplan_8cxx-example.xml


// File: gdcmscene_8cxx-example.xml


// File: gdcmtexture_8cxx-example.xml


// File: gdcmvolume_8cxx-example.xml


// File: GenAllVR_8cxx-example.xml


// File: GenerateDICOMDIR_8cs-example.xml


// File: GenerateRTSTRUCT_8cxx-example.xml


// File: GenerateStandardSOPClasses_8cxx-example.xml


// File: GenFakeIdentifyFile_8cxx-example.xml


// File: GenFakeImage_8cxx-example.xml


// File: GenLongSeqs_8cxx-example.xml


// File: GenSeqs_8cxx-example.xml


// File: GetArray_8cs-example.xml


// File: GetJPEGSamplePrecision_8cxx-example.xml


// File: GetPortionCSAHeader_8py-example.xml


// File: GetSequenceUltrasound_8cxx-example.xml


// File: GetSubSequenceData_8cxx-example.xml


// File: headsq2dcm_8py-example.xml


// File: HelloActiviz_8cs-example.xml


// File: HelloActiviz2_8cs-example.xml


// File: HelloActiviz3_8cs-example.xml


// File: HelloActiviz4_8cs-example.xml


// File: HelloActiviz5_8cs-example.xml


// File: HelloSimple_8java-example.xml


// File: HelloVizWorld_8cxx-example.xml


// File: HelloVTKWorld_8cs-example.xml


// File: HelloVTKWorld_8java-example.xml


// File: HelloVTKWorld2_8cs-example.xml


// File: HelloWorld_8cxx-example.xml


// File: HelloWorld_8py-example.xml


// File: iU22tomultisc_8cxx-example.xml


// File: LargeVRDSExplicit_8cxx-example.xml


// File: MagnifyFile_8cxx-example.xml


// File: ManipulateFile_8cs-example.xml


// File: ManipulateFile_8py-example.xml


// File: ManipulateSequence_8py-example.xml


// File: MergeFile_8py-example.xml


// File: MergeTwoFiles_8cxx-example.xml


// File: MetaImageMD5Activiz_8cs-example.xml


// File: MIPViewer_8java-example.xml


// File: MpegVideoInfo_8cs-example.xml


// File: MPRViewer_8java-example.xml


// File: MPRViewer2_8java-example.xml


// File: MrProtocol_8cxx-example.xml


// File: NewSequence_8cs-example.xml


// File: NewSequence_8py-example.xml


// File: offscreenimage_8cxx-example.xml


// File: PatchFile_8cxx-example.xml


// File: PhilipsPrivateRescaleInterceptSlope_8py-example.xml


// File: PlaySound_8py-example.xml


// File: pmsct_rgb1_8cxx-example.xml


// File: PrivateDict_8py-example.xml


// File: PublicDict_8cxx-example.xml


// File: QIDO-RS_8cxx-example.xml


// File: ReadAndDumpDICOMDIR_8cxx-example.xml


// File: ReadAndDumpDICOMDIR_8py-example.xml


// File: ReadAndPrintAttributes_8cxx-example.xml


// File: ReadExplicitLengthSQIVR_8cxx-example.xml


// File: ReadFiles_8java-example.xml


// File: ReadGEMSSDO_8cxx-example.xml


// File: ReadMultiTimesException_8cxx-example.xml


// File: ReadSeriesIntoVTK_8java-example.xml


// File: ReadUTF8QtDir_8cxx-example.xml


// File: RefCounting_8cs-example.xml


// File: ReformatFile_8cs-example.xml


// File: RemovePrivateTags_8py-example.xml


// File: RescaleImage_8cs-example.xml


// File: reslicesphere_8cxx-example.xml


// File: ReWriteSCAsMR_8py-example.xml


// File: rle2img_8cxx-example.xml


// File: rtstructapp_8cxx-example.xml


// File: ScanDirectory_8cs-example.xml


// File: ScanDirectory_8java-example.xml


// File: ScanDirectory_8py-example.xml


// File: SendFileSCU_8cs-example.xml


// File: SimplePrint_8cs-example.xml


// File: SimplePrintPatientName_8cs-example.xml


// File: SimpleScanner_8cxx-example.xml


// File: SortImage_8cxx-example.xml


// File: SortImage_8py-example.xml


// File: SortImage2_8cs-example.xml


// File: StandardizeFiles_8cs-example.xml


// File: StreamImageReaderTest_8cxx-example.xml


// File: TestByteSwap_8cxx-example.xml


// File: TestReader_8cxx-example.xml


// File: TestReader_8py-example.xml


// File: threadgdcm_8cxx-example.xml


// File: TraverseModules_8cxx-example.xml


// File: uid_unique_8cxx-example.xml


// File: VolumeSorter_8cxx-example.xml


// File: WriteBuffer_8py-example.xml


// File: indexpage.xml

