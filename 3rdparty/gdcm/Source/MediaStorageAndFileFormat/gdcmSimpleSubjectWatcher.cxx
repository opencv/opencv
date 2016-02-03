/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSimpleSubjectWatcher.h"
#include "gdcmEvent.h"
#include "gdcmAnonymizeEvent.h"
#include "gdcmDataSetEvent.h"
#include "gdcmProgressEvent.h"
#include "gdcmFileNameEvent.h"

namespace gdcm
{

SimpleSubjectWatcher::SimpleSubjectWatcher(Subject *s, const char *comment):m_Subject(s),m_Comment(comment)
{
  // Create a series of commands
  m_StartFilterCommand =      SimpleCommandType::New();
  m_EndFilterCommand =        SimpleCommandType::New();
  m_ProgressFilterCommand =   CommandType::New();
  m_FileNameFilterCommand =   CommandType::New();
  m_IterationFilterCommand =  SimpleCommandType::New();
  m_AbortFilterCommand =      SimpleCommandType::New();

  m_AnonymizeFilterCommand =  CommandType::New();
  m_DataSetFilterCommand =  CommandType::New();
  m_DataFilterCommand =  CommandType::New();

  // Assign the callbacks
  m_StartFilterCommand->SetCallbackFunction(this,
                                        &SimpleSubjectWatcher::StartFilter);
  m_EndFilterCommand->SetCallbackFunction(this,
                                        &SimpleSubjectWatcher::EndFilter);
  m_ProgressFilterCommand->SetCallbackFunction(this,
                                        &SimpleSubjectWatcher::ShowProgress);
  m_FileNameFilterCommand->SetCallbackFunction(this,
                                        &SimpleSubjectWatcher::ShowFileName);
  m_IterationFilterCommand->SetCallbackFunction(this,
                                        &SimpleSubjectWatcher::ShowIteration);
  m_AbortFilterCommand->SetCallbackFunction(this,
                                        &SimpleSubjectWatcher::ShowAbort);
  m_AnonymizeFilterCommand->SetCallbackFunction(this,
                                        &SimpleSubjectWatcher::ShowAnonymization);
  m_DataSetFilterCommand->SetCallbackFunction(this,
                                        &SimpleSubjectWatcher::ShowDataSet);
  m_DataFilterCommand->SetCallbackFunction(this,
                                        &SimpleSubjectWatcher::ShowData);

  // Add the commands as observers
  m_StartTag = m_Subject->AddObserver(StartEvent(), m_StartFilterCommand);
  m_EndTag = m_Subject->AddObserver(EndEvent(), m_EndFilterCommand);
  m_ProgressTag
    = m_Subject->AddObserver(ProgressEvent(), m_ProgressFilterCommand);
  m_FileNameTag
    = m_Subject->AddObserver(FileNameEvent(), m_FileNameFilterCommand);
  m_IterationTag
    = m_Subject->AddObserver(IterationEvent(), m_IterationFilterCommand);
  m_AbortTag
    = m_Subject->AddObserver(AbortEvent(), m_AbortFilterCommand);
  m_AnonymizeTag
    = m_Subject->AddObserver(AnonymizeEvent(), m_AnonymizeFilterCommand);
  m_DataSetTag
    = m_Subject->AddObserver(DataSetEvent(), m_DataSetFilterCommand);
  m_DataTag
    = m_Subject->AddObserver(DataEvent(), m_DataFilterCommand);

  m_TestAbort = false;
}

SimpleSubjectWatcher::~SimpleSubjectWatcher()
{
  // Remove any observers we have on the old process object
  if (m_Subject)
    {
    if (m_StartFilterCommand)
      {
      m_Subject->RemoveObserver(m_StartTag);
      }
    if (m_EndFilterCommand)
      {
      m_Subject->RemoveObserver(m_EndTag);
      }
    if (m_ProgressFilterCommand)
      {
      m_Subject->RemoveObserver(m_ProgressTag);
      }
    if (m_FileNameFilterCommand)
      {
      m_Subject->RemoveObserver(m_FileNameTag);
      }
    if (m_IterationFilterCommand)
      {
      m_Subject->RemoveObserver(m_IterationTag);
      }
    if (m_AbortFilterCommand)
      {
      m_Subject->RemoveObserver(m_AbortTag);
      }
    if (m_AnonymizeFilterCommand)
      {
      m_Subject->RemoveObserver(m_AnonymizeTag);
      }
    if (m_DataFilterCommand)
      {
      m_Subject->RemoveObserver(m_DataTag);
      }
    if (m_DataSetFilterCommand)
      {
      m_Subject->RemoveObserver(m_DataSetTag);
      }
    }
}

void SimpleSubjectWatcher::StartFilter()
{
  std::cout << "Start" << std::endl;
}
void SimpleSubjectWatcher::EndFilter()
{
  std::cout << "End" << std::endl;
}
void SimpleSubjectWatcher::ShowProgress(Subject *caller, const Event &evt)
{
  const ProgressEvent &pe = dynamic_cast<const ProgressEvent&>(evt);
  (void)caller;
  if( !m_Comment.empty() )
    std::cout << "(" << m_Comment << ") ";
  std::cout << "Progress: " << pe.GetProgress() << std::endl;
}
void SimpleSubjectWatcher::ShowFileName(Subject *caller, const Event &evt)
{
  const FileNameEvent &pe = dynamic_cast<const FileNameEvent&>(evt);
  (void)caller;
  if( !m_Comment.empty() )
    std::cout << "(" << m_Comment << ") ";
  std::cout << "FileName: " << pe.GetFileName() << std::endl;
}
void SimpleSubjectWatcher::ShowIteration()
{
  std::cout << "Iteration" << std::endl;
}
void SimpleSubjectWatcher::ShowAbort()
{
  std::cout << "Abort" << std::endl;
}
void SimpleSubjectWatcher::ShowAnonymization(Subject *caller, const Event &evt)
{
  const AnonymizeEvent &ae = dynamic_cast<const AnonymizeEvent&>(evt);
  (void)caller;
  std::cout << "AnonymizeEvent: " << ae.GetTag() << std::endl;
}
void SimpleSubjectWatcher::ShowData(Subject *caller, const Event &evt)
{
  const DataEvent &ae = dynamic_cast<const DataEvent&>(evt);
  (void)caller;
  std::cout << "DataEvent: " << ae.GetDataLength() << std::endl;
}
void SimpleSubjectWatcher::ShowDataSet(Subject *caller, const Event &evt)
{
  const DataSetEvent &ae = dynamic_cast<const DataSetEvent&>(evt);
  (void)caller;
  std::cout << "DataSetEvent: \n";
  std::cout << ae.GetDataSet() << std::endl;
}

void SimpleSubjectWatcher::TestAbortOn()
{
  m_TestAbort = true;
}
void SimpleSubjectWatcher::TestAbortOff()
{
  m_TestAbort = false;
}


} // end namespace gdcm
