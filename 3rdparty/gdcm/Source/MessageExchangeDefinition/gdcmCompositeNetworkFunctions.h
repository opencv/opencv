/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef GDCMCOMPOSITENETWORKFUNCTIONS_H
#define GDCMCOMPOSITENETWORKFUNCTIONS_H

#include "gdcmDirectory.h"
#include "gdcmBaseRootQuery.h" // EQueryLevel / EQueryType

#include <vector>
#include <string>

namespace gdcm
{
/**
 * \brief Composite Network Functions
 * These functions provide a generic API to the DICOM functions implemented in
 * GDCM.
 * Advanced users can use this code as a template for building their own
 * versions of these functions (for instance, to provide progress bars or some
 * other way of handling returned query information), but for most users, these
 * functions should be sufficient to interface with a PACS to a local machine.
 * Note that these functions are not contained within a static class or some
 * other class-style interface, because multiple connections can be
 * instantiated in the same program.  The DICOM standard is much more function
 * oriented rather than class oriented in this instance, so the design of this
 * API reflects that functional approach.
 * These functions implements the following SCU operations:
 * \li C-ECHO SCU
 * \li C-FIND SCU
 * \li C-STORE SCU
 * \li C-MOVE SCU (+internal C-STORE SCP)
 */
class GDCM_EXPORT CompositeNetworkFunctions 
{
public:
  /// The most basic network function.  Use this function to ensure that the
  /// remote server is responding on the given IP and port number as expected.
  /// \param aetitle when not set will default to 'GDCMSCU'
  /// \param call when not set will default to 'ANY-SCP'
  /// \warning This is an error to set remote to NULL or portno to 0
  /// \return true if it worked.
  static bool CEcho( const char *remote, uint16_t portno, const char *aetitle = NULL,
    const char *call = NULL );

  typedef std::pair<Tag, std::string> KeyValuePairType;
  typedef std::vector< KeyValuePairType  > KeyValuePairArrayType;

  /// This function will take a list of strings and tags and fill in a query that
  /// can be used for either CFind or CMove (depending on the input boolean
  /// \param inMove).
  /// Note that the caller is responsible for deleting the constructed query.
  /// This function is used to build both a move and a find query 
  /// (true for inMove if it's move, false if it's find)
  static BaseRootQuery* ConstructQuery(ERootType inRootType, EQueryLevel inQueryLevel,
    const DataSet& queryds, bool inMove = false );

  /// \deprecated
  static BaseRootQuery* ConstructQuery(ERootType inRootType, EQueryLevel inQueryLevel,
    const KeyValuePairArrayType& keys, bool inMove = false );

  /// This function will use the provided query to get files from a remote server.
  /// NOTE that this functionality is essentially equivalent to C-GET in the
  /// DICOM standard; however, C-GET has been deprecated, so this function
  /// allows for the user to ask a remote server for files matching a query and
  /// return them to the local machine.
  /// Files will be written to the given output directory.
  /// If the operation succeeds, the function returns true.
  /// This function is a prime candidate for being overwritten by expert users;
  /// if the datasets should remain in memory, for instance, that behavior can
  /// be changed by creating a user-level version of this function.
  /// \param aetitle when not set will default to 'GDCMSCU'
  /// \param call when not set will default to 'ANY-SCP'
  /// This is an error to set remote to NULL or portno to 0
  /// when \param outputdir is not set default to current dir ('.')
  /// \return true if it worked.
  static bool CMove( const char *remote, uint16_t portno, const BaseRootQuery* query,
    uint16_t portscp, const char *aetitle = NULL,
    const char *call = NULL, const char *outputdir = NULL);

  /// This function will use the provided query to determine what files a remote
  /// server contains that match the query strings.  The return is a vector of
  /// datasets that contain tags as reported by the server.  If the dataset is
  /// empty, then it is possible that an error condition was encountered; in
  /// which case, the user should monitor the error and warning streams.
  /// \param aetitle when not set will default to 'GDCMSCU'
  /// \param call when not set will default to 'ANY-SCP'
  /// \warning This is an error to set remote to NULL or portno to 0
  /// \return true if it worked.
  static bool CFind( const char *remote, uint16_t portno, 
    const BaseRootQuery* query,
    std::vector<DataSet> &retDataSets,
    const char *aetitle = NULL,
    const char *call = NULL );

  /// This function will place the provided files into the remote server.
  /// The function returns true if it worked for all files.
  /// \warning the server side can refuse an association on a given file
  /// \param aetitle when not set will default to 'GDCMSCU'
  /// \param call when not set will default to 'ANY-SCP'
  /// \warning This is an error to set remote to NULL or portno to 0
  /// \return true if it worked for all files
  static bool CStore( const char *remote, uint16_t portno,
    const Directory::FilenamesType & filenames,
    const char *aetitle = NULL, const char *call = NULL);
};

} // end namespace gdcm

#endif // GDCMCOMPOSITENETWORKFUNCTIONS_H
