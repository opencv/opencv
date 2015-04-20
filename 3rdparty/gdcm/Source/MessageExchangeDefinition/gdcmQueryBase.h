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
#ifndef GDCMQUERYBASE_H
#define GDCMQUERYBASE_H

#include "gdcmTag.h"
#include "gdcmDataElement.h"

#include <vector>

namespace gdcm
{
  enum ERootType
    {
    ePatientRootType,
    eStudyRootType
    };

/**
 * \brief QueryBase
 * contains: the base class for constructing a query dataset for a C-FIND and a
 * C-MOVE
 *
 * There are four levels of C-FIND and C-MOVE query:
 * \li Patient
 * \li Study
 * \li Series
 * \li Image
 *
 * Each one has its own required and optional tags. This class provides an
 * interface for getting those tags. This is an interface class.
 *
 * See 3.4 C 6.1 and 3.4 C 6.2 for the patient and study root query types.
 * These sections define the tags allowed by a particular query. The caller
 * must pass in which root type they want, patient or study. A third root type,
 * Modality Worklist Query, isn't yet supported.
 *
 * This class (or rather it's derived classes) will be held in the RootQuery
 * types. These query types actually make the dataset, and will use this
 * dataset to list the required, unique, and optional tags for each type of
 * query. This design is somewhat overly complicated, but is kept so that if we
 * ever wanted to try to guess the query type from the given tags, we could do
 * so.
 */
class GDCM_EXPORT QueryBase
{
  public:
    virtual ~QueryBase() {}

    virtual std::vector<Tag> GetRequiredTags(const ERootType& inRootType) const = 0;
    virtual std::vector<Tag> GetUniqueTags(const ERootType& inRootType) const = 0;
    virtual std::vector<Tag> GetOptionalTags(const ERootType& inRootType) const = 0;
    // C.4.1.2.1 Baseline Behavior of SCU
    // All C-FIND SCUs shall be capable of generating query requests which
    // meet the requirements of the Hierarchical Search.
    // The Identifier contained in a C-FIND request shall contain a single
    // value in the Unique Key Attribute for each level above the
    // Query/Retrieve level. No Required or Optional Keys shall be
    // specified which are associated with levels above the Query/Retrieve
    // level.
    /// Return all Unique Key for a particular Query Root type (from the same level and above).
    virtual std::vector<Tag> GetHierachicalSearchTags(const ERootType& inRootType) const = 0;

    /// In order to validate a query dataset, just check for the presence
    /// of a tag, not it's requirement level in the spec
    std::vector<Tag> GetAllTags(const ERootType& inRootType) const;

    /// In order to validate a query dataset we need to check that there
    /// exists at least one required (or unique) key
    std::vector<Tag> GetAllRequiredTags(const ERootType& inRootType) const;

    virtual const char * GetName() const = 0;
    virtual DataElement GetQueryLevel() const = 0;
  };
}

#endif //GDCMQUERYBASE_H
