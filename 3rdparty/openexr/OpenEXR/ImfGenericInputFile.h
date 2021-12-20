//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef IMFGENERICINPUTFILE_H_
#define IMFGENERICINPUTFILE_H_

#include "ImfForward.h"


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class IMF_EXPORT_TYPE GenericInputFile
{
    public:
        IMF_EXPORT
        virtual ~GenericInputFile();

    protected:
        IMF_EXPORT
        GenericInputFile();
        IMF_EXPORT
        void readMagicNumberAndVersionField(OPENEXR_IMF_INTERNAL_NAMESPACE::IStream& is, int& version);
};

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif /* IMFGENERICINPUTFILE_H_ */
