/***********************************************************************
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef OPENCV_FLANN_OBJECT_FACTORY_H_
#define OPENCV_FLANN_OBJECT_FACTORY_H_

//! @cond IGNORED

#include <map>

namespace cvflann
{

class CreatorNotFound
{
};

template<typename BaseClass,
         typename UniqueIdType,
         typename ObjectCreator = BaseClass* (*)()>
class ObjectFactory
{
    typedef ObjectFactory<BaseClass,UniqueIdType,ObjectCreator> ThisClass;
    typedef std::map<UniqueIdType, ObjectCreator> ObjectRegistry;

    // singleton class, private constructor
    ObjectFactory() {}

public:

    bool subscribe(UniqueIdType id, ObjectCreator creator)
    {
        if (object_registry.find(id) != object_registry.end()) return false;

        object_registry[id] = creator;
        return true;
    }

    bool unregister(UniqueIdType id)
    {
        return object_registry.erase(id) == 1;
    }

    ObjectCreator create(UniqueIdType id)
    {
        typename ObjectRegistry::const_iterator iter = object_registry.find(id);

        if (iter == object_registry.end()) {
            throw CreatorNotFound();
        }

        return iter->second;
    }

    static ThisClass& instance()
    {
        static ThisClass the_factory;
        return the_factory;
    }
private:
    ObjectRegistry object_registry;
};

}

//! @endcond

#endif /* OPENCV_FLANN_OBJECT_FACTORY_H_ */
