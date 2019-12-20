/* <copyright>
  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright (c) 2005-2014 Intel Corporation. All rights reserved.

  This program is free software; you can redistribute it and/or modify
  it under the terms of version 2 of the GNU General Public License as
  published by the Free Software Foundation.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin St - Fifth Floor, Boston, MA 02110-1301 USA.
  The full GNU General Public License is included in this distribution
  in the file called LICENSE.GPL.

  Contact Information:
  http://software.intel.com/en-us/articles/intel-vtune-amplifier-xe/

  BSD LICENSE

  Copyright (c) 2005-2014 Intel Corporation. All rights reserved.
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
</copyright> */

/*
 * This file implements an interface bridge from Low-Level Virtual Machine
 * llvm::JITEventListener to Intel JIT Profiling API.  It passes the function
 * and line information to the appropriate functions in the JIT profiling
 * interface so that any LLVM-based JIT engine can emit the JIT code
 * notifications that the profiler will receive.
 *
 * Usage model:
 *
 * 1. Register the listener implementation instance with the execution engine:
 *
 *    #include <llvm_jit_event_listener.hpp>
 *    ...
 *    ExecutionEngine *TheExecutionEngine;
 *    ...
 *    TheExecutionEngine = EngineBuilder(TheModule).create();
 *    ...
 *    __itt_llvm_jit_event_listener jitListener;
 *    TheExecutionEngine->RegisterJITEventListener(&jitListener);
 *    ...
 *
 * 2. When compiling make sure to add the ITT API include directory to the
 *    compiler include directories, ITT API library directory to the linker
 *    library directories and link with jitprofling static library.
 */

#ifndef __ITT_LLVM_JIT_EVENT_LISTENER_HPP__
#define __ITT_LLVM_JIT_EVENT_LISTENER_HPP__

#include "jitprofiling.h"

#include <llvm/Function.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/DebugInfo.h>

#include <map>
#include <cassert>

// Uncomment the line below to turn on logging to stderr
#define JITPROFILING_DEBUG_ENABLE

// Some elementary logging support
#ifdef JITPROFILING_DEBUG_ENABLE
#include <cstdio>
#include <cstdarg>
static void _jit_debug(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}
// Use the macro as JITDEBUG(("foo: %d", foo_val));
#define JITDEBUG(x) \
    do { \
        _jit_debug("jit-listener: "); \
        _jit_debug x; \
    } \
    while (0)
#else
#define JITDEBUG(x)
#endif

// LLVM JIT event listener, translates the notifications to the JIT profiling
// API information.
class __itt_llvm_jit_event_listener : public llvm::JITEventListener
{
public:
    __itt_llvm_jit_event_listener() {}

public:
    virtual void NotifyFunctionEmitted(const llvm::Function &F,
        void *Code, size_t Size, const EmittedFunctionDetails &Details)
    {
        std::string name = F.getName().str();
        JITDEBUG(("function jitted:\n"));
        JITDEBUG(("  addr=0x%08x\n", (int)Code));
        JITDEBUG(("  name=`%s'\n", name.c_str()));
        JITDEBUG(("  code-size=%d\n", (int)Size));
        JITDEBUG(("  line-infos-count=%d\n", Details.LineStarts.size()));

        // The method must not be in the map - the entry must have been cleared
        // from the map in NotifyFreeingMachineCode in case of rejitting.
        assert(m_addr2MethodId.find(Code) == m_addr2MethodId.end());

        int mid = iJIT_GetNewMethodID();
        m_addr2MethodId[Code] = mid;

        iJIT_Method_Load mload;
        memset(&mload, 0, sizeof mload);
        mload.method_id = mid;

        // Populate the method size and name information
        // TODO: The JIT profiling API should have members as const char pointers.
        mload.method_name = (char*)name.c_str();
        mload.method_load_address = Code;
        mload.method_size = (unsigned int)Size;

        // Populate line information now.
        // From the JIT API documentation it is not quite clear whether the
        // line information can be given in ranges, so we'll populate it for
        // every byte of the function, hmm.
        std::string srcFilePath;
        std::vector<LineNumberInfo> lineInfos;
        char *addr = (char*)Code;
        char *lineAddr = addr;          // Exclusive end point at which current
                                        // line info changes.
        const llvm::DebugLoc* loc = 0;  // Current line info
        int lineIndex = -1;             // Current index into the line info table
        for (int i = 0; i < Size; ++i, ++addr) {
            while (addr >= lineAddr) {
                if (lineIndex >= 0 && lineIndex < Details.LineStarts.size()) {
                    loc = &Details.LineStarts[lineIndex].Loc;
                    std::string p = getSrcFilePath(F.getContext(), *loc);
                    assert(srcFilePath.empty() || p == srcFilePath);
                    srcFilePath = p;
                } else {
                    loc = NULL;
                }
                lineIndex++;
                if (lineIndex >= 0 && lineIndex < Details.LineStarts.size()) {
                    lineAddr = (char*)Details.LineStarts[lineIndex].Address;
                } else {
                    lineAddr = addr + Size;
                }
            }
            if (loc) {
                int line = loc->getLine();
                LineNumberInfo info = { i, line };
                lineInfos.push_back(info);
                JITDEBUG(("  addr 0x%08x -> line %d\n", addr, line));
            }
        }
        if (!lineInfos.empty()) {
            mload.line_number_size = lineInfos.size();
            JITDEBUG(("  translated to %d line infos to JIT", (int)lineInfos.size()));
            mload.line_number_table = &lineInfos[0];
            mload.source_file_name = (char*)srcFilePath.c_str();
        }

        iJIT_NotifyEvent(iJVM_EVENT_TYPE_METHOD_LOAD_FINISHED, &mload);
    }

    virtual void NotifyFreeingMachineCode(void *OldPtr)
    {
        JITDEBUG(("function unjitted\n"));
        JITDEBUG(("  addr=0x%08x\n", (int)OldPtr));
        Addr2MethodId::iterator it = m_addr2MethodId.find(OldPtr);
        assert(it != m_addr2MethodId.end());
        iJIT_Method_Id mid = { it->second };
        iJIT_NotifyEvent(iJVM_EVENT_TYPE_METHOD_UNLOAD_START, &mid);
        m_addr2MethodId.erase(it);
    }

private:
    std::string getSrcFilePath(const llvm::LLVMContext& ctx, const llvm::DebugLoc& loc)
    {
        llvm::MDNode* node = loc.getAsMDNode(ctx);
        llvm::DILocation srcLoc(node);
        return srcLoc.getDirectory().str() + "/" + srcLoc.getFilename().str();
    }

private:
    /// Don't copy
    __itt_llvm_jit_event_listener(const __itt_llvm_jit_event_listener&);
    __itt_llvm_jit_event_listener& operator=(const __itt_llvm_jit_event_listener&);

private:
    typedef std::vector<LineNumberInfo> LineInfoList;

    // The method unload notification in VTune JIT profiling API takes the
    // method ID, not method address so have to maintain the mapping.  Is
    // there a more efficient and simple way to do this like attaching the
    // method ID information somehow to the LLVM function instance?
    //
    // TODO: It would be more convenient for the JIT API to take the method
    // address, not method ID.
    typedef std::map<const void*, int> Addr2MethodId;
    Addr2MethodId m_addr2MethodId;
};

#endif // Header guard
