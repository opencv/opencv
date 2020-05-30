// Protocol Buffers - Google's data interchange format
// Copyright 2008 Google Inc.  All rights reserved.
// https://developers.google.com/protocol-buffers/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Author: kenton@google.com (Kenton Varda)
//  Based on original Protocol Buffers design by
//  Sanjay Ghemawat, Jeff Dean, and others.

#include <google/protobuf/io/printer.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/stubs/logging.h>
#include <google/protobuf/stubs/common.h>

namespace google {
namespace protobuf {
namespace io {

Printer::Printer(ZeroCopyOutputStream* output, char variable_delimiter)
    : variable_delimiter_(variable_delimiter),
      output_(output),
      buffer_(NULL),
      buffer_size_(0),
      offset_(0),
      at_start_of_line_(true),
      failed_(false),
      annotation_collector_(NULL) {}

Printer::Printer(ZeroCopyOutputStream* output, char variable_delimiter,
                 AnnotationCollector* annotation_collector)
    : variable_delimiter_(variable_delimiter),
      output_(output),
      buffer_(NULL),
      buffer_size_(0),
      offset_(0),
      at_start_of_line_(true),
      failed_(false),
      annotation_collector_(annotation_collector) {}

Printer::~Printer() {
  // Only BackUp() if we have called Next() at least once and never failed.
  if (buffer_size_ > 0 && !failed_) {
    output_->BackUp(buffer_size_);
  }
}

bool Printer::GetSubstitutionRange(const char* varname,
                                   std::pair<size_t, size_t>* range) {
  std::map<string, std::pair<size_t, size_t> >::const_iterator iter =
      substitutions_.find(varname);
  if (iter == substitutions_.end()) {
    GOOGLE_LOG(DFATAL) << " Undefined variable in annotation: " << varname;
    return false;
  }
  if (iter->second.first > iter->second.second) {
    GOOGLE_LOG(DFATAL) << " Variable used for annotation used multiple times: "
                << varname;
    return false;
  }
  *range = iter->second;
  return true;
}

void Printer::Annotate(const char* begin_varname, const char* end_varname,
                       const string& file_path, const std::vector<int>& path) {
  if (annotation_collector_ == NULL) {
    // Can't generate signatures with this Printer.
    return;
  }
  std::pair<size_t, size_t> begin, end;
  if (!GetSubstitutionRange(begin_varname, &begin) ||
      !GetSubstitutionRange(end_varname, &end)) {
    return;
  }
  if (begin.first > end.second) {
    GOOGLE_LOG(DFATAL) << "  Annotation has negative length from " << begin_varname
                << " to " << end_varname;
  } else {
    annotation_collector_->AddAnnotation(begin.first, end.second, file_path,
                                         path);
  }
}

void Printer::Print(const std::map<string, string>& variables,
                    const char* text) {
  int size = strlen(text);
  int pos = 0;  // The number of bytes we've written so far.
  substitutions_.clear();
  line_start_variables_.clear();

  for (int i = 0; i < size; i++) {
    if (text[i] == '\n') {
      // Saw newline.  If there is more text, we may need to insert an indent
      // here.  So, write what we have so far, including the '\n'.
      WriteRaw(text + pos, i - pos + 1);
      pos = i + 1;

      // Setting this true will cause the next WriteRaw() to insert an indent
      // first.
      at_start_of_line_ = true;
      line_start_variables_.clear();

    } else if (text[i] == variable_delimiter_) {
      // Saw the start of a variable name.

      // Write what we have so far.
      WriteRaw(text + pos, i - pos);
      pos = i + 1;

      // Find closing delimiter.
      const char* end = strchr(text + pos, variable_delimiter_);
      if (end == NULL) {
        GOOGLE_LOG(DFATAL) << " Unclosed variable name.";
        end = text + pos;
      }
      int endpos = end - text;

      string varname(text + pos, endpos - pos);
      if (varname.empty()) {
        // Two delimiters in a row reduce to a literal delimiter character.
        WriteRaw(&variable_delimiter_, 1);
      } else {
        // Replace with the variable's value.
        std::map<string, string>::const_iterator iter = variables.find(varname);
        if (iter == variables.end()) {
          GOOGLE_LOG(DFATAL) << " Undefined variable: " << varname;
        } else {
          if (at_start_of_line_ && iter->second.empty()) {
            line_start_variables_.push_back(varname);
          }
          WriteRaw(iter->second.data(), iter->second.size());
          std::pair<std::map<string, std::pair<size_t, size_t> >::iterator,
                    bool>
              inserted = substitutions_.insert(std::make_pair(
                  varname,
                  std::make_pair(offset_ - iter->second.size(), offset_)));
          if (!inserted.second) {
            // This variable was used multiple times.  Make its span have
            // negative length so we can detect it if it gets used in an
            // annotation.
            inserted.first->second = std::make_pair(1, 0);
          }
        }
      }

      // Advance past this variable.
      i = endpos;
      pos = endpos + 1;
    }
  }

  // Write the rest.
  WriteRaw(text + pos, size - pos);
}

void Printer::Print(const char* text) {
  static std::map<string, string> empty;
  Print(empty, text);
}

void Printer::Print(const char* text,
                    const char* variable, const string& value) {
  std::map<string, string> vars;
  vars[variable] = value;
  Print(vars, text);
}

void Printer::Print(const char* text,
                    const char* variable1, const string& value1,
                    const char* variable2, const string& value2) {
  std::map<string, string> vars;
  vars[variable1] = value1;
  vars[variable2] = value2;
  Print(vars, text);
}

void Printer::Print(const char* text,
                    const char* variable1, const string& value1,
                    const char* variable2, const string& value2,
                    const char* variable3, const string& value3) {
  std::map<string, string> vars;
  vars[variable1] = value1;
  vars[variable2] = value2;
  vars[variable3] = value3;
  Print(vars, text);
}

void Printer::Print(const char* text,
                    const char* variable1, const string& value1,
                    const char* variable2, const string& value2,
                    const char* variable3, const string& value3,
                    const char* variable4, const string& value4) {
  std::map<string, string> vars;
  vars[variable1] = value1;
  vars[variable2] = value2;
  vars[variable3] = value3;
  vars[variable4] = value4;
  Print(vars, text);
}

void Printer::Print(const char* text,
                    const char* variable1, const string& value1,
                    const char* variable2, const string& value2,
                    const char* variable3, const string& value3,
                    const char* variable4, const string& value4,
                    const char* variable5, const string& value5) {
  std::map<string, string> vars;
  vars[variable1] = value1;
  vars[variable2] = value2;
  vars[variable3] = value3;
  vars[variable4] = value4;
  vars[variable5] = value5;
  Print(vars, text);
}

void Printer::Print(const char* text,
                    const char* variable1, const string& value1,
                    const char* variable2, const string& value2,
                    const char* variable3, const string& value3,
                    const char* variable4, const string& value4,
                    const char* variable5, const string& value5,
                    const char* variable6, const string& value6) {
  std::map<string, string> vars;
  vars[variable1] = value1;
  vars[variable2] = value2;
  vars[variable3] = value3;
  vars[variable4] = value4;
  vars[variable5] = value5;
  vars[variable6] = value6;
  Print(vars, text);
}

void Printer::Print(const char* text,
                    const char* variable1, const string& value1,
                    const char* variable2, const string& value2,
                    const char* variable3, const string& value3,
                    const char* variable4, const string& value4,
                    const char* variable5, const string& value5,
                    const char* variable6, const string& value6,
                    const char* variable7, const string& value7) {
  std::map<string, string> vars;
  vars[variable1] = value1;
  vars[variable2] = value2;
  vars[variable3] = value3;
  vars[variable4] = value4;
  vars[variable5] = value5;
  vars[variable6] = value6;
  vars[variable7] = value7;
  Print(vars, text);
}

void Printer::Print(const char* text,
                    const char* variable1, const string& value1,
                    const char* variable2, const string& value2,
                    const char* variable3, const string& value3,
                    const char* variable4, const string& value4,
                    const char* variable5, const string& value5,
                    const char* variable6, const string& value6,
                    const char* variable7, const string& value7,
                    const char* variable8, const string& value8) {
  std::map<string, string> vars;
  vars[variable1] = value1;
  vars[variable2] = value2;
  vars[variable3] = value3;
  vars[variable4] = value4;
  vars[variable5] = value5;
  vars[variable6] = value6;
  vars[variable7] = value7;
  vars[variable8] = value8;
  Print(vars, text);
}

void Printer::Indent() {
  indent_ += "  ";
}

void Printer::Outdent() {
  if (indent_.empty()) {
    GOOGLE_LOG(DFATAL) << " Outdent() without matching Indent().";
    return;
  }

  indent_.resize(indent_.size() - 2);
}

void Printer::PrintRaw(const string& data) {
  WriteRaw(data.data(), data.size());
}

void Printer::PrintRaw(const char* data) {
  if (failed_) return;
  WriteRaw(data, strlen(data));
}

void Printer::WriteRaw(const char* data, int size) {
  if (failed_) return;
  if (size == 0) return;

  if (at_start_of_line_ && (size > 0) && (data[0] != '\n')) {
    // Insert an indent.
    at_start_of_line_ = false;
    CopyToBuffer(indent_.data(), indent_.size());
    if (failed_) return;
    // Fix up empty variables (e.g., "{") that should be annotated as
    // coming after the indent.
    for (std::vector<string>::iterator i = line_start_variables_.begin();
         i != line_start_variables_.end(); ++i) {
      substitutions_[*i].first += indent_.size();
      substitutions_[*i].second += indent_.size();
    }
  }

  // If we're going to write any data, clear line_start_variables_, since
  // we've either updated them in the block above or they no longer refer to
  // the current line.
  line_start_variables_.clear();

  CopyToBuffer(data, size);
}

void Printer::CopyToBuffer(const char* data, int size) {
  if (failed_) return;
  if (size == 0) return;

  while (size > buffer_size_) {
    // Data exceeds space in the buffer.  Copy what we can and request a
    // new buffer.
    memcpy(buffer_, data, buffer_size_);
    offset_ += buffer_size_;
    data += buffer_size_;
    size -= buffer_size_;
    void* void_buffer;
    failed_ = !output_->Next(&void_buffer, &buffer_size_);
    if (failed_) return;
    buffer_ = reinterpret_cast<char*>(void_buffer);
  }

  // Buffer is big enough to receive the data; copy it.
  memcpy(buffer_, data, size);
  buffer_ += size;
  buffer_size_ -= size;
  offset_ += size;
}

}  // namespace io
}  // namespace protobuf
}  // namespace google
