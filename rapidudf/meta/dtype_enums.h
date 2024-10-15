/*
** BSD 3-Clause License
**
** Copyright (c) 2024, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
** All rights reserved.
**
** Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are met:
** * Redistributions of source code must retain the above copyright notice, this
** list of conditions and the following disclaimer.
**
** * Redistributions in binary form must reproduce the above copyright notice,
** this list of conditions and the following disclaimer in the documentation
** and/or other materials provided with the distribution.
**
** * Neither the name of the copyright holder nor the names of its
** contributors may be used to endorse or promote products derived from
** this software without specific prior written permission.
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
** AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
** IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
** DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
** FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
** DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
** SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
** CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
** OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace rapidudf {
enum CollectionType {
  COLLECTION_INVALID = 0,
  COLLECTION_VECTOR,
  COLLECTION_MAP,
  COLLECTION_SET,
  COLLECTION_UNORDERED_MAP,
  COLLECTION_UNORDERED_SET,
  COLLECTION_ABSL_SPAN,
  COLLECTION_TUPLE,
  COLLECTION_SIMD_VECTOR,
  COLLECTION_END,
};

constexpr std::array<std::string_view, COLLECTION_END> kCollectionTypeStrs = {
    "", "vector", "map", "set", "unordered_map", "unordered_set", "absl_span", "tuple", "simd_vector"};
enum FundamentalType {
  DATA_INVALID = 0,
  DATA_VOID,
  DATA_POINTER,
  DATA_BIT,
  DATA_U8,
  DATA_I8,
  DATA_U16,
  DATA_I16,
  DATA_U32,
  DATA_I32,
  DATA_U64,
  DATA_I64,
  DATA_F16,
  DATA_F32,
  DATA_F64,
  DATA_F80,
  DATA_STD_STRING_VIEW,
  DATA_STRING_VIEW,
  DATA_STRING,
  DATA_FLATBUFFERS_STRING,
  DATA_JSON,
  DATA_CONTEXT,
  DATA_DYN_OBJECT,
  DATA_TABLE,
  DATA_BUILTIN_TYPE_END,
  // DATA_SIMD_COLUMN,

  DATA_OBJECT_BEGIN = 64,
  DATA_COMPLEX_OBJECT = (1 << 14) - 1,
};

using u8 = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using i8 = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;
using f32 = float;
using f64 = double;

constexpr std::array<std::string_view, DATA_BUILTIN_TYPE_END> kFundamentalTypeStrs = {"invalid",
                                                                                      "void",
                                                                                      "pointer",
                                                                                      "bit",
                                                                                      "u8",
                                                                                      "i8",
                                                                                      "u16",
                                                                                      "i16",
                                                                                      "u32",
                                                                                      "i32",
                                                                                      "u64",
                                                                                      "i64",
                                                                                      "f16",
                                                                                      "f32",
                                                                                      "f64",
                                                                                      "f80",
                                                                                      "std_string_view",
                                                                                      "string_view",
                                                                                      "string",
                                                                                      "fbs_string",
                                                                                      "json",
                                                                                      "Context",
                                                                                      "dyn_obj",
                                                                                      "table"};
}  // namespace rapidudf