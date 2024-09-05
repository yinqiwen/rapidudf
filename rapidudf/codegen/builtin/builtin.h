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

#include <string_view>
#include "flatbuffers/flatbuffers.h"
#include "rapidudf/types/json_object.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {

bool compare_string_view(uint32_t op, StringView left, StringView right);
StringView cast_stdstr_to_string_view(const std::string& str);
StringView cast_fbsstr_to_string_view(const flatbuffers::String& str);
StringView cast_stdstrview_to_string_view(std::string_view str);

const JsonObject& json_member_get(const JsonObject& json, StringView key);
const JsonObject& json_array_get(const JsonObject& json, size_t idx);
bool json_cmp_string(uint32_t op, const JsonObject& json, StringView right, bool reverse);
bool json_cmp_int(uint32_t op, const JsonObject& json, int64_t right, bool reverse);
bool json_cmp_bool(uint32_t op, const JsonObject& json, bool right, bool reverse);
bool json_cmp_float(uint32_t op, const JsonObject& json, double right, bool reverse);
bool json_cmp_json(uint32_t op, const JsonObject& left, const JsonObject& right);

void init_builtin();
void init_builtin_stl_funcs();
void init_builtin_string_funcs();
void init_builtin_math_funcs();
void init_builtin_json_funcs();
void init_builtin_simd_vector_funcs();
bool is_builtin_math_func(std::string_view name);
bool register_builtin_math_func(std::string_view name);

}  // namespace rapidudf