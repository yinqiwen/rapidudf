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
namespace rapidudf {
namespace functions {
static constexpr std::string_view kBuiltinStringViewCmp = "rapidudf_compare_string_view";
static constexpr std::string_view kBuiltinCastStdStrToStringView = "rapidudf_cast_stdstr_to_string_view";
static constexpr std::string_view kBuiltinCastFbsStrToStringView = "rapidudf_cast_fbsstr_to_string_view";
static constexpr std::string_view kBuiltinCastStdStrViewToStringView = "rapidudf_cast_stdstrview_to_string_view";

static constexpr std::string_view kBuiltinJsonMemberGet = "rapidudf_json_member_get";
static constexpr std::string_view kBuiltinJsonArrayGet = "rapidudf_json_array_get";
static constexpr std::string_view kBuiltinJsonExtract = "rapidudf_json_extract";
// static constexpr std::string_view kBuiltinJsonCmpString = "rapidudf_json_cmp_string";
// static constexpr std::string_view kBuiltinJsonCmpInt = "rapidudf_json_cmp_int";
// static constexpr std::string_view kBuiltinJsonCmpFloat = "rapidudf_json_cmp_float";
// static constexpr std::string_view kBuiltinJsonCmpBool = "rapidudf_json_cmp_bool";
// static constexpr std::string_view kBuiltinJsonCmpJson = "rapidudf_json_cmp_json";

static constexpr std::string_view kBuiltinNewSimdVector = "rapidudf_new_simd_vector";
static constexpr std::string_view kBuiltinThrowVectorExprEx = "rapidudf_throw_vector_expr_error";

}  // namespace functions
}  // namespace rapidudf