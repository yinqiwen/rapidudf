/*
 * Copyright (c) 2024 yinqiwen yinqiwen@gmail.com. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

static constexpr std::string_view kBuiltinNewSimdVector = "rapidudf_new_simd_vector";
static constexpr std::string_view kBuiltinThrowVectorExprEx = "rapidudf_throw_vector_expr_error";

static constexpr std::string_view kTableGetColumnFunc = "get_column";

}  // namespace functions
}  // namespace rapidudf