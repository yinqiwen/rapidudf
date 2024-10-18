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
#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>

#include "rapidudf/reflect/stl.h"
#include "rapidudf/types/string_view.h"
namespace rapidudf {
namespace functions {
#define STL_DTYPES (uint32_t)(int32_t)(uint64_t)(int64_t)(float)(double)(std::string_view)(std::string)(StringView)

#define RUDF_STL_REFLECT_HELPER_INIT(r, STL_HELPER, i, TYPE) STL_HELPER<TYPE>::Init();
#define RUDF_STL_REFLECT_HELPER(STL_HELPER, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(RUDF_STL_REFLECT_HELPER_INIT, STL_HELPER, STL_DTYPES)

void init_builtin_stl_sets_funcs() {
  RUDF_STL_REFLECT_HELPER(reflect::StdSetHelper)
  RUDF_STL_REFLECT_HELPER(reflect::StdUnorderedSetHelper)
}
}  // namespace functions
}  // namespace rapidudf
