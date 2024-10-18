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
#include "rapidudf/log/log.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/types/simd/vector.h"
#include "x86simdsort.h"

namespace rapidudf {
namespace functions {
template <typename T>
void simd_vector_sort(Context& ctx, simd::Vector<T> data, bool descending) {
  if (data.IsReadonly()) {
    THROW_READONLY_ERR("can NOT sort on readonly vector");
  }
  x86simdsort::qsort(const_cast<T*>(data.Data()), data.Size(), ctx.HasNan(), descending);
}
template <typename T>
void simd_vector_select(Context& ctx, simd::Vector<T> data, size_t k, bool descending) {
  if (data.IsReadonly()) {
    THROW_READONLY_ERR("can NOT select on readonly vector");
  }
  x86simdsort::qselect(const_cast<T*>(data.Data()), k, data.Size(), ctx.HasNan(), descending);
}
template <typename T>
void simd_vector_topk(Context& ctx, simd::Vector<T> data, size_t k, bool descending) {
  if (data.IsReadonly()) {
    THROW_READONLY_ERR("can NOT topk on readonly vector");
  }
  x86simdsort::partial_qsort(const_cast<T*>(data.Data()), k, data.Size(), ctx.HasNan(), descending);
}
template <typename T>
simd::Vector<size_t> simd_vector_argsort(Context& ctx, simd::Vector<T> data, bool descending) {
  std::vector<size_t> idxs = x86simdsort::argsort(const_cast<T*>(data.Data()), data.Size(), ctx.HasNan(), descending);
  auto p = std::make_unique<std::vector<size_t>>(std::move(idxs));
  simd::Vector<size_t> ret(*p);
  ctx.Own(std::move(p));
  return ret;
}
template <typename T>
simd::Vector<size_t> simd_vector_argselect(Context& ctx, simd::Vector<T> data, size_t k, bool descending) {
  if (descending) {
    return simd_vector_argsort(ctx, data, descending);
  } else {
    auto idxs = x86simdsort::argselect(const_cast<T*>(data.Data()), k, data.Size(), ctx.HasNan());
    auto p = std::make_unique<std::vector<size_t>>(std::move(idxs));
    simd::Vector<size_t> ret(*p);
    ctx.Own(std::move(p));
    return ret;
  }
}

template <typename K, typename V>
void simd_vector_sort_key_value(Context& ctx, simd::Vector<K> key, simd::Vector<V> value, bool descending) {
  if (key.IsReadonly() || value.IsReadonly()) {
    THROW_READONLY_ERR(
        fmt::format("can NOT sort_key_value on readonly vector, key vector readobt:{}, value vector readonly:{}",
                    key.IsReadonly(), value.IsReadonly()));
  }

  x86simdsort::keyvalue_qsort(const_cast<K*>(key.Data()), const_cast<V*>(value.Data()), key.Size(), ctx.HasNan(),
                              descending);
}
template <typename K, typename V>
void simd_vector_topk_key_value(Context& ctx, simd::Vector<K> key, simd::Vector<V> value, size_t k, bool descending) {
  if (key.IsReadonly() || value.IsReadonly()) {
    THROW_READONLY_ERR("can NOT topk_key_value on readonly vector");
  }
  x86simdsort::keyvalue_partial_sort(const_cast<K*>(key.Data()), const_cast<V*>(value.Data()), k, key.Size(),
                                     ctx.HasNan(), descending);
}
template <typename K, typename V>
void simd_vector_select_key_value(Context& ctx, simd::Vector<K> key, simd::Vector<V> value, size_t k, bool descending) {
  if (key.IsReadonly() || value.IsReadonly()) {
    THROW_READONLY_ERR("can NOT select_key_value on readonly vector");
  }
  x86simdsort::keyvalue_select(const_cast<K*>(key.Data()), const_cast<V*>(value.Data()), k, key.Size(), ctx.HasNan(),
                               descending);
}

#define DEFINE_SORT_OP_TEMPLATE(r, func, ii, TYPE) \
  template void func<TYPE>(Context & ctx, simd::Vector<TYPE> data, bool descending);
#define DEFINE_SORT_OP(func, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SORT_OP_TEMPLATE, func, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SORT_OP(simd_vector_sort, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t)

#define DEFINE_SELECT_OP_TEMPLATE(r, func, ii, TYPE) \
  template void func<TYPE>(Context & ctx, simd::Vector<TYPE> data, size_t k, bool descending);
#define DEFINE_SELECT_OP(func, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SELECT_OP_TEMPLATE, func, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SELECT_OP(simd_vector_select, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t)
DEFINE_SELECT_OP(simd_vector_topk, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t)

#define DEFINE_ARGSORT_OP_TEMPLATE(r, func, ii, TYPE) \
  template simd::Vector<size_t> func<TYPE>(Context & ctx, simd::Vector<TYPE> data, bool descending);
#define DEFINE_ARGSORT_OP(func, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_ARGSORT_OP_TEMPLATE, func, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_ARGSORT_OP(simd_vector_argsort, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t)

#define DEFINE_ARGSELECT_OP_TEMPLATE(r, func, ii, TYPE) \
  template simd::Vector<size_t> func<TYPE>(Context & ctx, simd::Vector<TYPE> data, size_t, bool descending);
#define DEFINE_ARGSELECT_OP(func, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_ARGSELECT_OP_TEMPLATE, func, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_ARGSELECT_OP(simd_vector_argselect, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t)

#define KEY_VALUE_SORT_DTYPES (uint32_t)(int32_t)(uint64_t)(int64_t)(float)(double)
#define DEFINE_KEY_VALUE_SORT_FUNC(r, kv)                                                                      \
  template void simd_vector_sort_key_value<BOOST_PP_SEQ_ELEM(0, kv), BOOST_PP_SEQ_ELEM(1, kv)>(                \
      Context & ctx, simd::Vector<BOOST_PP_SEQ_ELEM(0, kv)> key, simd::Vector<BOOST_PP_SEQ_ELEM(1, kv)> value, \
      bool descending);                                                                                        \
  template void simd_vector_topk_key_value<BOOST_PP_SEQ_ELEM(0, kv), BOOST_PP_SEQ_ELEM(1, kv)>(                \
      Context & ctx, simd::Vector<BOOST_PP_SEQ_ELEM(0, kv)> key, simd::Vector<BOOST_PP_SEQ_ELEM(1, kv)> value, \
      size_t k, bool descending);                                                                              \
  template void simd_vector_select_key_value<BOOST_PP_SEQ_ELEM(0, kv), BOOST_PP_SEQ_ELEM(1, kv)>(              \
      Context & ctx, simd::Vector<BOOST_PP_SEQ_ELEM(0, kv)> key, simd::Vector<BOOST_PP_SEQ_ELEM(1, kv)> value, \
      size_t k, bool descending);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(DEFINE_KEY_VALUE_SORT_FUNC, (KEY_VALUE_SORT_DTYPES)(KEY_VALUE_SORT_DTYPES))
}  // namespace functions
}  // namespace rapidudf