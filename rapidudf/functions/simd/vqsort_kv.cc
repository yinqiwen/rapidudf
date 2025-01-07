/*
 * Copyright (c) 2024 qiyingwang <qiyingwang@tencent.com>. All rights reserved.
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
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

#include "rapidudf/context/context.h"
#include "rapidudf/functions/simd/vector_misc.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/string_view.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rapidudf/functions/simd/vqsort_kv.cc"  // this file

#include "hwy/foreach_target.h"  // must come before highway.h

#include "hwy/contrib/algo/find-inl.h"
#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/contrib/random/random-inl.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace rapidudf {
namespace functions {

namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// ------------------------------ HeapSort

template <class Traits, typename K, typename V>
void SiftDown(Traits st, K* HWY_RESTRICT lanes, V* HWY_RESTRICT vals, const size_t num_lanes, size_t start) {
  constexpr size_t N1 = st.LanesPerKey();
  const hn::FixedTag<K, N1> d;

  while (start < num_lanes) {
    const size_t left = 2 * start + N1;
    const size_t right = 2 * start + 2 * N1;
    if (left >= num_lanes) break;
    size_t idx_larger = start;
    const auto key_j = st.SetKey(d, lanes + start);
    if (AllTrue(d, st.Compare(d, key_j, st.SetKey(d, lanes + left)))) {
      idx_larger = left;
    }
    if (right < num_lanes &&
        hn::AllTrue(d, st.Compare(d, st.SetKey(d, lanes + idx_larger), st.SetKey(d, lanes + right)))) {
      idx_larger = right;
    }
    if (idx_larger == start) break;
    st.Swap(lanes + start, lanes + idx_larger);
    st.Swap(vals + start, vals + idx_larger);
    start = idx_larger;
  }
}

// Heapsort: O(1) space, O(N*logN) worst-case comparisons.
// Based on LLVM sanitizer_common.h, licensed under Apache-2.0.
template <class Traits, typename K, typename V>
void HeapSort(Traits st, K* HWY_RESTRICT lanes, V* HWY_RESTRICT vals, const size_t num_lanes) {
  constexpr size_t N1 = st.LanesPerKey();

  HWY_ASSERT(num_lanes >= 2 * N1);

  // Build heap.
  for (size_t i = ((num_lanes - N1) / N1 / 2) * N1; i != (~N1 + 1); i -= N1) {
    SiftDown(st, lanes, vals, num_lanes, i);
  }

  for (size_t i = num_lanes - N1; i != 0; i -= N1) {
    // Swap root with last
    st.Swap(lanes + 0, lanes + i);
    st.Swap(vals + 0, vals + i);

    // Sift down the new root.
    SiftDown(st, lanes, vals, i, 0);
  }
}

template <class Traits, typename K, typename V>
void HeapSelect(Traits st, K* HWY_RESTRICT lanes, V* HWY_RESTRICT vals, const size_t num_lanes, const size_t select) {
  constexpr size_t N1 = st.LanesPerKey();
  const size_t k = select + 1;

  HWY_ASSERT(k >= 2 * N1 && num_lanes >= 2 * N1);

  const hn::FixedTag<K, N1> d;

  // Build heap.
  for (size_t i = ((k - N1) / N1 / 2) * N1; i != (~N1 + 1); i -= N1) {
    SiftDown(st, lanes, vals, k, i);
  }

  for (size_t i = k; i <= num_lanes - N1; i += N1) {
    if (hn::AllTrue(d, st.Compare(d, st.SetKey(d, lanes + i), st.SetKey(d, lanes + 0)))) {
      // Swap root with last
      st.Swap(lanes + 0, lanes + i);
      st.Swap(vals + 0, vals + i);

      // Sift down the new root.
      SiftDown(st, lanes, vals, k, 0);
    }
  }

  st.Swap(lanes + 0, lanes + k - 1);
  st.Swap(vals + 0, vals + k - 1);
}

template <class Traits, typename K, typename V>
void HeapPartialSort(Traits st, K* HWY_RESTRICT lanes, V* HWY_RESTRICT vals, const size_t num_lanes,
                     const size_t select) {
  HeapSelect(st, lanes, vals, num_lanes, select);
  HeapSort(st, lanes, vals, select);
}

}  // namespace HWY_NAMESPACE
}  // namespace functions
}  // namespace rapidudf

HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rapidudf {
namespace functions {}
}  // namespace rapidudf
#endif  // HWY_ONCE