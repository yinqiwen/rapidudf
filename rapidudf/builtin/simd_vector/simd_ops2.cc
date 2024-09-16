/*
** BSD 3-Clause License
**
** Copyright (c) 2023, qiyingwang <qiyingwang@tencent.com>, the respective contributors, as shown by the AUTHORS file.
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
#include "rapidudf/builtin/simd_vector/simd_ops.h"

#include <boost/preprocessor/library.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/variadic/to_seq.hpp>
#include <cstring>
#include <type_traits>
#include <vector>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rapidudf/builtin/simd_vector/simd_ops2.cc"  // this file

#include "hwy/foreach_target.h"  // must come before highway.h
#include "hwy/highway.h"

#include "hwy/bit_set.h"
#include "hwy/contrib/algo/transform-inl.h"
#include "hwy/contrib/dot/dot-inl.h"
#include "hwy/contrib/math/math-inl.h"

#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/exception.h"
#include "rapidudf/meta/function.h"
#include "rapidudf/meta/optype.h"
#include "rapidudf/types/bit.h"
#include "rapidudf/types/string_view.h"

#include <x86intrin.h>

extern "C" {
extern __m128 Sleef_powf4_u10(__m128, __m128);
extern __m256 Sleef_powf8_u10(__m256, __m256);
extern __m512 Sleef_powf16_u10(__m512, __m512);

extern __m128d Sleef_powd2_u10(__m128d, __m128d);
extern __m256d Sleef_powd4_u10(__m256d, __m256d);
extern __m512d Sleef_powd8_u10(__m512d, __m512d);

extern __m128 Sleef_coshf4_u10(__m128);
extern __m256 Sleef_coshf8_u10(__m256);
extern __m512 Sleef_coshf16_u10(__m512);

extern __m128d Sleef_coshd2_u10(__m128d);
extern __m256d Sleef_coshd4_u10(__m256d);
extern __m512d Sleef_coshd8_u10(__m512d);

extern __m128 Sleef_tanf4_u10(__m128);
extern __m256 Sleef_tanf8_u10(__m256);
extern __m512 Sleef_tanf16_u10(__m512);

extern __m128d Sleef_tand2_u10(__m128d);
extern __m256d Sleef_tand4_u10(__m256d);
extern __m512d Sleef_tand8_u10(__m512d);
}
#include "sleef.h"
HWY_BEFORE_NAMESPACE();

namespace rapidudf {
namespace simd {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

inline uint8_t bit_set(uint8_t number, uint8_t n) { return number | ((uint8_t)1 << n); }
inline uint8_t bit_clear(uint8_t number, uint8_t n) { return number & ~((uint8_t)1 << n); }

template <typename T>
static constexpr size_t get_lanes() {
  using number_t = typename InternalType<T>::internal_type;
  const hn::ScalableTag<number_t> d;
  return hn::Lanes(d);
}

template <class D, class Func, typename T = hn::TFromD<D>>
void do_unary_transform(D d, const T* in1, size_t count, T* out, const Func& func) {
  const size_t N = hn::Lanes(d);
  size_t idx = 0;
  if (count >= N) {
    for (; idx <= count - N; idx += N) {
      const hn::Vec<D> v1 = hn::LoadU(d, in1 + idx);
      hn::StoreU(func(d, v1), d, out + idx);
    }
  }

  // `count` was a multiple of the vector length `N`: already done.
  if (HWY_UNLIKELY(idx == count)) return;
  const size_t remaining = count - idx;
  HWY_DASSERT(0 != remaining && remaining < N);
  const hn::Vec<D> v1 = hn::LoadN(d, in1 + idx, remaining);
  hn::StoreN(func(d, v1), d, out + idx, remaining);
}

template <class D, class Func, typename T1, typename T2, typename OUT>
void do_binary_transform(D d, T1 in1, T2 in2, size_t count, OUT* out, const Func& func) {
  constexpr size_t N = Lanes(d);
  size_t idx = 0;
  if (count >= N) {
    for (; idx <= count - N; idx += N) {
      hn::Vec<D> v1;
      if constexpr (std::is_same_v<hn::Vec<D>, T1>) {
        v1 = in1;
      } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T1>) {
        v1 = hn::LoadU(d, in1 + idx);
      } else {
        static_assert(sizeof(T1) == -1, "invalid T1");
      }
      hn::Vec<D> v2;
      if constexpr (std::is_same_v<hn::Vec<D>, T2>) {
        v2 = in2;
      } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T2>) {
        v2 = hn::LoadU(d, in2 + idx);
      } else {
        static_assert(sizeof(T2) == -1, "invalid T2");
      }
      auto result = func(d, v1, v2);
      if constexpr (std::is_same_v<decltype(result), hn::VFromD<D>>) {
        if constexpr (std::is_same_v<hn::TFromD<D>, OUT>) {
          hn::StoreU(result, d, out + idx);
        } else {
          static_assert(sizeof(OUT) == -1, "invalid out type");
        }
      } else if constexpr (std::is_same_v<decltype(result), hn::Mask<D>>) {
        if constexpr (std::is_same_v<uint8_t, OUT>) {
          size_t bits_byte_idx = idx / 8;
          size_t bit_cursor = idx % 8;
          uint8_t cache_byte = 0;
          if constexpr (N < 8) {
            cache_byte = out[bits_byte_idx];
          }
          hn::StoreMaskBits(d, result, out + bits_byte_idx);
          if (bit_cursor > 0) {
            uint8_t current_byte = out[bits_byte_idx];
            uint8_t restore_value = ((current_byte << bit_cursor) | cache_byte);
            out[bits_byte_idx] = restore_value;
          }
        } else {
          static_assert(sizeof(OUT) == -1, "invalid out type");
        }
      } else {
        static_assert(sizeof(OUT) == -1, "invalid result type");
      }
    }
  }

  // `count` was a multiple of the vector length `N`: already done.
  if (HWY_UNLIKELY(idx == count)) return;

  const size_t remaining = count - idx;
  HWY_DASSERT(0 != remaining && remaining < N);
  hn::Vec<D> v1;
  if constexpr (std::is_same_v<hn::Vec<D>, T1>) {
    v1 = in1;
  } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T1>) {
    v1 = hn::LoadN(d, in1 + idx, remaining);
  } else {
    static_assert(sizeof(T1) == -1, "invalid T1");
  }
  hn::Vec<D> v2;
  if constexpr (std::is_same_v<hn::Vec<D>, T2>) {
    v2 = in2;
  } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T2>) {
    v2 = hn::LoadN(d, in2 + idx, remaining);
  } else {
    static_assert(sizeof(T2) == -1, "invalid T2");
  }
  auto result = func(d, v1, v2);
  if constexpr (std::is_same_v<decltype(result), hn::VFromD<D>>) {
    if constexpr (std::is_same_v<hn::TFromD<D>, OUT>) {
      hn::StoreN(result, d, out + idx, remaining);
    } else {
      static_assert(sizeof(OUT) == -1, "invalid out type");
    }
  } else if constexpr (std::is_same_v<decltype(result), hn::Mask<D>>) {
    if constexpr (std::is_same_v<uint8_t, OUT>) {
      size_t bits_byte_idx = idx / 8;
      size_t bit_cursor = idx % 8;
      uint8_t cache_byte = out[bits_byte_idx];
      hn::StoreMaskBits(d, result, out + bits_byte_idx);
      if (bit_cursor > 0) {
        uint8_t current_byte = out[bits_byte_idx];
        uint8_t restore_value = ((cache_byte << bit_cursor) | current_byte);
        out[bits_byte_idx] = restore_value;
      }
    } else {
      static_assert(sizeof(OUT) == -1, "invalid out type");
    }
  } else {
    static_assert(sizeof(result) == -1, "invalid result type");
  }
}

template <class D, class Func, typename T1, typename T2, typename T3, typename OUT = hn::TFromD<D>>
void do_ternary_transform(D d, T1 in1, T2 in2, T3 in3, size_t count, OUT* out, const Func& func) {
  const size_t N = hn::Lanes(d);
  size_t idx = 0;
  if (count >= N) {
    for (; idx <= count - N; idx += N) {
      hn::Vec<D> v1;
      if constexpr (std::is_same_v<hn::Vec<D>, T1>) {
        v1 = in1;
      } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T1>) {
        v1 = hn::LoadU(d, in1 + idx);
      } else {
        static_assert(sizeof(T1) == -1, "invalid T1");
      }
      hn::Vec<D> v2;
      if constexpr (std::is_same_v<hn::Vec<D>, T2>) {
        v2 = in2;
      } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T2>) {
        v2 = hn::LoadU(d, in2 + idx);
      } else {
        static_assert(sizeof(T2) == -1, "invalid T2");
      }
      hn::Vec<D> v3;
      if constexpr (std::is_same_v<hn::Vec<D>, T3>) {
        v3 = in3;
      } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T3>) {
        v3 = hn::LoadU(d, in3 + idx);
      } else {
        static_assert(sizeof(T3) == -1, "invalid T2");
      }
      hn::StoreU(func(d, v1, v2, v3), d, out + idx);
    }
  }

  // `count` was a multiple of the vector length `N`: already done.
  if (HWY_UNLIKELY(idx == count)) return;

  const size_t remaining = count - idx;
  HWY_DASSERT(0 != remaining && remaining < N);
  hn::Vec<D> v1;
  if constexpr (std::is_same_v<hn::Vec<D>, T1>) {
    v1 = in1;
  } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T1>) {
    v1 = hn::LoadN(d, in1 + idx, remaining);
  } else {
    static_assert(sizeof(T1) == -1, "invalid T1");
  }
  hn::Vec<D> v2;
  if constexpr (std::is_same_v<hn::Vec<D>, T2>) {
    v2 = in2;
  } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T2>) {
    v2 = LoadN(d, in2 + idx, remaining);
  } else {
    static_assert(sizeof(T2) == -1, "invalid T2");
  }
  hn::Vec<D> v3;
  if constexpr (std::is_same_v<hn::Vec<D>, T3>) {
    v3 = in3;
  } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T3>) {
    v3 = LoadN(d, in3 + idx, remaining);
  } else {
    static_assert(sizeof(T3) == -1, "invalid T2");
  }
  hn::StoreN(func(d, v1, v2, v3), d, out + idx, remaining);
}

template <class D, typename T2, typename T3, typename OUT = hn::TFromD<D>>
void do_select(D d, Vector<Bit> cond, T2 in2, T3 in3, size_t count, OUT* out) {
  const size_t N = hn::Lanes(d);
  size_t idx = 0;
  if (count >= N) {
    for (; idx <= count - N; idx += N) {
      hn::Vec<D> v2;
      if constexpr (std::is_same_v<hn::Vec<D>, T2>) {
        v2 = in2;
      } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T2>) {
        v2 = hn::LoadU(d, in2 + idx);
      } else {
        static_assert(sizeof(T2) == -1, "invalid T2");
      }
      hn::Vec<D> v3;
      if constexpr (std::is_same_v<hn::Vec<D>, T3>) {
        v3 = in3;
      } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T3>) {
        v3 = hn::LoadU(d, in3 + idx);
      } else {
        static_assert(sizeof(T3) == -1, "invalid T2");
      }
      size_t bits_byte_cursor = idx / 8;
      size_t bits_cursor = idx % 8;
      hn::Vec<D> result;
      if (bits_cursor > 0) {
        uint8_t tmp_bits[8];
        tmp_bits[0] = cond.Data()[bits_byte_cursor];
        tmp_bits[0] = (tmp_bits[0] >> bits_cursor);
        auto mask = hn::LoadMaskBits(d, tmp_bits);
        result = hn::IfThenElse(mask, v2, v3);
      } else {
        auto mask = hn::LoadMaskBits(d, cond.Data() + bits_byte_cursor);
        result = hn::IfThenElse(mask, v2, v3);
      }
      hn::StoreU(result, d, out + idx);
    }
  }

  // `count` was a multiple of the vector length `N`: already done.
  if (HWY_UNLIKELY(idx == count)) return;

  const size_t remaining = count - idx;
  HWY_DASSERT(0 != remaining && remaining < N);

  hn::Vec<D> v2;
  if constexpr (std::is_same_v<hn::Vec<D>, T2>) {
    v2 = in2;
  } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T2>) {
    v2 = LoadN(d, in2 + idx, remaining);
  } else {
    static_assert(sizeof(T2) == -1, "invalid T2");
  }
  hn::Vec<D> v3;
  if constexpr (std::is_same_v<hn::Vec<D>, T3>) {
    v3 = in3;
  } else if constexpr (std::is_same_v<const hn::TFromD<D>*, T3>) {
    v3 = LoadN(d, in3 + idx, remaining);
  } else {
    static_assert(sizeof(T3) == -1, "invalid T2");
  }
  size_t bits_byte_cursor = idx / 8;
  size_t bits_cursor = idx % 8;
  hn::Vec<D> result;
  if (bits_cursor > 0) {
    uint8_t tmp_bits[8];
    tmp_bits[0] = cond.Data()[bits_byte_cursor];
    tmp_bits[0] = (tmp_bits[0] >> bits_cursor);
    auto mask = hn::LoadMaskBits(d, tmp_bits);
    result = hn::IfThenElse(mask, v2, v3);
  } else {
    auto mask = hn::LoadMaskBits(d, cond.Data() + bits_byte_cursor);
    result = hn::IfThenElse(mask, v2, v3);
  }
  hn::StoreN(result, d, out + idx, remaining);
}

template <class D, OpToken op, typename V = hn::VFromD<D>>
static inline auto do_simd_unary_op(D d, V lv) {
  if constexpr (op == OP_SQRT) {
    return hn::Sqrt(lv);
  } else if constexpr (op == OP_FLOOR) {
    return hn::Floor(lv);
  } else if constexpr (op == OP_ABS) {
    return hn::Abs(lv);
  } else if constexpr (op == OP_NOT) {
    return hn::Not(lv);
  } else if constexpr (op == OP_COS) {
    return hn::Cos(d, lv);
  } else if constexpr (op == OP_SIN) {
    return hn::Sin(d, lv);
  } else if constexpr (op == OP_SINH) {
    return hn::Sinh(d, lv);
  } else if constexpr (op == OP_ASIN) {
    return hn::Asin(d, lv);
  } else if constexpr (op == OP_ACOS) {
    return hn::Acos(d, lv);
  } else if constexpr (op == OP_ATAN) {
    return hn::Atan(d, lv);
  } else if constexpr (op == OP_ATANH) {
    return hn::Atanh(d, lv);
  } else if constexpr (op == OP_SINH) {
    return hn::Sinh(d, lv);
  } else if constexpr (op == OP_TANH) {
    return hn::Tanh(d, lv);
  } else if constexpr (op == OP_ASINH) {
    return hn::Asinh(d, lv);
  } else if constexpr (op == OP_ACOSH) {
    return hn::Acosh(d, lv);
  } else if constexpr (op == OP_EXP) {
    return hn::Exp(d, lv);
  } else if constexpr (op == OP_EXP2) {
    return hn::Exp2(d, lv);
  } else if constexpr (op == OP_EXPM1) {
    return hn::Expm1(d, lv);
  } else if constexpr (op == OP_LOG) {
    return hn::Log(d, lv);
  } else if constexpr (op == OP_LOG2) {
    return hn::Log2(d, lv);
  } else if constexpr (op == OP_LOG10) {
    return hn::Log10(d, lv);
  } else if constexpr (op == OP_LOG1P) {
    return hn::Log1p(d, lv);
  } else if constexpr (op == OP_CEIL) {
    return hn::Ceil(lv);
  } else if constexpr (op == OP_COSH) {
    if constexpr (std::is_same_v<hn::TFromV<V>, float>) {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_coshf16_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_coshf8_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2
      auto val = Sleef_coshf4_u10(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported type");
#endif
    } else {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_coshd8_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_coshd4_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSSE3
      auto val = Sleef_coshd2_u10(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported target");
#endif
    }
  } else if constexpr (op == OP_COSH) {
    if constexpr (std::is_same_v<hn::TFromV<V>, float>) {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_coshf16_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_coshf8_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2
      auto val = Sleef_coshf4_u10(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported type");
#endif
    } else {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_coshd8_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_coshd4_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSSE3
      auto val = Sleef_coshd2_u10(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported target");
#endif
    }
  } else if constexpr (op == OP_TAN) {
    if constexpr (std::is_same_v<hn::TFromV<V>, float>) {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_tanf16_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_tanf8_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2
      auto val = Sleef_tanf4_u10(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported type");
#endif
    } else {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_tand8_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_tand4_u10(lv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSSE3
      auto val = Sleef_tand2_u10(lv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported target");
#endif
    }
  } else {
    static_assert(sizeof(V) == -1, "unsupported op");
  }
}

template <class D, OpToken op, typename V = hn::VFromD<D>>
static inline auto do_simd_binary_op(D d, V lv, V rv) {
  if constexpr (op == OP_PLUS || op == OP_PLUS_ASSIGN) {
    // if constexpr (is_bit) {
    //   return hn::Or(lv, rv);
    // } else {
    //   return hn::Add(lv, rv);
    // }
    return hn::Add(lv, rv);
  } else if constexpr (op == OP_MINUS || op == OP_MINUS_ASSIGN) {
    // if constexpr (is_bit) {
    //   return hn::And(lv, hn::Not(rv));
    // } else {
    //   return hn::Sub(lv, rv);
    // }
    return hn::Sub(lv, rv);
  } else if constexpr (op == OP_MULTIPLY || op == OP_MULTIPLY_ASSIGN) {
    return hn::Mul(lv, rv);
  } else if constexpr (op == OP_DIVIDE || op == OP_DIVIDE_ASSIGN) {
    return hn::Div(lv, rv);
  } else if constexpr (op == OP_MOD || op == OP_MOD_ASSIGN) {
    return hn::Mod(lv, rv);
  } else if constexpr (op == OP_LOGIC_OR) {
    return hn::Or(lv, rv);
  } else if constexpr (op == OP_LOGIC_AND) {
    return hn::And(lv, rv);
  } else if constexpr (op == OP_GREATER) {
    return hn::Gt(lv, rv);
  } else if constexpr (op == OP_GREATER_EQUAL) {
    return hn::Ge(lv, rv);
  } else if constexpr (op == OP_LESS) {
    return hn::Lt(lv, rv);
  } else if constexpr (op == OP_LESS_EQUAL) {
    return hn::Le(lv, rv);
  } else if constexpr (op == OP_EQUAL) {
    return hn::Eq(lv, rv);
  } else if constexpr (op == OP_NOT_EQUAL) {
    return hn::Ne(lv, rv);
  } else if constexpr (op == OP_MAX) {
    return hn::Max(lv, rv);
  } else if constexpr (op == OP_MIN) {
    return hn::Min(lv, rv);
  } else if constexpr (op == OP_HYPOT) {
    return hn::Hypot(d, lv, rv);
  } else if constexpr (op == OP_ATAN2) {
    return hn::Atan2(d, lv, rv);
  } else if constexpr (op == OP_POW) {
    if constexpr (std::is_same_v<hn::TFromV<V>, float>) {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_powf16_u10(lv.raw, rv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_powf8_u10(lv.raw, rv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSSE3 || HWY_TARGET == HWY_SSE2
      auto val = Sleef_powf4_u10(lv.raw, rv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported type");
#endif
    } else {
#if HWY_TARGET == HWY_AVX3 || HWY_TARGET == HWY_AVX3_ZEN4 || HWY_TARGET == HWY_AVX3_DL || HWY_TARGET == HWY_AVX3_SPR
      auto val = Sleef_powd8_u10(lv.raw, rv.raw);
      return V{val};
#elif HWY_TARGET == HWY_AVX2
      auto val = Sleef_powd4_u10(lv.raw, rv.raw);
      return V{val};
#elif HWY_TARGET == HWY_SSE4 || HWY_TARGET == HWY_SSE2 || HWY_TARGET == HWY_SSSE3
      auto val = Sleef_powd2_u10(lv.raw, rv.raw);
      return V{val};
#else
      static_assert(sizeof(lv.raw) == -1, "unsupported target");
#endif
    }
  } else {
    static_assert(sizeof(V) == -1, "unsupported op");
  }
}

template <class D, OpToken op, typename V = hn::VFromD<D>>
static inline auto do_simd_ternary_op(D d, V a, V b, V c) {
  if constexpr (op == OP_CLAMP) {
    return hn::Clamp(a, b, c);
  } else if constexpr (op == OP_CONDITIONAL) {
    return hn::IfThenElse(a, b, c);
  } else if constexpr (op == OP_FMA) {
    return hn::MulAdd(a, b, c);
  } else if constexpr (op == OP_FMS) {
    return hn::MulSub(a, b, c);
  } else if constexpr (op == OP_MULADDSUB) {
    return hn::MulAddSub(a, b, c);
  } else if constexpr (op == OP_FNMA) {
    return hn::NegMulAdd(a, b, c);
  } else if constexpr (op == OP_FNMS) {
    return hn::NegMulSub(a, b, c);
  } else {
    static_assert(sizeof(V) == -1, "unsupported op");
  }
}

template <typename T>
static auto get_constant(T v) {
  if constexpr (std::is_same_v<Bit, T>) {
    uint8_t t = v ? 1 : 0;
    return t;
  } else {
    return v;
  }
}
template <OpToken op, typename T = void>
static inline bool do_string_view_cmp(StringView lv, StringView rv) {
  if constexpr (op == OP_GREATER) {
    return lv > rv;
  } else if constexpr (op == OP_GREATER_EQUAL) {
    return lv >= rv;
  } else if constexpr (op == OP_LESS) {
    return lv < rv;
  } else if constexpr (op == OP_LESS_EQUAL) {
    return lv <= rv;
  } else if constexpr (op == OP_EQUAL) {
    return lv == rv;
  } else if constexpr (op == OP_NOT_EQUAL) {
    return lv != rv;
  } else {
    static_assert(sizeof(T) == -1, "unsupported cmp op");
  }
}

template <typename T, typename R, OpToken op>
static Vector<R> simd_vector_binary_scalar_op(Context& ctx, Vector<T> left, T right, bool reverse) {
  using number_t = typename InternalType<T>::internal_type;
  // constexpr bool is_bit = std::is_same_v<T, Bit>;
  const hn::ScalableTag<number_t> d;
  // constexpr auto lanes = hn::Lanes(d);
  using output_t = typename InternalType<R>::internal_type;

  VectorData result_data;
  if (ctx.IsTemporary(left)) {
    result_data = left.RawData();
  } else {
    result_data = ctx.NewSimdVector<R>(get_lanes<R>(), left.Size());
  }

  auto rv = hn::Set(d, get_constant(right));
  if (reverse) {
    do_binary_transform(d, rv, left.Data(), left.ElementSize(), result_data.MutableData<output_t>(),
                        do_simd_binary_op<decltype(d), op>);
  } else {
    do_binary_transform(d, left.Data(), rv, left.ElementSize(), result_data.MutableData<output_t>(),
                        do_simd_binary_op<decltype(d), op>);
  }

  return Vector<R>(result_data);
}

template <OpToken op>
Vector<Bit> simd_vector_string_cmp(Context& ctx, Vector<StringView> left, Vector<StringView> right) {
  if (left.Size() != right.Size()) {
    THROW_LOGIC_ERR(fmt::format("vector string_view size mismatch {}:{}", left.Size(), right.Size()));
  }
  auto lanes = get_lanes<Bit>();
  VectorData result_data;
  if (ctx.IsTemporary(left)) {
    result_data = left.RawData();
  } else if (ctx.IsTemporary(right)) {
    result_data = right.RawData();
  } else {
    result_data = ctx.NewSimdVector<Bit>(lanes, left.Size());
  }
  size_t bitset_n = left.Size() / 64;
  if (left.Size() % 64 > 0) {
    bitset_n++;
  }
  std::vector<hwy::BitSet64> bitsets(bitset_n);
  for (size_t i = 0; i < left.Size(); i++) {
    bool v = do_string_view_cmp<op>(left[i], right[i]);
    if (v) {
      bitsets[i / 64].Set(i % 64);
    } else {
      bitsets[i / 64].Clear(i % 64);
    }
  }
  memcpy(result_data.MutableData<uint8_t*>(), bitsets.data(), sizeof(uint64_t) * bitsets.size());
  return Vector<Bit>(result_data);
}

template <OpToken op>
Vector<Bit> simd_vector_string_cmp_scalar(Context& ctx, Vector<StringView> left, StringView right, bool reverse) {
  VectorData result_data;
  auto lanes = get_lanes<Bit>();
  if (ctx.IsTemporary(left)) {
    result_data = left.RawData();
  } else {
    result_data = ctx.NewSimdVector<Bit>(lanes, left.Size());
  }
  size_t bitset_n = left.Size() / 64;
  if (left.Size() % 64 > 0) {
    bitset_n++;
  }
  std::vector<hwy::BitSet64> bitsets(bitset_n);
  for (size_t i = 0; i < left.Size(); i++) {
    bool v = reverse ? do_string_view_cmp<op>(right, left[i]) : do_string_view_cmp<op>(left[i], right);
    if (v) {
      bitsets[i / 64].Set(i % 64);
    } else {
      bitsets[i / 64].Clear(i % 64);
    }
  }
  memcpy(result_data.MutableData<uint8_t*>(), bitsets.data(), sizeof(uint64_t) * bitsets.size());
  return Vector<Bit>(result_data);
}

template <typename OPT>
Vector<typename OPT::operand_t_1> simd_vector_binary_vector_scalar_op_impl(Context& ctx,
                                                                           Vector<typename OPT::operand_t> left,
                                                                           typename OPT::operand_t right) {
  if constexpr (std::is_same_v<StringView, typename OPT::operand_t>) {
    return simd_vector_string_cmp_scalar<OPT::op>(ctx, left, right, false);
  } else {
    return simd_vector_binary_scalar_op<typename OPT::operand_t, typename OPT::operand_t_1, OPT::op>(ctx, left, right,
                                                                                                     false);
  }
}

template <typename OPT>
Vector<typename OPT::operand_t_1> simd_vector_binary_scalar_vector_op_impl(Context& ctx, typename OPT::operand_t left,
                                                                           Vector<typename OPT::operand_t> right) {
  if constexpr (std::is_same_v<StringView, typename OPT::operand_t>) {
    return simd_vector_string_cmp_scalar<OPT::op>(ctx, right, left, true);
  } else {
    return simd_vector_binary_scalar_op<typename OPT::operand_t, typename OPT::operand_t_1, OPT::op>(ctx, right, left,
                                                                                                     true);
  }
}

template <typename OPT>
Vector<typename OPT::operand_t_1> simd_vector_binary_op_impl(Context& ctx, Vector<typename OPT::operand_t> left,
                                                             Vector<typename OPT::operand_t> right) {
  if (left.Size() != right.Size()) {
    THROW_LOGIC_ERR(
        fmt::format("vector binary op:{} arg vector's size mismatch {}:{}", OPT::op, left.Size(), right.Size()));
  }
  using operand_t = typename OPT::operand_t;
  using operand_t_1 = typename OPT::operand_t_1;
  if constexpr (std::is_same_v<StringView, operand_t>) {
    return simd_vector_string_cmp<OPT::op>(ctx, left, right);
  } else {
    using number_t = typename InternalType<operand_t>::internal_type;
    using output_t = typename InternalType<operand_t_1>::internal_type;
    // constexpr bool is_bit = std::is_same_v<T, Bit>;
    const hn::ScalableTag<number_t> d;
    // constexpr auto lanes = hn::Lanes(d);
    VectorData result_data;
    if (ctx.IsTemporary(left)) {
      result_data = left.RawData();
    } else if (ctx.IsTemporary(right)) {
      result_data = right.RawData();
    } else {
      result_data = ctx.NewSimdVector<operand_t_1>(get_lanes<operand_t_1>(), left.Size());
    }
    do_binary_transform(d, left.Data(), right.Data(), left.ElementSize(), result_data.MutableData<output_t>(),
                        do_simd_binary_op<decltype(d), OPT::op>);

    return Vector<operand_t_1>(result_data);
  }
}

template <typename T, typename D>
static inline auto select_ternary_value(Vector<Bit> cond, hn::VFromD<D> true_val, hn::VFromD<D> false_val, size_t i) {
  const D d;
  constexpr auto lanes = hn::Lanes(d);
  size_t bits_byte_cursor = i / 8;
  size_t bits_cursor = i % 8;
  if constexpr (lanes < 8) {
    uint8_t tmp_bits[8];
    tmp_bits[0] = cond.Data()[bits_byte_cursor];
    tmp_bits[0] = (tmp_bits[0] >> bits_cursor);
    auto mask = hn::LoadMaskBits(d, tmp_bits);
    auto v = hn::IfThenElse(mask, true_val, false_val);
    return v;
  } else {
    auto mask = hn::LoadMaskBits(d, cond.Data() + bits_byte_cursor);
    auto v = hn::IfThenElse(mask, true_val, false_val);
    return v;
  }
}

template <typename OPT>
Vector<typename OPT::operand_t> simd_vector_ternary_op_impl(Context& ctx, Vector<typename OPT::operand_t_1> a,
                                                            Vector<typename OPT::operand_t> b,
                                                            Vector<typename OPT::operand_t> c) {
  if (a.Size() != b.Size() || a.Size() != c.Size() || c.Size() != b.Size()) {
    THROW_LOGIC_ERR(
        fmt::format("vector ternary op:{} arg vector's size mismatch {}:{}:{}", OPT::op, a.Size(), b.Size(), c.Size()));
  }
  using operand_t = typename OPT::operand_t;
  using number_t = typename InternalType<operand_t>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  VectorData result_data;
  if (ctx.IsTemporary(a) && a.BytesCapacity() >= sizeof(number_t) * a.Size()) {
    result_data = a.RawData();
  } else if (ctx.IsTemporary(b)) {
    result_data = b.RawData();
  } else if (ctx.IsTemporary(c)) {
    result_data = c.RawData();
  } else {
    // result_data = arena_new_vector<T>(a.Size());
    result_data = ctx.NewSimdVector<number_t>(lanes, a.Size());
  }
  if constexpr (OPT::op == OP_CONDITIONAL) {
    do_select(d, a, b.Data(), c.Data(), b.Size(), result_data.MutableData<number_t>());
  } else {
    do_ternary_transform(d, a.Data(), b.Data(), c.Data(), b.Size(), result_data.MutableData<number_t>(),
                         do_simd_ternary_op<decltype(d), OPT::op>);
  }

  return Vector<operand_t>(result_data);
}
template <typename OPT>
Vector<typename OPT::operand_t> simd_vector_ternary_vector_vector_scalar_op_impl(Context& ctx,
                                                                                 Vector<typename OPT::operand_t_1> a,
                                                                                 Vector<typename OPT::operand_t> b,
                                                                                 typename OPT::operand_t c) {
  if (a.Size() != b.Size()) {
    THROW_LOGIC_ERR(fmt::format("vector ternary op:{} arg vector's size mismatch {}:{}", OPT::op, a.Size(), b.Size()));
  }
  using operand_t = typename OPT::operand_t;
  using number_t = typename InternalType<operand_t>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  VectorData result_data;
  if (ctx.IsTemporary(a) && a.BytesCapacity() >= sizeof(number_t) * a.Size()) {
    result_data = a.RawData();
  } else if (ctx.IsTemporary(b)) {
    result_data = b.RawData();
  } else {
    // result_data = arena_new_vector<T>(a.Size());
    result_data = ctx.NewSimdVector<number_t>(lanes, a.Size());
  }
  auto cv = hn::Set(d, get_constant(c));
  if constexpr (OPT::op == OP_CONDITIONAL) {
    do_select(d, a, b.Data(), cv, b.Size(), result_data.MutableData<number_t>());
  } else {
    do_ternary_transform(d, a.Data(), b.Data(), cv, b.Size(), result_data.MutableData<number_t>(),
                         do_simd_ternary_op<decltype(d), OPT::op>);
  }

  return Vector<operand_t>(result_data);
}
template <typename OPT>
Vector<typename OPT::operand_t> simd_vector_ternary_vector_scalar_vector_op_impl(Context& ctx,
                                                                                 Vector<typename OPT::operand_t_1> a,
                                                                                 typename OPT::operand_t b,
                                                                                 Vector<typename OPT::operand_t> c) {
  if (a.Size() != c.Size()) {
    THROW_LOGIC_ERR(fmt::format("vector ternary op:{} arg vector's size mismatch {}:{}", OPT::op, a.Size(), c.Size()));
  }
  using operand_t = typename OPT::operand_t;
  using number_t = typename InternalType<operand_t>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);

  VectorData result_data;
  if (ctx.IsTemporary(a) && a.BytesCapacity() >= sizeof(number_t) * a.Size()) {
    result_data = a.RawData();
  } else if (ctx.IsTemporary(c)) {
    result_data = c.RawData();
  } else {
    // result_data = arena_new_vector<T>(a.Size());
    result_data = ctx.NewSimdVector<number_t>(lanes, a.Size());
  }
  auto bv = hn::Set(d, get_constant(b));
  if constexpr (OPT::op == OP_CONDITIONAL) {
    do_select(d, a, bv, c.Data(), a.Size(), result_data.MutableData<number_t>());
  } else {
    do_ternary_transform(d, a.Data(), bv, c.Data(), a.Size(), result_data.MutableData<number_t>(),
                         do_simd_ternary_op<decltype(d), OPT::op>);
  }

  return Vector<operand_t>(result_data);
}
template <typename OPT>
Vector<typename OPT::operand_t> simd_vector_ternary_vector_scalar_scalar_op_impl(Context& ctx,
                                                                                 Vector<typename OPT::operand_t_1> a,
                                                                                 typename OPT::operand_t b,
                                                                                 typename OPT::operand_t c) {
  using operand_t = typename OPT::operand_t;
  using number_t = typename InternalType<operand_t>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);

  VectorData result_data;
  if (ctx.IsTemporary(a) && a.BytesCapacity() >= sizeof(number_t) * a.Size()) {
    result_data = a.RawData();
  } else {
    // result_data = arena_new_vector<T>(a.Size());
    result_data = ctx.NewSimdVector<number_t>(lanes, a.Size());
  }
  auto bv = hn::Set(d, get_constant(b));
  auto cv = hn::Set(d, get_constant(c));
  if constexpr (OPT::op == OP_CONDITIONAL) {
    do_select(d, a, bv, cv, a.Size(), result_data.MutableData<number_t>());
  } else {
    do_ternary_transform(d, a.Data(), bv, cv, a.Size(), result_data.MutableData<number_t>(),
                         do_simd_ternary_op<decltype(d), OPT::op>);
  }

  return Vector<operand_t>(result_data);
}
template <typename OPT>
Vector<typename OPT::operand_t> simd_vector_ternary_scalar_vector_vector_op_impl(Context& ctx,
                                                                                 typename OPT::operand_t_1 a,
                                                                                 Vector<typename OPT::operand_t> b,
                                                                                 Vector<typename OPT::operand_t> c) {
  if (b.Size() != c.Size()) {
    THROW_LOGIC_ERR(fmt::format("vector ternary op:{} arg vector's size mismatch {}:{}", OPT::op, b.Size(), c.Size()));
  }
  using operand_t = typename OPT::operand_t;
  using number_t = typename InternalType<operand_t>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  VectorData result_data;
  if (ctx.IsTemporary(b)) {
    result_data = b.RawData();
  } else if (ctx.IsTemporary(c)) {
    result_data = c.RawData();
  } else {
    // result_data = arena_new_vector<T>(b.Size());
    result_data = ctx.NewSimdVector<number_t>(lanes, b.Size());
  }
  auto av = hn::Set(d, get_constant(a));
  do_ternary_transform(d, av, b.Data(), c.Data(), b.Size(), result_data.MutableData<number_t>(),
                       do_simd_ternary_op<decltype(d), OPT::op>);

  return Vector<operand_t>(result_data);
}
template <typename OPT>
Vector<typename OPT::operand_t> simd_vector_ternary_scalar_scalar_vector_op_impl(Context& ctx,
                                                                                 typename OPT::operand_t_1 a,
                                                                                 typename OPT::operand_t b,
                                                                                 Vector<typename OPT::operand_t> c) {
  using operand_t = typename OPT::operand_t;
  using number_t = typename InternalType<operand_t>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  VectorData result_data;
  if (ctx.IsTemporary(c)) {
    result_data = c.RawData();
  } else {
    // result_data = arena_new_vector<T>(c.Size());
    result_data = ctx.NewSimdVector<number_t>(lanes, c.Size());
  }
  auto av = hn::Set(d, get_constant(a));
  auto bv = hn::Set(d, get_constant(b));
  do_ternary_transform(d, av, bv, c.Data(), c.Size(), result_data.MutableData<number_t>(),
                       do_simd_ternary_op<decltype(d), OPT::op>);

  return Vector<operand_t>(result_data);
}
template <typename OPT>
Vector<typename OPT::operand_t> simd_vector_ternary_scalar_vector_scalar_op_impl(Context& ctx,
                                                                                 typename OPT::operand_t_1 a,
                                                                                 Vector<typename OPT::operand_t> b,
                                                                                 typename OPT::operand_t c) {
  using operand_t = typename OPT::operand_t;
  using number_t = typename InternalType<operand_t>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  VectorData result_data;
  if (ctx.IsTemporary(b)) {
    result_data = b.RawData();
  } else {
    // result_data = arena_new_vector<T>(b.Size());
    result_data = ctx.NewSimdVector<number_t>(lanes, b.Size());
  }
  auto av = hn::Set(d, get_constant(a));
  auto cv = hn::Set(d, get_constant(c));
  do_ternary_transform(d, av, b.Data(), cv, b.Size(), result_data.MutableData<number_t>(),
                       do_simd_ternary_op<decltype(d), OPT::op>);

  return Vector<operand_t>(result_data);
}

template <typename OPT>
Vector<typename OPT::operand_t> simd_vector_unary_op_impl(Context& ctx, Vector<typename OPT::operand_t> left) {
  using number_t = typename InternalType<typename OPT::operand_t>::internal_type;
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  VectorData result_data;
  if (ctx.IsTemporary(left)) {
    result_data = left.RawData();
  } else {
    result_data = ctx.NewSimdVector<number_t>(lanes, left.Size());
  }
  do_unary_transform(d, left.Data(), left.ElementSize(), result_data.MutableData<number_t>(),
                     do_simd_unary_op<decltype(d), OPT::op>);
  return Vector<typename OPT::operand_t>(result_data);
}

template <typename T>
T simd_vector_dot_impl(Vector<T> left, Vector<T> right) {
  if (left.Size() != right.Size()) {
    THROW_LOGIC_ERR(fmt::format("vector dot size mismatch {}:{}", left.Size(), right.Size()));
  }
  using D = hn::ScalableTag<T>;
  const D d;
  constexpr auto lanes = hn::Lanes(d);
  T val;
  if (left.Size() >= lanes) {
    constexpr auto assumptions = hn::Dot::Assumptions::kAtLeastOneVector;
    val = hn::Dot::Compute<assumptions, D, T>(d, left.Data(), right.Data(), left.Size());
  } else {
    constexpr auto assumptions = hn::Dot::Assumptions::kPaddedToVector;
    val = hn::Dot::Compute<assumptions, D, T>(d, left.Data(), right.Data(), left.Size());
  }
  return val;
}

template <typename T>
T simd_vector_sum_impl(Vector<T> left) {
  using number_t = typename InternalType<T>::internal_type;
  T sum = {};
  const hn::ScalableTag<number_t> d;
  constexpr auto lanes = hn::Lanes(d);
  size_t i = 0;
  for (; (i + lanes) < left.Size(); i += lanes) {
    auto lv = hn::LoadU(d, left.Data() + i);
    auto sum_v = hn::ReduceSum(d, lv);
    sum += sum_v;
  }
  if (i < left.Size()) {
    for (; i < left.Size(); i++) {
      sum += left[i];
    }
  }
  return sum;
}

template <typename T>
Vector<T> simd_vector_iota_impl(Context& ctx, T start, uint32_t n) {
  // auto result_data = arena_new_vector<T>(n);
  const hn::ScalableTag<T> d;
  constexpr auto lanes = hn::Lanes(d);
  auto result_data = ctx.NewSimdVector<T>(lanes, n);
  uint8_t* arena_data = result_data.template MutableData<uint8_t>();
  size_t i = 0;
  for (; i < n; i += lanes) {
    auto v = hn::Iota(d, start + i);
    hn::StoreU(v, d, reinterpret_cast<T*>(arena_data + i * sizeof(T)));
  }
  return Vector<T>(result_data);
}

template <typename T>
Vector<T> simd_vector_clone_impl(Context& ctx, Vector<T> data) {
  auto result_data = ctx.NewSimdVector<T>(get_lanes<T>(), data.Size());
  memcpy(result_data.template MutableData<uint8_t>(), data.Data(), result_data.BytesCapacity());
  return Vector<T>(result_data);
}

}  // namespace HWY_NAMESPACE
}  // namespace simd
}  // namespace rapidudf
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rapidudf {
namespace simd {

template <typename T, OpToken opt, typename T1 = void>
struct OpTypes {
  using operand_t = T;
  using operand_t_1 = T1;
  static constexpr OpToken op = opt;
};

template <typename T>
Vector<T> simd_vector_clone(Context& ctx, Vector<T> data) {
  HWY_EXPORT_T(Table1, simd_vector_clone_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, data);
}
template <typename T>
Vector<T> simd_vector_iota(Context& ctx, T start, uint32_t n) {
  HWY_EXPORT_T(Table1, simd_vector_iota_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, start, n);
}
template <typename T>
T simd_vector_sum(Vector<T> left) {
  HWY_EXPORT_T(Table1, simd_vector_sum_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(left);
}

template <typename T>
T simd_vector_dot(Vector<T> left, Vector<T> right) {
  HWY_EXPORT_T(Table1, simd_vector_dot_impl<T>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(left, right);
}

template <typename T, OpToken op>
Vector<T> simd_vector_unary_op(Context& ctx, Vector<T> left) {
  using OPT = OpTypes<T, op>;
  HWY_EXPORT_T(Table1, simd_vector_unary_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, left);
}

template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_scalar_vector_scalar_op(Context& ctx, R a, Vector<T> b, T c) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_ternary_scalar_vector_scalar_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, a, b, c);
}
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_scalar_scalar_vector_op(Context& ctx, R a, T b, Vector<T> c) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_ternary_scalar_scalar_vector_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, a, b, c);
}
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_scalar_vector_vector_op(Context& ctx, R a, Vector<T> b, Vector<T> c) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_ternary_scalar_vector_vector_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, a, b, c);
}
template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_vector_scalar_scalar_op(Context& ctx, Vector<R> a, T b, T c) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_ternary_vector_scalar_scalar_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, a, b, c);
}

template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_vector_scalar_vector_op(Context& ctx, Vector<R> a, T b, Vector<T> c) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_ternary_vector_scalar_vector_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, a, b, c);
}

template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_vector_vector_scalar_op(Context& ctx, Vector<R> a, Vector<T> b, T c) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_ternary_vector_vector_scalar_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, a, b, c);
}

template <typename R, typename T, OpToken op>
Vector<T> simd_vector_ternary_op(Context& ctx, Vector<R> a, Vector<T> b, Vector<T> c) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_ternary_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, a, b, c);
}

template <typename T, typename R, OpToken op>
Vector<R> simd_vector_binary_op(Context& ctx, Vector<T> left, Vector<T> right) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_binary_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, left, right);
}

template <typename T, typename R, OpToken op>
Vector<R> simd_vector_binary_vector_scalar_op(Context& ctx, Vector<T> left, T right) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_binary_vector_scalar_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, left, right);
}

template <typename T, typename R, OpToken op>
Vector<R> simd_vector_binary_scalar_vector_op(Context& ctx, T left, Vector<T> right) {
  using OPT = OpTypes<T, op, R>;
  HWY_EXPORT_T(Table1, simd_vector_binary_scalar_vector_op_impl<OPT>);
  return HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, left, right);
}

#define DEFINE_SIMD_UNARY_OP_TEMPLATE(r, op, ii, TYPE) \
  template Vector<TYPE> simd_vector_unary_op<TYPE, op>(Context & ctx, Vector<TYPE> left);
#define DEFINE_SIMD_UNARY_OP(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_UNARY_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_UNARY_OP(OP_NOT, Bit);
DEFINE_SIMD_UNARY_OP(OP_COS, float, double);
DEFINE_SIMD_UNARY_OP(OP_SIN, float, double);
DEFINE_SIMD_UNARY_OP(OP_TAN, float, double);
DEFINE_SIMD_UNARY_OP(OP_TANH, float, double);
DEFINE_SIMD_UNARY_OP(OP_ATANH, float, double);
DEFINE_SIMD_UNARY_OP(OP_SINH, float, double);
DEFINE_SIMD_UNARY_OP(OP_COSH, float, double);
DEFINE_SIMD_UNARY_OP(OP_ACOS, float, double);
DEFINE_SIMD_UNARY_OP(OP_ASIN, float, double);
DEFINE_SIMD_UNARY_OP(OP_ATAN, float, double);
DEFINE_SIMD_UNARY_OP(OP_ACOSH, float, double);
DEFINE_SIMD_UNARY_OP(OP_ASINH, float, double);
DEFINE_SIMD_UNARY_OP(OP_EXP, float, double);
DEFINE_SIMD_UNARY_OP(OP_EXP2, float, double);
DEFINE_SIMD_UNARY_OP(OP_EXPM1, float, double);
DEFINE_SIMD_UNARY_OP(OP_LOG, float, double);
DEFINE_SIMD_UNARY_OP(OP_LOG2, float, double);
DEFINE_SIMD_UNARY_OP(OP_LOG10, float, double);
DEFINE_SIMD_UNARY_OP(OP_LOG1P, float, double);
DEFINE_SIMD_UNARY_OP(OP_SQRT, float, double);
DEFINE_SIMD_UNARY_OP(OP_FLOOR, float, double);
DEFINE_SIMD_UNARY_OP(OP_CEIL, float, double);
DEFINE_SIMD_UNARY_OP(OP_ABS, float, double, int64_t, int32_t, int16_t, int8_t);

#define DEFINE_SIMD_BINARY_BOOL_OP_TEMPLATE(r, op, ii, TYPE)                                                       \
  template Vector<Bit> simd_vector_binary_op<TYPE, Bit, op>(Context & ctx, Vector<TYPE> left, Vector<TYPE> right); \
  template Vector<Bit> simd_vector_binary_vector_scalar_op<TYPE, Bit, op>(Context & ctx, Vector<TYPE> left,        \
                                                                          TYPE right);                             \
  template Vector<Bit> simd_vector_binary_scalar_vector_op<TYPE, Bit, op>(Context & ctx, TYPE left, Vector<TYPE> right);

#define DEFINE_SIMD_BINARY_BOOL_OP(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_BINARY_BOOL_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_BINARY_BOOL_OP(OP_GREATER, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t, StringView);
DEFINE_SIMD_BINARY_BOOL_OP(OP_GREATER_EQUAL, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t,
                           uint8_t, int8_t, StringView);
DEFINE_SIMD_BINARY_BOOL_OP(OP_LESS, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t, StringView);
DEFINE_SIMD_BINARY_BOOL_OP(OP_LESS_EQUAL, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t,
                           uint8_t, int8_t, StringView);
DEFINE_SIMD_BINARY_BOOL_OP(OP_EQUAL, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t, StringView);
DEFINE_SIMD_BINARY_BOOL_OP(OP_NOT_EQUAL, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t,
                           uint8_t, int8_t, StringView);
DEFINE_SIMD_BINARY_BOOL_OP(OP_LOGIC_AND, Bit);
DEFINE_SIMD_BINARY_BOOL_OP(OP_LOGIC_OR, Bit);

#define DEFINE_SIMD_BINARY_MATH_OP_TEMPLATE(r, op, ii, TYPE)                                                         \
  template Vector<TYPE> simd_vector_binary_op<TYPE, TYPE, op>(Context & ctx, Vector<TYPE> left, Vector<TYPE> right); \
  template Vector<TYPE> simd_vector_binary_vector_scalar_op<TYPE, TYPE, op>(Context & ctx, Vector<TYPE> left,        \
                                                                            TYPE right);                             \
  template Vector<TYPE> simd_vector_binary_scalar_vector_op<TYPE, TYPE, op>(Context & ctx, TYPE left,                \
                                                                            Vector<TYPE> right);

#define DEFINE_SIMD_BINARY_MATH_OP(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_BINARY_MATH_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))

DEFINE_SIMD_BINARY_MATH_OP(OP_PLUS, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t, Bit);
DEFINE_SIMD_BINARY_MATH_OP(OP_MINUS, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t, Bit);
DEFINE_SIMD_BINARY_MATH_OP(OP_MULTIPLY, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_DIVIDE, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_MOD, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_PLUS_ASSIGN, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t,
                           uint8_t, int8_t, Bit);
DEFINE_SIMD_BINARY_MATH_OP(OP_MINUS_ASSIGN, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t,
                           uint8_t, int8_t, Bit);
DEFINE_SIMD_BINARY_MATH_OP(OP_MULTIPLY_ASSIGN, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t,
                           uint8_t, int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_DIVIDE_ASSIGN, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t,
                           uint8_t, int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_MOD_ASSIGN, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_MAX, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_MIN, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                           int8_t);
DEFINE_SIMD_BINARY_MATH_OP(OP_HYPOT, float, double);
DEFINE_SIMD_BINARY_MATH_OP(OP_ATAN2, float, double);
DEFINE_SIMD_BINARY_MATH_OP(OP_POW, float, double);

#define DEFINE_SIMD_TERNARY_COND_OP_TEMPLATE(r, op, ii, TYPE)                                              \
  template Vector<TYPE> simd_vector_ternary_op<Bit, TYPE, OP_CONDITIONAL>(Context & ctx, Vector<Bit> a,    \
                                                                          Vector<TYPE> b, Vector<TYPE> c); \
  template Vector<TYPE> simd_vector_ternary_vector_vector_scalar_op<Bit, TYPE, OP_CONDITIONAL>(            \
      Context & ctx, Vector<Bit> a, Vector<TYPE> b, TYPE c);                                               \
  template Vector<TYPE> simd_vector_ternary_vector_scalar_vector_op<Bit, TYPE, OP_CONDITIONAL>(            \
      Context & ctx, Vector<Bit> a, TYPE b, Vector<TYPE> c);                                               \
  template Vector<TYPE> simd_vector_ternary_vector_scalar_scalar_op<Bit, TYPE, OP_CONDITIONAL>(            \
      Context & ctx, Vector<Bit> a, TYPE b, TYPE c);

#define DEFINE_SIMD_TERNARY_COND_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_TERNARY_COND_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_TERNARY_COND_OP(float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);

#define DEFINE_SIMD_TERNARY_OP_TEMPLATE(r, op, ii, TYPE)                                                             \
  template Vector<TYPE> simd_vector_ternary_op<TYPE, TYPE, op>(Context & ctx, Vector<TYPE> a, Vector<TYPE> b,        \
                                                               Vector<TYPE> c);                                      \
  template Vector<TYPE> simd_vector_ternary_vector_vector_scalar_op<TYPE, TYPE, op>(Context & ctx, Vector<TYPE> a,   \
                                                                                    Vector<TYPE> b, TYPE c);         \
  template Vector<TYPE> simd_vector_ternary_vector_scalar_vector_op<TYPE, TYPE, op>(Context & ctx, Vector<TYPE> a,   \
                                                                                    TYPE b, Vector<TYPE> c);         \
  template Vector<TYPE> simd_vector_ternary_vector_scalar_scalar_op<TYPE, TYPE, op>(Context & ctx, Vector<TYPE> a,   \
                                                                                    TYPE b, TYPE c);                 \
  template Vector<TYPE> simd_vector_ternary_scalar_vector_vector_op<TYPE, TYPE, op>(Context & ctx, TYPE a,           \
                                                                                    Vector<TYPE> b, Vector<TYPE> c); \
  template Vector<TYPE> simd_vector_ternary_scalar_scalar_vector_op<TYPE, TYPE, op>(Context & ctx, TYPE a, TYPE b,   \
                                                                                    Vector<TYPE> c);                 \
  template Vector<TYPE> simd_vector_ternary_scalar_vector_scalar_op<TYPE, TYPE, op>(Context & ctx, TYPE a,           \
                                                                                    Vector<TYPE> b, TYPE c);

#define DEFINE_SIMD_TERNARY_OP(op, ...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_TERNARY_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_TERNARY_OP(OP_CLAMP, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                       int8_t);
DEFINE_SIMD_TERNARY_OP(OP_FMA, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);
DEFINE_SIMD_TERNARY_OP(OP_FMS, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);
DEFINE_SIMD_TERNARY_OP(OP_MULADDSUB, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                       int8_t);
DEFINE_SIMD_TERNARY_OP(OP_FNMA, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                       int8_t);
DEFINE_SIMD_TERNARY_OP(OP_FNMS, float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t,
                       int8_t);

#define DEFINE_SIMD_DOT_OP_TEMPLATE(r, op, ii, TYPE) \
  template TYPE simd_vector_dot(Vector<TYPE> left, Vector<TYPE> right);
#define DEFINE_SIMD_DOT_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_DOT_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_DOT_OP(float, double);

#define DEFINE_SIMD_IOTA_OP_TEMPLATE(r, op, ii, TYPE) \
  template Vector<TYPE> simd_vector_iota(Context&, TYPE start, uint32_t n);
#define DEFINE_SIMD_IOTA_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_IOTA_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_IOTA_OP(float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);

#define DEFINE_SIMD_CLONE_OP_TEMPLATE(r, op, ii, TYPE) \
  template Vector<TYPE> simd_vector_clone(Context& ctx, Vector<TYPE> data);
#define DEFINE_SIMD_CLONE_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_CLONE_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_CLONE_OP(float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);

#define DEFINE_SIMD_SUM_OP_TEMPLATE(r, op, ii, TYPE) template TYPE simd_vector_sum(Vector<TYPE> vec);
#define DEFINE_SIMD_SUM_OP(...) \
  BOOST_PP_SEQ_FOR_EACH_I(DEFINE_SIMD_SUM_OP_TEMPLATE, op, BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))
DEFINE_SIMD_SUM_OP(float, double, uint64_t, int64_t, uint32_t, int32_t, uint16_t, int16_t, uint8_t, int8_t);
}  // namespace simd
}  // namespace rapidudf
#endif  // HWY_ONCE
