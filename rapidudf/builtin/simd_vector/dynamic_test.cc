// Generates code for every target that this compiler can support.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "rapidudf/builtin/simd_vector/dynamic_test.cc"  // this file
#include "hwy/foreach_target.h"                                            // must come before highway.h
#include "hwy/highway.h"
#include "rapidudf/context/context.h"
#include "rapidudf/types/simd_vector.h"

HWY_BEFORE_NAMESPACE();
namespace rapidudf {
namespace HWY_NAMESPACE {

// Highway ops reside here; ADL does not find templates nor builtins.
namespace hn = hwy::HWY_NAMESPACE;

// Example of a type-agnostic (caller-specified lane type) and width-agnostic
// (uses best available instruction set) function in a header.
//
// Computes x[i] = mul_array[i] * x_array[i] + add_array[i] for i < size.
template <typename T>
HWY_MAYBE_UNUSED void MulAddLoop(const typename T::_0* HWY_RESTRICT mul_array,
                                 const typename T::_0* HWY_RESTRICT add_array, const size_t size,
                                 typename T::_0* HWY_RESTRICT x_array) {
  const hn::ScalableTag<T> d;
  for (size_t i = 0; i < size; i += hn::Lanes(d)) {
    const auto mul = hn::Load(d, mul_array + i);
    const auto add = hn::Load(d, add_array + i);
    auto x = hn::Load(d, x_array + i);
    x = hn::MulAdd(mul, x, add);
    hn::Store(x, d, x_array + i);
  }
}

template <typename T>
simd::Vector<T> simd_vector_clone_impl(Context& ctx, simd::Vector<T> data) {
  return simd::Vector<T>();
  // return 1;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace rapidudf
HWY_AFTER_NAMESPACE();

// The table of pointers to the various implementations in HWY_NAMESPACE must
// be compiled only once (foreach_target #includes this file multiple times).
// HWY_ONCE is true for only one of these 'compilation passes'.
#if HWY_ONCE

namespace rapidudf {
template <typename T0, typename T1>
struct ManyArgs {
  using _0 = T0;
  using _1 = T1;
};
template <typename T>
void CallMulAddLoop(const T* HWY_RESTRICT mul_array, const T* HWY_RESTRICT add_array, const size_t size,
                    T* HWY_RESTRICT x_array) {
  // This must reside outside of HWY_NAMESPACE because it references (calls the
  // appropriate one from) the per-target implementations there.
  // For static dispatch, use HWY_STATIC_DISPATCH.
  using XType = ManyArgs<T, int>;
  HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(MulAddLoop<XType>)(mul_array, add_array, size, x_array);
  //   return HWY_DYNAMIC_DISPATCH(MulAddLoop)(mul_array, add_array, size, x_array);
}

template <typename T>
simd::Vector<T> simd_vector_clone1(Context& ctx, simd::Vector<T> data) {
  HWY_EXPORT_T(Table1, simd_vector_clone_impl<T>);
  //   HWY_DYNAMIC_DISPATCH_T(HWY_DISPATCH_TABLE_T())

  auto x = HWY_DYNAMIC_DISPATCH_T(Table1)(ctx, data);
  //   int a = HWY_EXPORT_AND_DYNAMIC_DISPATCH_T(simd_vector_clone_impl<T>)(ctx, data);
}

}  // namespace rapidudf
#endif  // HWY_ONCE