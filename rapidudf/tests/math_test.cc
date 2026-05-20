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

// Consolidated math-operation test suite. Replaces arithmetic_test.cc,
// the earlier minimal math_test.cc, and the math-heavy tests that lived in
// simd_vector_test.cc.
//
// Coverage matrix:
//   * Arithmetic operators  + - * / %                   (i32 / f32 / f64; scalar + vector)
//   * Power                 ^   (= pow)                 (i64 scalar; f32/f64 scalar+vector)
//   * Unary FP functions    sqrt, cbrt, exp, exp2,
//                           expm1, log, log2, log10,
//                           log1p, sin, cos, tan,
//                           asin, acos, atan, sinh,
//                           cosh, tanh, asinh, acosh,
//                           atanh, floor, ceil, round,
//                           rint, trunc, abs, erf, erfc (f32 / f64; scalar + vector)
//   * Binary FP functions   pow, hypot, atan2,
//                           abs_diff, max, min          (f32 / f64; scalar + vector)
//   * Integer-only          abs, abs_diff, %            (i32; scalar + vector)
//   * Complex expressions   trig tree, wilson_ctr,
//                           sigmoid, boost_scores
//   * Negative type tests   bit/string for arithmetic;
//                           int for trig

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "rapidudf/functions/simd/vector_misc.h"
#include "rapidudf/log/log.h"
#include "rapidudf/rapidudf.h"

using namespace rapidudf;

namespace {

// ---------------------------------------------------------------------------
// Compile helpers
// ---------------------------------------------------------------------------

template <typename Ret, typename A>
auto JitScalar1(const std::string& expr) {
  JitCompiler compiler;
  auto rc = compiler.CompileExpression<Ret, A>(expr, {"x"});
  EXPECT_TRUE(rc.ok()) << rc.status().ToString();
  return std::move(rc).value();
}

template <typename Ret, typename A, typename B>
auto JitScalar2(const std::string& expr) {
  JitCompiler compiler;
  auto rc = compiler.CompileExpression<Ret, A, B>(expr, {"x", "y"});
  EXPECT_TRUE(rc.ok()) << rc.status().ToString();
  return std::move(rc).value();
}

template <typename T>
auto JitVector1(const std::string& expr) {
  JitCompiler compiler;
  auto rc = compiler.CompileExpression<Vector<T>, Context&, Vector<T>>(expr, {"_", "x"});
  EXPECT_TRUE(rc.ok()) << rc.status().ToString();
  return std::move(rc).value();
}

template <typename T>
auto JitVector2(const std::string& expr) {
  JitCompiler compiler;
  auto rc = compiler.CompileExpression<Vector<T>, Context&, Vector<T>, Vector<T>>(expr, {"_", "x", "y"});
  EXPECT_TRUE(rc.ok()) << rc.status().ToString();
  return std::move(rc).value();
}

template <typename T>
::testing::AssertionResult ApproxEq(T actual, T expected, const char* tag = "") {
  if constexpr (std::is_floating_point_v<T>) {
    // 4-ULP tolerance via gtest's float-equal helpers.
    auto fpc = ::testing::internal::FloatingPoint<T>(actual);
    auto efp = ::testing::internal::FloatingPoint<T>(expected);
    if (fpc.AlmostEquals(efp)) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << tag << " actual=" << actual << " expected=" << expected << " diff=" << (actual - expected);
  } else {
    if (actual == expected) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << tag << " actual=" << actual << " expected=" << expected;
  }
}

// Scalar unary helper: compile `<fn>(x)` for type T and check against `native`
// over a sample of inputs.
template <typename T, typename NativeFn>
void CheckScalarUnaryFn(const char* fn_name, NativeFn native, const std::vector<T>& inputs) {
  auto f = JitScalar1<T, T>(std::string(fn_name) + "(x)");
  for (T x : inputs) {
    EXPECT_TRUE(ApproxEq(f(x), native(x), fn_name)) << " input=" << x;
  }
}

// Vector unary helper: compile `<fn>(x)` for simd_vector<T> and check.
template <typename T, typename NativeFn>
void CheckVectorUnaryFn(const char* fn_name, NativeFn native, const std::vector<T>& inputs) {
  auto f = JitVector1<T>(std::string(fn_name) + "(x)");
  Context ctx;
  auto out = f(ctx, inputs);
  ASSERT_EQ(out.Size(), inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    EXPECT_TRUE(ApproxEq(out[i], native(inputs[i]), fn_name)) << " idx=" << i << " input=" << inputs[i];
  }
}

template <typename T, typename NativeFn>
void CheckScalarBinaryFn(const char* fn_name, NativeFn native, const std::vector<std::pair<T, T>>& inputs) {
  auto f = JitScalar2<T, T, T>(std::string(fn_name) + "(x,y)");
  for (auto [x, y] : inputs) {
    EXPECT_TRUE(ApproxEq(f(x, y), native(x, y), fn_name)) << " x=" << x << " y=" << y;
  }
}

template <typename T, typename NativeFn>
void CheckVectorBinaryFn(const char* fn_name, NativeFn native, const std::vector<std::pair<T, T>>& inputs) {
  std::vector<T> xs, ys;
  xs.reserve(inputs.size());
  ys.reserve(inputs.size());
  for (auto [x, y] : inputs) {
    xs.push_back(x);
    ys.push_back(y);
  }
  auto f = JitVector2<T>(std::string(fn_name) + "(x,y)");
  Context ctx;
  auto out = f(ctx, xs, ys);
  ASSERT_EQ(out.Size(), inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    EXPECT_TRUE(ApproxEq(out[i], native(xs[i], ys[i]), fn_name)) << " idx=" << i;
  }
}

// Sample input domains for FP unary functions.
const std::vector<float> kF32SafeInputs = {0.5f, 1.0f, 1.5f, 2.5f, 4.0f, 9.0f, 0.25f, 100.0f};
const std::vector<double> kF64SafeInputs = {0.5, 1.0, 1.5, 2.5, 4.0, 9.0, 0.25, 100.0};
const std::vector<float> kF32SignedInputs = {-2.5f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 2.5f};
const std::vector<double> kF64SignedInputs = {-2.5, -1.0, -0.25, 0.0, 0.25, 1.0, 2.5};
const std::vector<float> kF32UnitInputs = {-0.9f, -0.5f, 0.0f, 0.5f, 0.9f};   // for asin/acos/atanh
const std::vector<double> kF64UnitInputs = {-0.9, -0.5, 0.0, 0.5, 0.9};

}  // namespace

// ===========================================================================
// SECTION 1 -- Arithmetic operators (+, -, *, /, %)
// ===========================================================================

namespace {

template <typename T>
void RunArithScalar(const char* op, std::function<T(T, T)> native, const std::vector<std::pair<T, T>>& cases) {
  std::string expr = "x" + std::string(op) + "y";
  auto f = JitScalar2<T, T, T>(expr);
  for (auto [a, b] : cases) {
    EXPECT_TRUE(ApproxEq(f(a, b), native(a, b), op)) << " x=" << a << " y=" << b;
  }
}

template <typename T>
void RunArithVector(const char* op, std::function<T(T, T)> native, const std::vector<T>& xs, const std::vector<T>& ys) {
  ASSERT_EQ(xs.size(), ys.size());
  std::string expr = "x" + std::string(op) + "y";
  auto f = JitVector2<T>(expr);
  Context ctx;
  auto out = f(ctx, xs, ys);
  ASSERT_EQ(out.Size(), xs.size());
  for (size_t i = 0; i < xs.size(); ++i) {
    EXPECT_TRUE(ApproxEq(out[i], native(xs[i], ys[i]), op)) << " idx=" << i;
  }
}

}  // namespace

TEST(MathArith, Scalar_Add) {
  RunArithScalar<int32_t>("+", [](int32_t a, int32_t b) { return a + b; }, {{1, 2}, {-3, 4}, {100, -50}});
  RunArithScalar<float>("+", [](float a, float b) { return a + b; }, {{1.5f, 2.5f}, {-1.0f, 0.25f}});
  RunArithScalar<double>("+", [](double a, double b) { return a + b; }, {{111.1, 222.2}, {1.1, 2.2}});
}

TEST(MathArith, Scalar_Sub) {
  RunArithScalar<int32_t>("-", [](int32_t a, int32_t b) { return a - b; }, {{100, 121}, {0, 5}, {-7, -3}});
  RunArithScalar<float>("-", [](float a, float b) { return a - b; }, {{3.5f, 1.5f}, {-1.0f, 0.25f}});
  RunArithScalar<double>("-", [](double a, double b) { return a - b; }, {{30000.14, 12421.4}, {0.0, 1.0}});
}

TEST(MathArith, Scalar_Mul) {
  RunArithScalar<int32_t>("*", [](int32_t a, int32_t b) { return a * b; }, {{100, 121}, {-3, 4}});
  RunArithScalar<float>("*", [](float a, float b) { return a * b; }, {{2.5f, 4.0f}});
  RunArithScalar<double>("*", [](double a, double b) { return a * b; }, {{30000.14, 12421.4}});
}

TEST(MathArith, Scalar_Div) {
  RunArithScalar<int32_t>("/", [](int32_t a, int32_t b) { return a / b; }, {{100, 7}, {-50, 3}});
  RunArithScalar<float>("/", [](float a, float b) { return a / b; }, {{1.0f, 2.0f}, {-3.5f, 7.0f}});
  RunArithScalar<double>("/", [](double a, double b) { return a / b; }, {{30000.14, 12421.4}});
}

TEST(MathArith, Scalar_Mod) {
  RunArithScalar<int32_t>("%", [](int32_t a, int32_t b) { return a % b; }, {{100, 121}, {17, 5}, {-13, 4}});
}

TEST(MathArith, Vector_Add) {
  std::vector<int32_t> xi = {1, -2, 3, -4, 4, 8, -1};
  std::vector<int32_t> yi = {10, -20, 30, -40, 40, 80, -10};
  RunArithVector<int32_t>("+", [](int32_t a, int32_t b) { return a + b; }, xi, yi);

  std::vector<float> xf = {1, -2.1f, 3.2f, -4, 4, 8.66f, -1, 6.7f};
  std::vector<float> yf = {1, -2.1f, 3.2f, -4, 4, 8.457f, -1, 2.2f};
  RunArithVector<float>("+", [](float a, float b) { return a + b; }, xf, yf);

  std::vector<double> xd = {1, -2.1, 3.2, -4, 4};
  std::vector<double> yd = {1, -2.1, 3.2, -4, 4};
  RunArithVector<double>("+", [](double a, double b) { return a + b; }, xd, yd);
}

TEST(MathArith, Vector_Sub) {
  std::vector<int32_t> xi = {1, -2, 3, -4, 4, 8, -1};
  std::vector<int32_t> yi = {10, -20, 30, -40, 40, 80, -10};
  RunArithVector<int32_t>("-", [](int32_t a, int32_t b) { return a - b; }, xi, yi);

  std::vector<float> xf = {1, -2.1f, 3.2f, -4, 4};
  std::vector<float> yf = {1, -2.1f, 3.2f, -4, 4};
  RunArithVector<float>("-", [](float a, float b) { return a - b; }, xf, yf);

  std::vector<double> xd = {1, -2.1, 3.2, -4, 4};
  std::vector<double> yd = {1, -2.1, 3.2, -4, 4};
  RunArithVector<double>("-", [](double a, double b) { return a - b; }, xd, yd);
}

TEST(MathArith, Vector_Mul) {
  std::vector<int32_t> xi = {1, -2, 3, -4, 4};
  std::vector<int32_t> yi = {10, -20, 30, -40, 40};
  RunArithVector<int32_t>("*", [](int32_t a, int32_t b) { return a * b; }, xi, yi);

  std::vector<float> xf = {1, -2.1f, 3.2f, -4, 4};
  std::vector<float> yf = {1, -2.1f, 3.2f, -4, 4};
  RunArithVector<float>("*", [](float a, float b) { return a * b; }, xf, yf);

  std::vector<double> xd = {1, -2.1, 3.2, -4, 4};
  std::vector<double> yd = {1, -2.1, 3.2, -4, 4};
  RunArithVector<double>("*", [](double a, double b) { return a * b; }, xd, yd);
}

TEST(MathArith, Vector_Div) {
  std::vector<int32_t> xi = {10, -20, 30, -40, 80};
  std::vector<int32_t> yi = {3, -7, 5, 2, 9};
  RunArithVector<int32_t>("/", [](int32_t a, int32_t b) { return a / b; }, xi, yi);

  std::vector<float> xf = {1.0f, -2.1f, 3.2f, -4.0f, 4.0f};
  std::vector<float> yf = {2.0f, 1.5f, -3.2f, 8.0f, 0.5f};
  RunArithVector<float>("/", [](float a, float b) { return a / b; }, xf, yf);

  std::vector<double> xd = {1.0, -2.1, 3.2};
  std::vector<double> yd = {2.0, 1.5, -3.2};
  RunArithVector<double>("/", [](double a, double b) { return a / b; }, xd, yd);
}

TEST(MathArith, Vector_Mod_I32) {
  std::vector<int32_t> x = {1, -2, 3, -4, 4, 8, -1};
  std::vector<int32_t> y = {10, -20, 30, -40, 40, 80, -10};
  RunArithVector<int32_t>("%", [](int32_t a, int32_t b) { return a % b; }, x, y);
}

TEST(MathArith, Scalar_With_Literal) {
  // Mixed scalar-and-literal forms.
  auto fadd = JitVector1<float>("x+5");
  auto fsub = JitVector1<float>("x-5");
  auto fmul = JitVector1<float>("5*x");
  auto fdiv = JitVector1<float>("5/x");
  auto fmod_i = JitVector1<int32_t>("x%5");
  auto fchain = JitVector1<float>("x+5+10");

  Context ctx;
  std::vector<float> v = {1.0f, 2.0f, 3.0f};
  std::vector<int32_t> vi = {1, 2, 3};
  for (size_t i = 0; i < v.size(); ++i) {
    EXPECT_FLOAT_EQ(fadd(ctx, v)[i], v[i] + 5);
    EXPECT_FLOAT_EQ(fsub(ctx, v)[i], v[i] - 5);
    EXPECT_FLOAT_EQ(fmul(ctx, v)[i], 5 * v[i]);
    EXPECT_FLOAT_EQ(fdiv(ctx, v)[i], 5 / v[i]);
    EXPECT_EQ(fmod_i(ctx, vi)[i], vi[i] % 5);
    EXPECT_FLOAT_EQ(fchain(ctx, v)[i], v[i] + 5 + 10);
  }
}

// ===========================================================================
// SECTION 2 -- Power operator `^`
// ===========================================================================

TEST(MathPow, Scalar_F64) {
  auto f = JitScalar2<double, double, double>("x^y");
  EXPECT_DOUBLE_EQ(f(30000.14, 5.2), std::pow(30000.14, 5.2));
}

TEST(MathPow, Scalar_I64) {
  auto f = JitScalar2<int64_t, int64_t, int64_t>("x^y");
  EXPECT_EQ(f(100, 6), static_cast<int64_t>(std::pow(100, 6)));
}

TEST(MathPow, Scalar_F32_With_Multiply) {
  auto f = JitScalar2<float, float, float>("2*x^y");
  EXPECT_FLOAT_EQ(f(3, 2), 2 * std::pow(3.0f, 2.0f));
  EXPECT_FLOAT_EQ(f(7, 5), 2 * std::pow(7.0f, 5.0f));
}

TEST(MathPow, Scalar_Multi_Term) {
  // Ranking-style multi-term pow expression (kept from arithmetic_test.cc).
  std::string expr =
      "(Click^10.0)*((Like+0.000082)^4.7)*(Inter^3.5)*((Join+0.000024)^5.5)*(TimeV1^7.0)*"
      "((PostComment+0.000024)^3.5)*((PositiveCommentV1+0.0038)^1.0)*(ExpoTimeV1^1.5)";
  JitCompiler compiler;
  auto rc = compiler.CompileExpression<double, double, double, double, double, double, double, double, double>(
      expr, {"Click", "Like", "Inter", "Join", "TimeV1", "PostComment", "PositiveCommentV1", "ExpoTimeV1"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  double Click = 10, Like = 20, Inter = 2, Join = 3, TimeV1 = 11, PostComment = 12, PositiveCommentV1 = 3,
         ExpoTimeV1 = 4;
  double expected = std::pow(Click, 10.0) * std::pow(Like + 0.000082, 4.7) * std::pow(Inter, 3.5) *
                    std::pow(Join + 0.000024, 5.5) * std::pow(TimeV1, 7.0) * std::pow(PostComment + 0.000024, 3.5) *
                    std::pow(PositiveCommentV1 + 0.0038, 1.0) * std::pow(ExpoTimeV1, 1.5);
  EXPECT_DOUBLE_EQ(f(Click, Like, Inter, Join, TimeV1, PostComment, PositiveCommentV1, ExpoTimeV1), expected);
}

TEST(MathPow, Vector_F32) {
  std::vector<float> x = {1, 2.1f, 3.2f, 4, 4, 8.66f, 1, 6.7f};
  std::vector<float> y = {1.1f, 2.1f, 3.1f, 4.1f, 4.1f, 2.7f, 1.2f, 2.4f};
  auto f = JitVector2<float>("x^y");
  Context ctx;
  auto out = f(ctx, x, y);
  ASSERT_EQ(out.Size(), x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(out[i], std::pow(x[i], y[i])) << " idx=" << i;
  }
}

TEST(MathPow, Vector_F64) {
  std::vector<double> x = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  auto f = JitVector1<double>("pow(x,10)");
  Context ctx;
  auto out = f(ctx, x);
  ASSERT_EQ(out.Size(), x.size());
  for (size_t i = 0; i < x.size(); ++i) {
    EXPECT_DOUBLE_EQ(out[i], std::pow(x[i], 10.0));
  }
}

// ===========================================================================
// SECTION 3 -- Unary FP functions (sqrt, trig, hyperbolic, exp/log, rounding,
//                                  erf/erfc) for f32 and f64, scalar + vector
// ===========================================================================

#define MATH_UNARY_FP_TEST(NAME, FN, INPUTS_F32, INPUTS_F64)                                             \
  TEST(MathUnary, Scalar_##NAME##_F32) {                                                                 \
    CheckScalarUnaryFn<float>(#FN, [](float v) { return std::FN(v); }, INPUTS_F32);                      \
  }                                                                                                      \
  TEST(MathUnary, Scalar_##NAME##_F64) {                                                                 \
    CheckScalarUnaryFn<double>(#FN, [](double v) { return std::FN(v); }, INPUTS_F64);                    \
  }                                                                                                      \
  TEST(MathUnary, Vector_##NAME##_F32) {                                                                 \
    CheckVectorUnaryFn<float>(#FN, [](float v) { return std::FN(v); }, INPUTS_F32);                      \
  }                                                                                                      \
  TEST(MathUnary, Vector_##NAME##_F64) {                                                                 \
    CheckVectorUnaryFn<double>(#FN, [](double v) { return std::FN(v); }, INPUTS_F64);                    \
  }

// Domain: positive (sqrt, log family) -> kFxxSafeInputs.
MATH_UNARY_FP_TEST(Sqrt, sqrt, kF32SafeInputs, kF64SafeInputs)
MATH_UNARY_FP_TEST(Log, log, kF32SafeInputs, kF64SafeInputs)
MATH_UNARY_FP_TEST(Log2, log2, kF32SafeInputs, kF64SafeInputs)
MATH_UNARY_FP_TEST(Log10, log10, kF32SafeInputs, kF64SafeInputs)
MATH_UNARY_FP_TEST(Log1p, log1p, kF32SafeInputs, kF64SafeInputs)

// Domain: any real -> kFxxSignedInputs.
MATH_UNARY_FP_TEST(Exp, exp, kF32SignedInputs, kF64SignedInputs)
MATH_UNARY_FP_TEST(Exp2, exp2, kF32SignedInputs, kF64SignedInputs)
MATH_UNARY_FP_TEST(Expm1, expm1, kF32SignedInputs, kF64SignedInputs)
MATH_UNARY_FP_TEST(Sin, sin, kF32SignedInputs, kF64SignedInputs)
MATH_UNARY_FP_TEST(Cos, cos, kF32SignedInputs, kF64SignedInputs)
MATH_UNARY_FP_TEST(Tan, tan, kF32SignedInputs, kF64SignedInputs)
MATH_UNARY_FP_TEST(Atan, atan, kF32SignedInputs, kF64SignedInputs)
MATH_UNARY_FP_TEST(Sinh, sinh, kF32SignedInputs, kF64SignedInputs)
MATH_UNARY_FP_TEST(Cosh, cosh, kF32SignedInputs, kF64SignedInputs)
MATH_UNARY_FP_TEST(Tanh, tanh, kF32SignedInputs, kF64SignedInputs)
MATH_UNARY_FP_TEST(Asinh, asinh, kF32SignedInputs, kF64SignedInputs)
MATH_UNARY_FP_TEST(Erf, erf, kF32SignedInputs, kF64SignedInputs)
MATH_UNARY_FP_TEST(Erfc, erfc, kF32SignedInputs, kF64SignedInputs)
MATH_UNARY_FP_TEST(Floor, floor, kF32SignedInputs, kF64SignedInputs)
MATH_UNARY_FP_TEST(Ceil, ceil, kF32SignedInputs, kF64SignedInputs)
MATH_UNARY_FP_TEST(Trunc, trunc, kF32SignedInputs, kF64SignedInputs)
MATH_UNARY_FP_TEST(Abs, abs, kF32SignedInputs, kF64SignedInputs)

// Round: scalar uses std::round (ties away from zero) while the vector path
// uses Highway's Round (ties to even). Avoid `.5` inputs so both impls agree.
namespace {
const std::vector<float> kF32RoundInputs = {-2.7f, -1.3f, -0.4f, 0.0f, 0.4f, 1.3f, 2.7f};
const std::vector<double> kF64RoundInputs = {-2.7, -1.3, -0.4, 0.0, 0.4, 1.3, 2.7};
}  // namespace
MATH_UNARY_FP_TEST(Round, round, kF32RoundInputs, kF64RoundInputs)

// Domain: [-1, 1] -> kFxxUnitInputs.
MATH_UNARY_FP_TEST(Asin, asin, kF32UnitInputs, kF64UnitInputs)
MATH_UNARY_FP_TEST(Acos, acos, kF32UnitInputs, kF64UnitInputs)
MATH_UNARY_FP_TEST(Atanh, atanh, kF32UnitInputs, kF64UnitInputs)

// Domain: [1, inf) for acosh.
TEST(MathUnary, Scalar_Acosh_F32) {
  CheckScalarUnaryFn<float>("acosh", [](float v) { return std::acosh(v); }, {1.0f, 1.5f, 2.5f, 4.0f});
}
TEST(MathUnary, Scalar_Acosh_F64) {
  CheckScalarUnaryFn<double>("acosh", [](double v) { return std::acosh(v); }, {1.0, 1.5, 2.5, 4.0});
}
TEST(MathUnary, Vector_Acosh_F32) {
  CheckVectorUnaryFn<float>("acosh", [](float v) { return std::acosh(v); }, {1.0f, 1.5f, 2.5f, 4.0f});
}
TEST(MathUnary, Vector_Acosh_F64) {
  CheckVectorUnaryFn<double>("acosh", [](double v) { return std::acosh(v); }, {1.0, 1.5, 2.5, 4.0});
}

#undef MATH_UNARY_FP_TEST

// Rint: SLEEF rint == round-to-nearest-even ("banker's rounding"). std::rint
// follows the current rounding mode (default also nearest-even), so direct
// comparison is OK on default mode.
TEST(MathUnary, Scalar_Rint_F32) {
  CheckScalarUnaryFn<float>("rint", [](float v) { return std::rint(v); }, kF32SignedInputs);
}
TEST(MathUnary, Scalar_Rint_F64) {
  CheckScalarUnaryFn<double>("rint", [](double v) { return std::rint(v); }, kF64SignedInputs);
}
TEST(MathUnary, Vector_Rint_F32) {
  CheckVectorUnaryFn<float>("rint", [](float v) { return std::rint(v); }, kF32SignedInputs);
}
TEST(MathUnary, Vector_Rint_F64) {
  CheckVectorUnaryFn<double>("rint", [](double v) { return std::rint(v); }, kF64SignedInputs);
}

// ===========================================================================
// SECTION 4 -- Binary FP functions (pow, hypot, atan2, abs_diff, min, max)
// ===========================================================================

namespace {

const std::vector<std::pair<float, float>> kF32BinPairs = {
    {1.0f, 2.0f}, {0.5f, 0.5f}, {3.0f, 4.0f}, {-1.5f, 2.5f}, {7.7f, 1.1f}};
const std::vector<std::pair<double, double>> kF64BinPairs = {
    {1.0, 2.0}, {0.5, 0.5}, {3.0, 4.0}, {-1.5, 2.5}, {7.7, 1.1}};

}  // namespace

TEST(MathBinary, Hypot_Scalar_F32) {
  CheckScalarBinaryFn<float>("hypot", [](float a, float b) { return std::hypot(a, b); }, kF32BinPairs);
}
TEST(MathBinary, Hypot_Scalar_F64) {
  CheckScalarBinaryFn<double>("hypot", [](double a, double b) { return std::hypot(a, b); }, kF64BinPairs);
}
TEST(MathBinary, Hypot_Vector_F32) {
  CheckVectorBinaryFn<float>("hypot", [](float a, float b) { return std::hypot(a, b); }, kF32BinPairs);
}
TEST(MathBinary, Hypot_Vector_F64) {
  CheckVectorBinaryFn<double>("hypot", [](double a, double b) { return std::hypot(a, b); }, kF64BinPairs);
}

TEST(MathBinary, Atan2_Scalar_F32) {
  CheckScalarBinaryFn<float>("atan2", [](float a, float b) { return std::atan2(a, b); }, kF32BinPairs);
}
TEST(MathBinary, Atan2_Scalar_F64) {
  CheckScalarBinaryFn<double>("atan2", [](double a, double b) { return std::atan2(a, b); }, kF64BinPairs);
}
TEST(MathBinary, Atan2_Vector_F32) {
  CheckVectorBinaryFn<float>("atan2", [](float a, float b) { return std::atan2(a, b); }, kF32BinPairs);
}
TEST(MathBinary, Atan2_Vector_F64) {
  CheckVectorBinaryFn<double>("atan2", [](double a, double b) { return std::atan2(a, b); }, kF64BinPairs);
}

TEST(MathBinary, Pow_Scalar_F32) {
  CheckScalarBinaryFn<float>(
      "pow", [](float a, float b) { return std::pow(a, b); },
      {{2.0f, 3.0f}, {0.5f, 4.0f}, {3.0f, 2.5f}, {7.7f, 1.1f}});
}
TEST(MathBinary, Pow_Scalar_F64) {
  CheckScalarBinaryFn<double>(
      "pow", [](double a, double b) { return std::pow(a, b); },
      {{2.0, 3.0}, {0.5, 4.0}, {3.0, 2.5}, {7.7, 1.1}});
}

TEST(MathBinary, AbsDiff_Vector_F32) {
  std::vector<float> x = {1, -2.1f, 3.2f, -4, 4};
  std::vector<float> y = {2, -2.1f, 1.0f, 7, -4};
  auto f = JitVector2<float>("abs_diff(x,y)");
  Context ctx;
  auto out = f(ctx, x, y);
  ASSERT_EQ(out.Size(), x.size());
  for (size_t i = 0; i < x.size(); ++i) EXPECT_FLOAT_EQ(out[i], std::abs(x[i] - y[i]));
}

TEST(MathBinary, AbsDiff_Vector_I32) {
  std::vector<int32_t> x = {1, -2, 3, -4, 4};
  std::vector<int32_t> y = {2, -2, 1, 7, -4};
  auto f = JitVector2<int32_t>("abs_diff(x,y)");
  Context ctx;
  auto out = f(ctx, x, y);
  ASSERT_EQ(out.Size(), x.size());
  for (size_t i = 0; i < x.size(); ++i) EXPECT_EQ(out[i], std::abs(x[i] - y[i]));
}

TEST(MathBinary, Max_Min_Scalar) {
  auto fmax_i = JitScalar2<int32_t, int32_t, int32_t>("max(x,y)");
  auto fmin_i = JitScalar2<int32_t, int32_t, int32_t>("min(x,y)");
  EXPECT_EQ(fmax_i(3, 7), 7);
  EXPECT_EQ(fmin_i(3, 7), 3);
  auto fmax_f = JitScalar2<float, float, float>("max(x,y)");
  auto fmin_f = JitScalar2<float, float, float>("min(x,y)");
  EXPECT_FLOAT_EQ(fmax_f(1.5f, 2.5f), 2.5f);
  EXPECT_FLOAT_EQ(fmin_f(1.5f, 2.5f), 1.5f);
}

// ===========================================================================
// SECTION 5 -- Integer-specific ops (abs, abs_diff, modulo)
// ===========================================================================

TEST(MathInt, Abs_Scalar_I32) {
  auto f = JitScalar1<int32_t, int32_t>("abs(x)");
  EXPECT_EQ(f(-7), 7);
  EXPECT_EQ(f(0), 0);
  EXPECT_EQ(f(123), 123);
}

TEST(MathInt, Abs_Vector_I32) {
  std::vector<int32_t> x = {-7, 0, 123, -1, 99};
  auto f = JitVector1<int32_t>("abs(x)");
  Context ctx;
  auto out = f(ctx, x);
  ASSERT_EQ(out.Size(), x.size());
  for (size_t i = 0; i < x.size(); ++i) EXPECT_EQ(out[i], std::abs(x[i]));
}

// ===========================================================================
// SECTION 6 -- Negative type tests
// ===========================================================================

TEST(MathNegative, Arith_Bit_Type_Scalar) {
  JitCompiler compiler;
  for (auto op : {"x+y", "x-y", "x*y", "x/y", "x%y"}) {
    auto rc = compiler.CompileExpression<Bit, Bit, Bit>(op, {"x", "y"});
    EXPECT_FALSE(rc.ok()) << op;
  }
}

TEST(MathNegative, Arith_StringView_Type_Vector) {
  JitCompiler compiler;
  for (auto op : {"x+y", "x-y", "x*y", "x/y"}) {
    auto rc = compiler.CompileExpression<Vector<StringView>, Context&, Vector<StringView>, Vector<StringView>>(
        op, {"_", "x", "y"});
    EXPECT_FALSE(rc.ok()) << op;
  }
}

TEST(MathNegative, Pow_Vector_Int_Forbidden) {
  JitCompiler compiler;
  auto rc = compiler.CompileExpression<Vector<int>, Context&, Vector<int>, Vector<int>>("x^y", {"_", "x", "y"});
  EXPECT_FALSE(rc.ok());
}

// ===========================================================================
// SECTION 7 -- Complex expressions
// ===========================================================================

TEST(MathComplex, Trig_Tree_Vector_F64) {
  // x + (cos(y - sin(2/x*pi)) - sin(x - cos(2*y/pi))) - y
  std::string source = R"(
    simd_vector<f64> test_func(Context ctx, simd_vector<f64> x, simd_vector<f64> y, double pi) {
      return x + (cos(y - sin(2 / x * pi)) - sin(x - cos(2 * y / pi))) - y;
    }
  )";
  JitCompiler compiler;
  auto rc =
      compiler.CompileFunction<Vector<double>, Context&, Vector<double>, Vector<double>, double>(source);
  ASSERT_TRUE(rc.ok()) << rc.status().ToString();
  auto f = std::move(rc.value());

  double pi = 3.14159265358979323846264338327950288419716939937510;
  std::vector<double> xs, ys;
  for (size_t i = 0; i < 1024; ++i) {
    xs.push_back(i + 1.0);
    ys.push_back(i + 101.0);
  }
  Context ctx;
  auto out = f(ctx, xs, ys, pi);
  ASSERT_EQ(out.Size(), xs.size());
  for (size_t i = 0; i < xs.size(); ++i) {
    double x = xs[i], y = ys[i];
    double expected = x + (std::cos(y - std::sin(2 / x * pi)) - std::sin(x - std::cos(2 * y / pi))) - y;
    EXPECT_DOUBLE_EQ(out[i], expected);
  }
}

TEST(MathComplex, Exp_Tree_Scalar_F64) {
  // a * exp(2.2 / 3.3 * t) + c
  std::string source = R"(
    double test_func(double a, double t, double c) {
      return a * exp(2.2/3.3*t) + c;
    }
  )";
  JitCompiler compiler;
  auto rc = compiler.CompileFunction<double, double, double, double>(source);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  double a = 1.2, t = 2.1, c = 3.3;
  EXPECT_DOUBLE_EQ(f(a, t, c), a * std::exp(2.2 / 3.3 * t) + c);
}

namespace {

float __attribute__((noinline)) NativeSigmoid(float duration, float alpha, float beta) {
  float x = (duration - alpha) / beta;
  return 1.0f / (1 + std::exp(-x));
}

}  // namespace

TEST(MathComplex, Sigmoid_Vector_F32) {
  std::string source = R"(
    simd_vector<f32> sigmoid(Context ctx, simd_vector<f32> duration, f32 alpha, f32 beta) {
      auto x = (duration - alpha) / beta;
      return 1.0_f32 / (1_f32 + exp(-x));
    }
  )";
  JitCompiler compiler;
  auto rc =
      compiler.CompileFunction<Vector<float>, Context&, Vector<float>, float, float>(source);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> distr(1.0f, 100.0f);
  std::vector<float> duration;
  for (size_t i = 0; i < 1024; ++i) duration.push_back(distr(gen));

  float alpha = 30.0f, beta = 10.0f;
  Context ctx;
  auto out = f(ctx, duration, alpha, beta);
  ASSERT_EQ(out.Size(), duration.size());
  for (size_t i = 0; i < duration.size(); ++i) {
    EXPECT_FLOAT_EQ(out[i], NativeSigmoid(duration[i], alpha, beta)) << " idx=" << i;
  }
}

namespace {

double NativeWilsonCtr(double exp_cnt, double clk_cnt) {
  return std::log10(exp_cnt) *
         (clk_cnt / exp_cnt + 1.96 * 1.96 / (2 * exp_cnt) -
          1.96 / (2 * exp_cnt) *
              std::sqrt(4 * exp_cnt * (1 - clk_cnt / exp_cnt) * clk_cnt / exp_cnt + 1.96 * 1.96)) /
         (1 + 1.96 * 1.96 / exp_cnt);
}

}  // namespace

TEST(MathComplex, WilsonCtr_Vector_F64) {
  std::string source = R"(
    simd_vector<f64> wilson_ctr(Context ctx, simd_vector<f64> exp_cnt, simd_vector<f64> clk_cnt) {
      return log10(exp_cnt) *
             (clk_cnt / exp_cnt + 1.96 * 1.96 / (2 * exp_cnt) -
              1.96 / (2 * exp_cnt) *
                sqrt(4 * exp_cnt * (1 - clk_cnt / exp_cnt) * clk_cnt / exp_cnt + 1.96 * 1.96)) /
             (1 + 1.96 * 1.96 / exp_cnt);
    }
  )";
  JitCompiler compiler;
  auto rc = compiler.CompileFunction<Vector<double>, Context&, Vector<double>, Vector<double>>(source);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());

  std::vector<double> exp_cnt, clk_cnt;
  for (size_t i = 0; i < 1024; ++i) {
    int v = static_cast<int>(i % 90 + 1);
    clk_cnt.push_back(static_cast<double>(v));
    exp_cnt.push_back(static_cast<double>(v + 10));
  }
  Context ctx;
  auto out = f(ctx, exp_cnt, clk_cnt);
  ASSERT_EQ(out.Size(), exp_cnt.size());
  for (size_t i = 0; i < exp_cnt.size(); ++i) {
    EXPECT_DOUBLE_EQ(out[i], NativeWilsonCtr(exp_cnt[i], clk_cnt[i])) << " idx=" << i;
  }
}

struct Feeds {
  rapidudf::Vector<double> Click;
  rapidudf::Vector<double> Like;
};
RUDF_STRUCT_FIELDS(Feeds, Click, Like)

TEST(MathComplex, BoostScores_Feeds_Struct) {
  std::string source = R"(
    simd_vector<f64> boost_scores(Context ctx, Feeds feeds) {
      auto score = pow(feeds.Click, 10.0);
      score *= pow(feeds.Like + 0.000082, 4.7);
      return score;
    }
  )";
  rapidudf::JitCompiler compiler;
  auto rc = compiler.CompileFunction<rapidudf::Vector<double>, rapidudf::Context&, Feeds&>(source);
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());

  std::vector<double> clicks, likes;
  for (size_t i = 0; i < 100; ++i) {
    clicks.push_back(i + 11.0);
    likes.push_back(i + 12.0);
  }
  Feeds feeds;
  feeds.Click = clicks;
  feeds.Like = likes;
  rapidudf::Context ctx;
  auto out = f(ctx, feeds);
  ASSERT_EQ(out.Size(), clicks.size());
  for (size_t i = 0; i < clicks.size(); ++i) {
    double expected = std::pow(clicks[i], 10.0) * std::pow(likes[i] + 0.000082, 4.7);
    EXPECT_DOUBLE_EQ(out[i], expected) << " idx=" << i;
  }
}

// ===========================================================================
// SECTION 8 -- Distance / inner-product C++ library APIs
//   `dot_distance`, `l2_distance`, `cos_distance` are exposed as templated
//   C++ APIs in `rapidudf::functions::simd_vector_*`. They have OP_* enums
//   in optype.h but are not yet wired up as JIT builtins under those names
//   (the JIT-side mangled symbol differs from the actual function name --
//   tracked separately).
// ===========================================================================

namespace {

float NativeDot(const std::vector<float>& a, const std::vector<float>& b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
}

float NativeL2Norm(const std::vector<float>& v) { return std::sqrt(NativeDot(v, v)); }

float NativeL2Distance(const std::vector<float>& a, const std::vector<float>& b) {
  float s = 0;
  for (size_t i = 0; i < a.size(); ++i) s += (a[i] - b[i]) * (a[i] - b[i]);
  return std::sqrt(s);
}

float NativeCosineDistance(const std::vector<float>& a, const std::vector<float>& b) {
  float dot = NativeDot(a, b);
  float na = NativeL2Norm(a);
  float nb = NativeL2Norm(b);
  if (na == 0.0f || nb == 0.0f) return 0.0f;
  return 1.0f - dot / (na * nb);
}

}  // namespace

TEST(MathDistance, Dot_LibAPI_F32) {
  std::vector<float> a, b;
  for (size_t i = 0; i < 100; ++i) {
    a.push_back(i + 1.1f);
    b.push_back(i + 0.5f);
  }
  EXPECT_FLOAT_EQ(rapidudf::functions::simd_vector_dot_distance<float>(a, b), NativeDot(a, b));
}

TEST(MathDistance, L2Distance_LibAPI_F32) {
  std::vector<float> a, b;
  for (size_t i = 0; i < 100; ++i) {
    a.push_back(i + 1.1f);
    b.push_back(i + 1.88f);
  }
  EXPECT_FLOAT_EQ(rapidudf::functions::simd_vector_l2_distance<float>(a, b), NativeL2Distance(a, b));
}

TEST(MathDistance, CosineDistance_LibAPI_F32) {
  std::vector<float> a, b;
  for (size_t i = 0; i < 124; ++i) {
    a.push_back(i + 1.1f);
    b.push_back(i + 1.58f);
  }
  EXPECT_FLOAT_EQ(rapidudf::functions::simd_vector_cosine_distance<float>(a, b), NativeCosineDistance(a, b));
}

// ===========================================================================
// SECTION 9 -- Misc (random builtin)
// ===========================================================================

TEST(MathMisc, Random_Smoke) {
  // Just verify random() compiles and produces values; no equality check.
  rapidudf::JitCompiler compiler;
  auto rc = compiler.CompileExpression<uint64_t, uint64_t>("random(now_s())", {"x"});
  ASSERT_TRUE(rc.ok());
  auto f = std::move(rc.value());
  // Three calls, just ensure it doesn't crash.
  (void)f(1);
  (void)f(2);
  (void)f(3);
}
