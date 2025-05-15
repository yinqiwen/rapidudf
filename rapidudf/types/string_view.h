/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include <functional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "fmt/format.h"

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"

namespace rapidudf {

// Variable length string or binary type for use in vectors. This has
// semantics similar to std::string_view or folly::StringPiece and
// exposes a subset of the interface. If the string is 12 characters
// or less, it is inlined and no reference is held. If it is longer, a
// reference to the string is held and the 4 first characters are
// cached in the StringView. This allows failing comparisons early and
// reduces the CPU cache working set when dealing with short strings.
//
// Adapted from TU Munich Umbra and CWI DuckDB.
//
// TODO: Extend the interface to parity with folly::StringPiece as needed.
struct StringView {
 public:
  using value_type = char;

  static constexpr size_t kPrefixSize = 4 * sizeof(char);
  static constexpr size_t kInlineSize = 12;

  StringView() {
    static_assert(sizeof(StringView) == 16);
    memset(this, 0, sizeof(StringView));
  }

  StringView(const StringView& other) = default;
  StringView& operator=(const StringView& other) = default;

  StringView(const char* data, int32_t len) : size_(len) {
    // VELOX_CHECK_GE(len, 0);
    // VELOX_DCHECK(data || len == 0);
    if (isInline()) {
      // Zero the inline part.
      // this makes sure that inline strings can be compared for equality with 2
      // int64 compares.
      memset(prefix_, 0, kPrefixSize);
      if (size_ == 0) {
        return;
      }
      // small string: inlined. Zero the last 8 bytes first to allow for whole
      // word comparison.
      value_.data = nullptr;
      memcpy(prefix_, data, size_);
    } else {
      // large string: store pointer
      memcpy(prefix_, data, kPrefixSize);
      value_.data = data;
    }
  }
  template <typename Str>
  static std::vector<StringView> makeVector(const std::vector<Str>& vec) {
    std::vector<StringView> vals(vec.size());
    for (size_t i = 0; i < vals.size(); i++) {
      vals[i] = StringView(vec[i]);
    }
    return vals;
  }

  static StringView makeInline(std::string str) {
    // VELOX_DCHECK(isInline(str.size()));
    return StringView{str};
  }

  // Making StringView implicitly constructible/convertible from char* and
  // string literals, in order to allow for a more flexible API and optional
  // interoperability. E.g:
  //
  //   StringView sv = "literal";
  //   std::optional<StringView> osv = "literal";
  //
  /* implicit */ StringView(const char* data) : StringView(data, strlen(data)) {}

  StringView(const std::string& value) : StringView(value.data(), value.size()) {}
  explicit StringView(std::string&& value) = delete;

  explicit StringView(std::string_view value) : StringView(value.data(), value.size()) {}

  bool isInline() const { return isInline(size_); }

  static constexpr bool isInline(uint32_t size) { return size <= kInlineSize; }

  const char* data() && = delete;
  const char* data() const& { return isInline() ? prefix_ : value_.data; }

  size_t size() const { return size_; }

  size_t capacity() const { return size_; }

  // friend std::ostream& operator<<(std::ostream& os, const StringView& stringView) {
  //   os.write(stringView.data(), stringView.size());
  //   return os;
  // }

  bool operator==(const StringView& other) const {
    // Compare lengths and first 4 characters.
    if (sizeAndPrefixAsInt64() != other.sizeAndPrefixAsInt64()) {
      return false;
    }
    if (isInline()) {
      // The inline part is zeroed at construction, so we can compare
      // a word at a time if data extends past 'prefix_'.
      return size_ <= kPrefixSize || inlinedAsInt64() == other.inlinedAsInt64();
    }
    // Sizes are equal and this is not inline, therefore both are out
    // of line and have kPrefixSize first in common.
    return memcmp(value_.data + kPrefixSize, other.value_.data + kPrefixSize, size_ - kPrefixSize) == 0;
  }

  bool operator!=(const StringView& other) const { return !(*this == other); }

  // Returns 0, if this == other
  //       < 0, if this < other
  //       > 0, if this > other
  int32_t compare(const StringView& other) const {
    if (prefixAsInt() != other.prefixAsInt()) {
      // The result is decided on prefix. The shorter will be less
      // because the prefix is padded with zeros.
      return memcmp(prefix_, other.prefix_, kPrefixSize);
    }
    int32_t size = std::min(size_, other.size_) - kPrefixSize;
    if (size <= 0) {
      // One ends within the prefix.
      return size_ - other.size_;
    }
    if (size <= static_cast<int32_t>(kInlineSize) && isInline() && other.isInline()) {
      int32_t result = memcmp(value_.inlined, other.value_.inlined, size);
      return (result != 0) ? result : size_ - other.size_;
    }
    int32_t result = memcmp(data() + kPrefixSize, other.data() + kPrefixSize, size);
    return (result != 0) ? result : size_ - other.size_;
  }

  bool operator<(const StringView& other) const { return compare(other) < 0; }

  bool operator<=(const StringView& other) const { return compare(other) <= 0; }

  bool operator>(const StringView& other) const { return compare(other) > 0; }

  bool operator>=(const StringView& other) const { return compare(other) >= 0; }

  // StringView& operator=(const std::string& str) {
  //   StringView other(str);
  //   mempcpy(this, &other, sizeof(StringView));
  //   return *this;
  // }

  operator std::string() const { return std::string(data(), size()); }

  std::string str() const { return *this; }

  std::string getString() const { return *this; }

  std::string materialize() const { return *this; }

  operator std::string_view() && = delete;
  explicit operator std::string_view() const& { return std::string_view(data(), size()); }
  absl::string_view get_absl_string_view() const { return absl::string_view(data(), size()); }
  std::string_view get_string_view() const { return std::string_view(data(), size()); }

  const char* begin() && = delete;
  const char* begin() const& { return data(); }

  const char* end() && = delete;
  const char* end() const& { return data() + size(); }

  bool empty() const { return size() == 0; }

  /// Searches for 'key == strings[i]'for i >= 0 < numStrings. If
  /// 'indices' is given. searches for 'key ==
  /// strings[indices[i]]. Returns the first i for which the strings
  /// match or -1 if no match is found. Uses SIMD to accelerate the
  /// search. Accesses StringView bodies in 32 byte vectors, thus
  /// expects up to 31 bytes of addressable padding after out of
  /// line strings. This is the case for velox Buffers.
  static int32_t linearSearch(StringView key, const StringView* strings, const int32_t* indices, int32_t numStrings);

  static bool contains(StringView s, StringView part) {
    return absl::StrContains(s.get_absl_string_view(), part.get_absl_string_view());
  }
  static bool starts_with(StringView s, StringView part) {
    return absl::StartsWith(s.get_absl_string_view(), part.get_absl_string_view());
  }
  static bool ends_with(StringView s, StringView part) {
    return absl::EndsWith(s.get_absl_string_view(), part.get_absl_string_view());
  }
  static bool contains_ignore_case(StringView s, StringView part) {
    return absl::StrContainsIgnoreCase(s.get_absl_string_view(), part.get_absl_string_view());
  }
  static bool starts_with_ignore_case(StringView s, StringView part) {
    return absl::StartsWith(s.get_absl_string_view(), part.get_absl_string_view());
  }
  static bool ends_with_ignore_case(StringView s, StringView part) {
    return absl::EndsWith(s.get_absl_string_view(), part.get_absl_string_view());
  }

 private:
  inline int64_t sizeAndPrefixAsInt64() const { return reinterpret_cast<const int64_t*>(this)[0]; }

  inline int64_t inlinedAsInt64() const { return reinterpret_cast<const int64_t*>(this)[1]; }

  int32_t prefixAsInt() const { return *reinterpret_cast<const int32_t*>(&prefix_); }

  // We rely on all members being laid out top to bottom . C++
  // guarantees this.
  uint32_t size_;
  char prefix_[4];
  union {
    char inlined[8];
    const char* data;
  } value_;
};

// This creates a user-defined literal for StringView. You can use it as:
//
//   auto myStringView = "my string"_sv;
//   auto vec = {"str1"_sv, "str2"_sv};
inline StringView operator"" _sv(const char* str, size_t len) { return StringView(str, len); }

bool compare_string_view(uint32_t op, StringView left, StringView right);

}  // namespace rapidudf

namespace fmt {
template <>
struct formatter<rapidudf::StringView> : private formatter<string_view> {
  using formatter<string_view>::parse;

  template <typename Context>
  typename Context::iterator format(rapidudf::StringView s, Context& ctx) const {
    return formatter<string_view>::format(string_view{s.data(), s.size()}, ctx);
  }
};

}  // namespace fmt

namespace std {
template <>
struct hash<::rapidudf::StringView> {
  size_t operator()(const ::rapidudf::StringView view) const {
    return std::hash<std::string_view>{}({view.data(), view.size()});
  }
};
template <>
struct is_trivially_destructible<::rapidudf::StringView> {
  static constexpr bool value = true;
};
}  // namespace std
