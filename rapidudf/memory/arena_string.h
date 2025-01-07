/*
 * Copyright (c) 2025 qiyingwang <qiyingwang@tencent.com>. All rights reserved.
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
#include <string>
#include "rapidudf/memory/arena.h"
namespace rapidudf {
namespace arena {
template <typename C, typename CH = ::std::char_traits<C>, typename R = ThreadCachedArenaAllocator<C>>
class BasicString : public ::std::basic_string<C, CH, R> {
 private:
  using Base = ::std::basic_string<C, CH, R>;

 public:
  using typename Base::const_pointer;
  using typename Base::value_type;

  using Base::Base;
  using Base::operator=;

  using Base::append;
  using Base::clear;
  using Base::data;
  using Base::get_allocator;
  using Base::size;
  inline BasicString(const typename Base::allocator_type& allocator) noexcept : Base(allocator) {}
  inline BasicString(BasicString&& other, const typename Base::allocator_type& allocator) noexcept : Base(allocator) {
    operator=(::std::move(other));
  }
  inline BasicString(const BasicString& other) noexcept : Base(other.get_allocator()) { operator=(other); }

  template <typename A>
  inline BasicString(const ::std::basic_string<C, CH, A>& other,
                     const typename Base::allocator_type& allocator) noexcept
      : Base(other.begin(), other.end(), allocator) {}

  template <typename A>
  inline BasicString& operator=(const ::std::basic_string<C, ::std::char_traits<C>, A>& other) noexcept {
    static_cast<Base*>(this)->assign(other.c_str(), other.size());
    return *this;
  }

  inline BasicString& operator=(const BasicString& other) noexcept {
    *static_cast<Base*>(this) = other;
    return *this;
  }

  inline BasicString& operator=(BasicString&& other) noexcept {
    if (get_allocator() == other.get_allocator()) {
      swap(other);
    } else {
      *this = other;
    }
    return *this;
  }

  inline void swap(BasicString& other) noexcept {
    assert(get_allocator() == other.get_allocator() && "can not swap string with different allocator");
    Base::swap(other);
  }
};

using String = BasicString<char>;
}  // namespace arena
}  // namespace rapidudf

namespace std {
template <typename CH, typename R>
struct is_trivially_destructible<::rapidudf::arena::BasicString<CH, ::std::char_traits<CH>, R>> {
  static constexpr bool value = true;
};

}  // namespace std