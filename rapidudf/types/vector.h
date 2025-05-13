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
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>
#include "absl/status/statusor.h"
#include "arrow/api.h"
#include "arrow/type_traits.h"
#include "rapidudf/common/arrow_helper.h"
#include "rapidudf/log/log.h"
namespace rapidudf {

template <typename T>
struct VectorBase {
  static size_t GetSize(const T* t) { return t->Size(); }
  static auto GetData(const T* t, size_t offset) { return t->Data(offset); }
};

template <typename T>
class Vector : public VectorBase<Vector<T>> {
 public:
  using value_type = typename VectorElementTypeTraits<T>::DataType;
  using value_iterator_func = std::function<std::optional<value_type>()>;

  static absl::StatusOr<Vector> Wrap(arrow::MemoryPool* pool, const T* data, size_t size, size_t capacity = 0,
                                     bool copy = false);
  static absl::StatusOr<Vector> Make(arrow::MemoryPool* pool, const value_iterator_func& iter, size_t length);
  static absl::StatusOr<Vector> Make(arrow::MemoryPool* pool, size_t size);
  static absl::StatusOr<Vector> Wrap(arrow::MemoryPool* pool, const std::vector<T>& data, bool copy = false) {
    return Wrap(pool, data.data(), data.size(), data.capacity(), copy);
  }

  static absl::StatusOr<Vector> Make(arrow::MemoryPool* pool, const T* data, size_t size) {
    return Wrap(pool, data, size, size, true);
  }

  static absl::StatusOr<Vector> Make(arrow::MemoryPool* pool, const std::vector<T>& data) {
    if constexpr (std::is_same_v<T, Bit> || std::is_same_v<T, bool>) {
      auto iter = data.begin();
      return Make(
          pool,
          [&]() -> std::optional<bool> {
            if (iter == data.end()) {
              return {};
            }
            auto v = *iter;
            iter++;
            return v;
          },
          data.size());
    } else if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, std::string_view>) {
      auto iter = data.begin();
      return Make(
          pool,
          [&]() -> std::optional<value_type> {
            if (iter == data.end()) {
              return {};
            }
            std::string_view v = *iter;
            iter++;
            return v;
          },
          data.size());
    } else {
      return Wrap(pool, data.data(), data.size(), data.capacity(), true);
    }
  }

  size_t Size() const { return data_->length(); }
  auto Data(size_t offset = 0) const {
    if constexpr (std::is_same_v<T, Bit> || std::is_same_v<T, bool>) {
      return data_->data()->GetValues<uint8_t>(1, offset / 8);
    } else {
      auto actual_array = static_cast<typename arrow::CTypeTraits<T>::ArrayType*>(data_.get());
      return actual_array->raw_values() + offset;
    }
  }
  auto MutableData(size_t offset = 0) {
    if constexpr (std::is_same_v<T, Bit> || std::is_same_v<T, bool>) {
      return data_->data()->GetMutableValues<uint8_t>(1, offset / 8);
    } else {
      auto actual_array = static_cast<typename arrow::CTypeTraits<T>::ArrayType*>(data_.get());
      return const_cast<T*>(actual_array->raw_values() + offset);
    }
  }

  auto Value(size_t offset) const {
    auto actual_array = static_cast<typename arrow::CTypeTraits<T>::ArrayType*>(data_.get());
    return actual_array->Value(offset);
  }

  Vector<T> Slice(int64_t offset, int64_t length) const;

 private:
  Vector(std::shared_ptr<arrow::Array> p) : data_(p) {}
  std::shared_ptr<arrow::Array> data_;
};
template <typename T>
absl::StatusOr<Vector<T>> Vector<T>::Wrap(arrow::MemoryPool* pool, const T* data, size_t size, size_t capacity,
                                          bool copy) {
  if (capacity == 0) {
    capacity = size;
  }
  if constexpr (std::is_same_v<T, Bit> || std::is_same_v<T, bool>) {
    static_assert(sizeof(T) == -1, "unsupported bool/bit type to wrap");
  }

  std::shared_ptr<arrow::Array> array;
  if (copy) {
    std::unique_ptr<arrow::ArrayBuilder> builder;
    auto status = arrow::MakeBuilder(pool, get_arrow_dtype<T>(), &builder);
    if (!status.ok()) {
      RUDF_LOG_RETURN_FMT_ERROR("Builder create failed:{}", status.ToString());
    }
    auto actual_builder = static_cast<typename arrow::CTypeTraits<T>::BuilderType*>(builder.get());
    if constexpr (std::is_pointer_v<T>) {
      const uint64_t* int_data = reinterpret_cast<const uint64_t*>(data);
      status = actual_builder->AppendValues(int_data, size);
    } else {
      status = actual_builder->AppendValues(data, size);
    }

    if (!status.ok()) {
      RUDF_LOG_RETURN_FMT_ERROR("Builder append values failed:{}", status.ToString());
    }
    status = actual_builder->Finish(&array);
    if (!status.ok()) {
      RUDF_LOG_RETURN_FMT_ERROR("Builder finish failed:{}", status.ToString());
    }
  } else {
    typename arrow::CTypeTraits<T>::ArrowType arrow_type;
    size_t byte_length = 0;
    if (arrow_type.byte_width() > 0) {
      byte_length = size * arrow_type.byte_width();
    } else {
      byte_length = size / 8;
      if (size % 8 > 0) {
        byte_length += 1;
      }
    }
    std::shared_ptr<arrow::Buffer> buffer = arrow::Buffer::Wrap(data, byte_length);
    std::shared_ptr<arrow::ArrayData> array_data =
        arrow::ArrayData::Make(get_arrow_dtype<T>(), size, {nullptr, buffer});
    array = arrow::MakeArray(array_data);
  }
  return Vector<T>(array);
}
template <typename T>
absl::StatusOr<Vector<T>> Vector<T>::Make(arrow::MemoryPool* pool, const value_iterator_func& iter, size_t length) {
  std::unique_ptr<arrow::ArrayBuilder> builder;
  auto status = arrow::MakeBuilder(pool, get_arrow_dtype<T>(), &builder);
  if (!status.ok()) {
    RUDF_LOG_RETURN_FMT_ERROR("Builder create failed:{}", status.ToString());
  }
  auto actual_builder = static_cast<typename arrow::CTypeTraits<T>::BuilderType*>(builder.get());
  status = actual_builder->Reserve(length);
  if (!status.ok()) {
    RUDF_LOG_RETURN_FMT_ERROR("Builder reserve:{} failed:{}", length, status.ToString());
  }
  while (1) {
    auto val = iter();
    if (val.has_value()) {
      status = actual_builder->Append(val.value());
      if (!status.ok()) {
        RUDF_LOG_RETURN_FMT_ERROR("Builder append  failed:{}", status.ToString());
      }
    } else {
      break;
    }
  }
  std::shared_ptr<arrow::Array> array_data;
  status = actual_builder->Finish(&array_data);
  if (!status.ok()) {
    RUDF_LOG_RETURN_FMT_ERROR("Builder finish failed:{}", status.ToString());
  }
  return Vector<T>(array_data);
}
template <typename T>
absl::StatusOr<Vector<T>> Vector<T>::Make(arrow::MemoryPool* pool, size_t size) {
  std::unique_ptr<arrow::ArrayBuilder> builder;
  auto status = arrow::MakeBuilder(pool, get_arrow_dtype<T>(), &builder);
  if (!status.ok()) {
    RUDF_LOG_RETURN_FMT_ERROR("Builder create failed:{}", status.ToString());
  }
  auto actual_builder = static_cast<typename arrow::CTypeTraits<T>::BuilderType*>(builder.get());
  status = actual_builder->AppendEmptyValues(size);
  if (!status.ok()) {
    RUDF_LOG_RETURN_FMT_ERROR("Builder append empty values failed:{}", status.ToString());
  }
  std::shared_ptr<arrow::Array> array_data;
  status = actual_builder->Finish(&array_data);
  if (!status.ok()) {
    RUDF_LOG_RETURN_FMT_ERROR("Builder finish failed:{}", status.ToString());
  }
  return Vector<T>(array_data);
}
template <typename T>
Vector<T> Vector<T>::Slice(int64_t offset, int64_t length) const {
  auto new_data = data_->Slice(offset, length);
  return Vector<T>(new_data);
}

}  // namespace rapidudf