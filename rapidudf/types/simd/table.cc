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

#include "rapidudf/types/simd/table.h"
#include <cstring>
#include "rapidudf/functions/simd/vector.h"
#include "rapidudf/log/log.h"
#include "rapidudf/meta/dtype.h"
#include "rapidudf/meta/dtype_enums.h"
#include "rapidudf/types/pointer.h"
#include "rapidudf/types/simd/vector.h"
#include "rapidudf/types/string_view.h"

namespace rapidudf {
namespace simd {

TableSchema* TableSchema::GetMutable(const std::string& name) {
  DynObjectSchema* s = DynObjectSchema::GetMutable(name);
  return reinterpret_cast<TableSchema*>(s);
}

const TableSchema* TableSchema::Get(const std::string& name) { return GetMutable(name); }

const TableSchema* TableSchema::GetOrCreate(const std::string& name, InitFunc&& init) {
  const DynObjectSchema* s = DynObjectSchema::GetOrCreate(
      name,
      [&](DynObjectSchema* s) {
        TableSchema* table_schema = reinterpret_cast<TableSchema*>(s);
        init(table_schema);
        Flags flags(true);
        table_schema->SetFlags(flags);
      },
      sizeof(Table));
  return reinterpret_cast<const TableSchema*>(s);
}

typename Table::SmartPtr TableSchema::NewTable(Context& ctx) const {
  uint8_t* bytes = new uint8_t[ByteSize()];
  memset(bytes, 0, ByteSize());
  try {
    new (bytes) Table(ctx, this);
  } catch (...) {
    throw;
  }
  Table::SmartPtr p(reinterpret_cast<Table*>(bytes));
  return p;
}

Table* Table::Clone() {
  uint8_t* new_table = reinterpret_cast<uint8_t*>(ctx_.ArenaAllocate(schema_->ByteSize()));
  uint32_t bytes = schema_->ByteSize();
  memcpy(new_table, this, bytes);
  return reinterpret_cast<Table*>(new_table);
}
Vector<int32_t> Table::GetIndices() {
  if (indices_.Size() == 0) {
    auto* first_column = reinterpret_cast<VectorData*>(this + 1);
    indices_ = functions::simd_vector_iota<int32_t>(ctx_, 0, first_column->Size());
  }
  auto* p = ctx_.ArenaAllocate(sizeof(int32_t) * indices_.Size());
  memcpy(p, indices_.Data(), sizeof(int32_t) * indices_.Size());
  VectorData vdata(p, indices_.Size(), sizeof(int32_t) * indices_.Size());
  vdata.SetReadonly(false);
  return Vector<int32_t>(vdata);
}

void Table::SetColumn(uint32_t offset, VectorData vec) {
  uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this) + offset;
  memcpy(vec_ptr, &vec, sizeof(vec));
}

size_t Table::Size() const { return schema_->FieldCount(); }

Table* Table::Filter(Vector<Bit> bits) {
  Table* this_table = this;
  Table* new_table = Clone();
  schema_->VisitField([&](const std::string& name, const DType& dtype, uint32_t offset) {
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this_table) + offset;
    VectorData new_vec;
    switch (dtype.GetFundamentalType()) {
      case DATA_BIT: {
        new_vec =
            functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<Bit>*>(vec_ptr), bits).GetVectorData();
        break;
      }
      case DATA_U8: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<uint8_t>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_U16: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<uint16_t>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_U32: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<uint32_t>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_U64: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<uint64_t>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_I8: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<int8_t>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_I16: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<int16_t>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_I32: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<int32_t>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_I64: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<int64_t>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_F32: {
        new_vec =
            functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<float>*>(vec_ptr), bits).GetVectorData();
        break;
      }
      case DATA_F64: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<double>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
      }
      case DATA_STRING_VIEW: {
        new_vec = functions::simd_vector_filter(ctx_, *reinterpret_cast<simd::Vector<StringView>*>(vec_ptr), bits)
                      .GetVectorData();
        break;
        break;
      }
      default: {
        RUDF_ERROR("Unsupported dtype:{} for column:{}", dtype, name);
        return;
      }
    }
    new_table->SetColumn(offset, new_vec);
  });
  return new_table;
}
Table* Table::Take(uint32_t k) {
  Table* this_table = this;
  Table* new_table = Clone();
  schema_->VisitField([&](const std::string& name, const DType& dtype, uint32_t offset) {
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this_table) + offset;
    VectorData new_vec = *(reinterpret_cast<VectorData*>(vec_ptr));
    new_vec.SetSize(k);
    new_table->SetColumn(offset, new_vec);
  });
  return new_table;
}

template <typename T>
Table* Table::OrderBy(Vector<T> by, bool descending) {
  auto indices = GetIndices();
  functions::simd_vector_sort_key_value(ctx_, by, indices, descending);
  Table* this_table = this;
  Table* new_table = Clone();
  schema_->VisitField([&](const std::string& name, const DType& dtype, uint32_t offset) {
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this_table) + offset;
    VectorData new_vec = *(reinterpret_cast<VectorData*>(vec_ptr));
    if (new_vec.Data() == by.GetVectorData().Data()) {
      new_table->SetColumn(offset, new_vec);
      return;
    }
    switch (dtype.GetFundamentalType()) {
      case DATA_BIT: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<Bit>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_U8: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<uint8_t>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_U16: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<uint16_t>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_U32: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<uint32_t>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_U64: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<uint64_t>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_I8: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<int8_t>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_I16: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<int16_t>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_I32: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<int32_t>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_I64: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<int64_t>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_F32: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<float>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_F64: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<double>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_STRING_VIEW: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<StringView>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_POINTER: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<Pointer>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      default: {
        RUDF_ERROR("Unsupported dtype:{} for column:{}", dtype, name);
        return;
      }
    }
    new_table->SetColumn(offset, new_vec);
  });
  return new_table;
}
template <typename T>
Table* Table::Topk(Vector<T> by, uint32_t k, bool descending) {
  auto indices = GetIndices();
  functions::simd_vector_topk_key_value(ctx_, by, indices, k, descending);
  indices = indices.Resize(k);
  Table* this_table = this;
  Table* new_table = Clone();
  schema_->VisitField([&](const std::string& name, const DType& dtype, uint32_t offset) {
    uint8_t* vec_ptr = reinterpret_cast<uint8_t*>(this_table) + offset;
    VectorData new_vec = *(reinterpret_cast<VectorData*>(vec_ptr));
    if (new_vec.Data() == by.GetVectorData().Data()) {
      new_vec.SetSize(k);
      new_table->SetColumn(offset, new_vec);
      return;
    }
    switch (dtype.GetFundamentalType()) {
      case DATA_BIT: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<Bit>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_U8: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<uint8_t>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_U16: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<uint16_t>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_U32: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<uint32_t>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_U64: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<uint64_t>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_I8: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<int8_t>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_I16: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<int16_t>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_I32: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<int32_t>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_I64: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<int64_t>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_F32: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<float>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_F64: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<double>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_STRING_VIEW: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<StringView>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      case DATA_POINTER: {
        new_vec = functions::simd_vector_gather(ctx_, *reinterpret_cast<simd::Vector<Pointer>*>(vec_ptr), indices)
                      .GetVectorData();
        break;
      }
      default: {
        RUDF_ERROR("Unsupported dtype:{} for column:{}", dtype, name);
        return;
      }
    }
    new_table->SetColumn(offset, new_vec);
  });
  return new_table;
}

template Table* Table::OrderBy<uint32_t>(Vector<uint32_t> by, bool descending);
template Table* Table::OrderBy<int32_t>(Vector<int32_t> by, bool descending);
template Table* Table::OrderBy<uint64_t>(Vector<uint64_t> by, bool descending);
template Table* Table::OrderBy<int64_t>(Vector<int64_t> by, bool descending);
template Table* Table::OrderBy<float>(Vector<float> by, bool descending);
template Table* Table::OrderBy<double>(Vector<double> by, bool descending);

template Table* Table::Topk<uint32_t>(Vector<uint32_t> by, uint32_t k, bool descending);
template Table* Table::Topk<int32_t>(Vector<int32_t> by, uint32_t k, bool descending);
template Table* Table::Topk<uint64_t>(Vector<uint64_t> by, uint32_t k, bool descending);
template Table* Table::Topk<int64_t>(Vector<int64_t> by, uint32_t k, bool descending);
template Table* Table::Topk<float>(Vector<float> by, uint32_t k, bool descending);
template Table* Table::Topk<double>(Vector<double> by, uint32_t k, bool descending);

}  // namespace simd
}  // namespace rapidudf