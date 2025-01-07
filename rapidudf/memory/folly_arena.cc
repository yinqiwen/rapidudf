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
#include "rapidudf/memory/folly_arena.h"
namespace rapidudf {
namespace folly {

void* Arena::allocateSlow(size_t size) {
  char* start;
  size_t block_body_size = std::max(size, minBlockSize());
  if (std::numeric_limits<size_t>::max() - sizeof(Block) < block_body_size) {
    throw std::bad_alloc();
  }

  size_t allocSize = block_body_size + sizeof(Block);
  if (sizeLimit_ != kNoSizeLimit && allocSize > sizeLimit_ - totalAllocatedSize_) {
    // throw_exception<std::bad_alloc>();
    throw std::bad_alloc();
  }

  if (size > minBlockSize()) {
    // Allocate a large block for this chunk only; don't change ptr_ and end_,
    // let them point into a normal block (or none, if they're null)
    allocSize = sizeof(LargeBlock) + size;
    // void* mem = AllocTraits::allocate(alloc(), allocSize);
    char* mem = new char[allocSize];
    auto blk = new (mem) LargeBlock(allocSize);
    start = align(blk->start());
    largeBlocks_.push_back(*blk);
  } else {
    // Allocate a normal sized block and carve out size bytes from it
    // Will allocate more than size bytes if convenient
    // (via ArenaAllocatorTraits::goodSize()) as we'll try to pack small
    // allocations in this block.
    allocSize = blockGoodAllocSize();
    // void* mem = AllocTraits::allocate(alloc(), allocSize);
    char* mem = new char[allocSize];
    auto blk = new (mem) Block();
    start = align(blk->start());
    blocks_.push_back(*blk);
    currentBlock_ = blocks_.last();
    ptr_ = start + size;
    end_ = start + allocSize - sizeof(Block);
    assert(ptr_ <= end_);
  }

  totalAllocatedSize_ += allocSize;
  return start;
}

void Arena::freeMemory(char* p) { delete[] p; }

}  // namespace folly
}  // namespace rapidudf