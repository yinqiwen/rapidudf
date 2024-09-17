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
#include "rapidudf/jit/llvm/jit_cache.h"
#include "rapidudf/jit/llvm/jit_session.h"
namespace rapidudf {
namespace llvm {
size_t JitCompilerCache::RemoveExpiredCache(std::chrono::seconds ttl_secs) {
  auto& cache_map = GetCache();
  auto& cache_mutex = GetCacheMutex();
  std::vector<std::string> expired_keys;
  {
    std::lock_guard<std::mutex> guard(cache_mutex);
    for (auto& [key, item] : cache_map) {
      auto duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() -
                                                                       item.latest_visit_time);
      if (duration > ttl_secs) {
        expired_keys.emplace_back(key);
      }
    }
  }
  if (!expired_keys.empty()) {
    std::lock_guard<std::mutex> guard(cache_mutex);
    for (auto& key : expired_keys) {
      cache_map.erase(key);
      RUDF_INFO("Remove expired compiled source:{}", key);
    }
  }
  return expired_keys.size();
}
}  // namespace llvm
}  // namespace rapidudf