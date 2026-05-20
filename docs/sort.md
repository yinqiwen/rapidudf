# SIMD Key-Value Sort 实现方案

## 目标

在 rapidudf 中实现类似 [x86-simd-sort](https://github.com/numpy/x86-simd-sort) 的 `keyvalue_qsort`：

```cpp
void keyvalue_qsort(T1* key, T2* val, size_t size, bool hasnan, bool descending);
```

使用 [highway](https://github.com/google/highway) 的 SIMD 原语实现跨平台支持（SSE/AVX/NEON/SVE/RVV），不依赖 highway 的 VQSort 实现。

## 两个参考实现的对比

| 维度 | highway VQSort | x86-simd-sort |
|---|---|---|
| 数据布局 | AoS — KV 打包成单个数组 (`K32V32`/`K64V64`) | SoA — key 和 value 是两个独立数组 |
| 类型灵活性 | 固定配对: `u32+u32` 或 `u64+u64` | `T1` 和 `T2` 可以不同类型 |
| NaN 处理 | 通过 trait 系统处理 | 显式 `hasnan` 参数 |
| 平台支持 | 跨平台 (SSE/AVX/NEON/SVE/RVV) | 仅 x86 (AVX2/AVX-512) |
| 分区指令 | `CompressStore` (单侧) | `double_compressstore` (双侧) |

## 为什么不直接用 highway VQSort 包装

highway VQSort 的 KV 排序要求将 key/value 打包成 `K32V32`（64-bit，key 在高 32 位）或 `K64V64`（128-bit，key 在高 64 位）：

```cpp
struct alignas(8) K32V32 { uint32_t value; uint32_t key; };
struct alignas(16) K64V64 { uint64_t value; uint64_t key; };
```

用 VQSort 实现 `keyvalue_qsort` 需要 pack → sort → unpack：

```cpp
// 需要额外 N * sizeof(K32V32) 或 N * sizeof(K64V64) 的临时内存
auto* packed = new K32V32[n];
for (size_t i = 0; i < n; i++)
    packed[i] = {BitCast<uint32_t>(val[i]), BitCast<uint32_t>(key[i])};
VQSort(packed, n, SortAscending());
for (size_t i = 0; i < n; i++) { ... unpack ... }
delete[] packed;
```

额外内存开销 `N * 8` 或 `N * 16` 字节。虽然相对于排序本身的时间可以接受，但可以完全避免。

## 基于 highway 原语的零额外内存方案

### 核心思路

直接翻译 x86-simd-sort 的双数组算法，用 highway 原语替代 x86 专属指令。

### highway 提供的关键原语

| x86-simd-sort 操作 | highway 对应 | 平台覆盖 |
|---|---|---|
| `min`/`max` | `Min`/`Max` | 全平台 |
| compressstore (单侧) | `CompressStore` | AVX-512 原生，其他平台有 emulate |
| `double_compressstore` | **无直接对应**（见下文） | — |
| mask 操作 | `Lt`/`Eq`/`IfThenElse`/`Not` | 全平台 |
| permute for values | `TwoTablesLookupLanes` | 全平台 |
| `fpclass` (NaN 检测) | `IsNaN` | 全平台 |

### `double_compressstore` 的替代方案

x86-simd-sort 的 `double_compressstore` 一条指令把 `≤ pivot` 写左侧、`> pivot` 写右侧。highway 没有直接对应，有三种替代：

**方案 A：两次 CompressStore（通用，全平台）**

```cpp
auto mask = Lt(keys_vec, pivot_vec);
// left: compress + store
size_t n_match = CompressStore(keys_vec, mask, d_key, keys + writeL);
CompressStore(vals_vec, mask, d_val, vals + writeL);
writeL += n_match;
// right: CompressNot + store
auto k_gt = CompressNot(keys_vec, mask);
auto v_gt = CompressNot(vals_vec, mask);
size_t n_gt = N - n_match;
writeR -= n_gt;
StoreU(k_gt, keys + writeR);
StoreU(v_gt, vals + writeR);
```

代价：数据被读两次。在现代乱序 CPU 上，两次 pass 的开销相对于排序本身很小（~10-15% 分区时间）。

**方案 B：CompressIsPartition（SVE/Scalar 等平台）**

当 `CompressIsPartition<T>::value == true` 时，`Compress(v, mask)` 的结果中低 lane = mask-true 元素，高 lane = mask-false 元素。一次 Compress 即可获取两侧数据。

适用平台：Scalar（始终）、SVE_256/SVE2_128（8-byte 类型）。

**方案 C：平台特化（AVX-512）**

在 `HWY_AVX3` 路径中直接用 `vpcompressd` 手动模拟 `double_compressstore`。

### 类型宽度不同时的处理

当 `sizeof(T1) != sizeof(T2)` 时（如 int64 key + int32 value），使用 half-vector 策略：

```cpp
// key: full-width (8 lanes on AVX-512)
using DK = ScalableTag<uint64_t>;
// value: half-width (same lane count, narrower regs)
using DV = Half<ScalableTag<uint64_t>>;
```

mask 从 key 类型到 value 类型的转换，通过 `CompressStore` 返回的 count 或 index-based gather 实现。

## 算法总览

```
KVSort(keys, vals, N, hasnan, descending)
│
├─ hasnan? → move_nans_to_end(keys, vals, N)  // IsNaN + scalar swap
│
├─ descending? → sort ascending, then reverse both arrays
│
└─ kvsort_impl(keys, vals, N)
    │
    ├─ N ≤ kBaseCaseSize?
    │   └─ kv_sort_network(keys, vals, N)  // bitonic sort, 全寄存器
    │
    ├─ max_iters == 0?
    │   └─ kv_heap_sort(keys, vals, N)     // 退化为堆排序
    │
    └─ kv_quicksort(keys, vals, N, max_iters)
        ├─ pivot = median_of_keys(keys, N)
        ├─ kv_partition(keys, vals, N, pivot)  // 双数组分区
        └─ 递归 sort 左半 + 右半
```

## 关键实现细节

### 1. COEX — 排序网络核心（条件交换）

```cpp
template <class DK, class DV>
HWY_INLINE void COEX(Vec<DK> k1, Vec<DK>& k2,
                      Vec<DV>& v1, Vec<DV>& v2) {
    auto k_min = Min(k1, k2);
    auto k_max = Max(k1, k2);
    auto mask = Eq(k_min, k1);
    auto v1_new = IfThenElse(mask, v1, v2);
    auto v2_new = IfThenElse(mask, v2, v1);
    k1 = k_min; k2 = k_max;
    v1 = v1_new; v2 = v2_new;
}
```

key 的比较结果直接作为 mask 搬运 value，不需要 pack。排序网络（bitonic sort）的每一步都同时操作两个数组。

### 2. 双数组分区

```cpp
template <class DK, class DV>
size_t kv_partition(DK d_key, DV d_val,
                    T_Key* keys, T_Val* vals, size_t n,
                    T_Key pivot) {
    size_t writeL = 0, writeR = n;
    size_t i = 0;
    const size_t N = Lanes(d_key);

    for (; i + N <= n; i += N) {
        auto k = LoadU(d_key, keys + i);
        auto v = LoadU(d_val, vals + i);
        auto mask = Lt(k, Set(d_key, pivot));

        size_t n_match = CompressStore(k, mask, d_key, keys + writeL);
        CompressStore(v, mask, d_val, vals + writeL);
        writeL += n_match;

        auto k_gt = CompressNot(k, mask);
        auto v_gt = CompressNot(v, mask);
        size_t n_gt = N - n_match;
        writeR -= n_gt;
        StoreU(k_gt, keys + writeR);
        StoreU(v_gt, vals + writeR);
    }
    // scalar tail for remaining elements
    // ...
}
```

### 3. NaN 处理

```cpp
// 批量检测 NaN，标量 swap 到末尾
size_t move_nans_to_end(T_Key* keys, T_Val* vals, size_t n) {
    size_t left = 0, right = n - 1;
    while (left <= right) {
        if (IsNaN(keys[right])) { right--; continue; }
        if (IsNaN(keys[left])) {
            std::swap(keys[left], keys[right]);
            std::swap(vals[left], vals[right]);
            right--;
        }
        left++;
    }
    return right + 1;  // 有效长度
}
```

### 4. descending 处理

sort 完成后反转两个数组：

```cpp
std::reverse(keys, keys + n);
std::reverse(vals, vals + n);
```

O(n) 开销，不影响 O(n log n) 的排序。

## 代码量估算

| 模块 | 代码量 |
|---|---|
| COEX + bitonic sorting network | ~300 行 |
| kv_partition (双数组分区) | ~200 行 |
| kv_heap_sort (堆排序) | ~100 行 |
| NaN handling | ~50 行 |
| 入口 + 类型分发 + 模板特化 | ~100 行 |
| 测试 | ~300 行 |
| **合计** | ~800-1200 行（不含测试 ~650-850 行） |

## 性能预估

| 场景 | 相对 x86-simd-sort |
|---|---|
| AVX-512 分区（两次 CompressStore） | 慢 ~10-15% |
| SVE（CompressIsPartition=true） | 持平 |
| NEON（无原生 compress） | 较慢，依赖 emulate |
| 排序网络 (COEX) | 等价 |
| 整体 on AVX-512 | 慢 ~10-20% |
| 整体 on SVE/NEON/RVV | N/A（x86-simd-sort 不支持） |

## 支持的类型

与 x86-simd-sort 一致：

- Key: `int32_t`, `uint32_t`, `float`, `int64_t`, `uint64_t`, `double`
- Value: `int32_t`, `uint32_t`, `float`, `int64_t`, `uint64_t`, `double`
- 约束: `sizeof(T1) == sizeof(T2)`（32-bit key 配 32-bit value，64-bit 配 64-bit）
- 不支持 16-bit 类型（与 x86-simd-sort 一致）

## 后续优化方向

1. **OpenMP 并行**：大数组（>10000 元素）时用 `#pragma omp task` 并行递归子问题
2. **AVX-512 double_compressstore 特化**：在 `HWY_AVX3` 路径中手写汇编模拟双侧压缩
3. **Partial sort / Select**：实现 `keyvalue_partial_sort` 和 `keyvalue_select`
