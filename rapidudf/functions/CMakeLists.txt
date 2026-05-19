add_subdirectory(simd)

rapidudf_add_sources(
  rapidudf/functions/functions.cc
  rapidudf/functions/json.cc
  rapidudf/functions/math.cc
  rapidudf/functions/simd_table.cc
  rapidudf/functions/simd_vector.cc
  rapidudf/functions/stl_maps.cc
  rapidudf/functions/stl_sets.cc
  rapidudf/functions/stl_vector.cc
  rapidudf/functions/strings.cc
  rapidudf/functions/time.cc
)
