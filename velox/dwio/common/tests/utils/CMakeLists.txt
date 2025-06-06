# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

velox_add_library(
  velox_dwio_common_test_utils
  BatchMaker.cpp
  DataFiles.cpp
  DataSetBuilder.cpp
  FilterGenerator.cpp
  UnitLoaderTestTools.cpp
  E2EFilterTestBase.cpp)

velox_link_libraries(
  velox_dwio_common_test_utils
  Folly::folly
  fmt::fmt
  glog::glog
  GTest::gtest
  velox_dwio_common
  velox_dwio_common_exception
  velox_exception
  velox_functions_prestosql
  velox_memory
  velox_parse_parser
  velox_type
  velox_type_fbhive
  velox_vector
  velox_vector_test_lib)

# older versions of GCC need it to allow std::filesystem
if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
  target_link_libraries(velox_dwio_common_test_utils stdc++fs)
endif()
