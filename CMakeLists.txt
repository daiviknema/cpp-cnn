cmake_minimum_required(VERSION 3.5)

project(cpp-cnn)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(Boost 1.40 COMPONENTS unit_test_framework REQUIRED)
find_package(Armadillo 6.5 REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
include_directories(${ARMADILLO_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(
  convolution_layer_test
  tests/convolution_layer_test.cpp
  )

target_link_libraries(convolution_layer_test LINK_PUBLIC ${Boost_LIBRARIES})
target_link_libraries(convolution_layer_test LINK_PUBLIC ${ARMADILLO_LIBRARIES})

add_executable(
  dense_layer_test
  tests/dense_layer_test.cpp
  )

target_link_libraries(dense_layer_test LINK_PUBLIC ${Boost_LIBRARIES})
target_link_libraries(dense_layer_test LINK_PUBLIC ${ARMADILLO_LIBRARIES})

add_executable(
  max_pooling_layer_test
  tests/max_pooling_layer_test.cpp
  )

target_link_libraries(max_pooling_layer_test LINK_PUBLIC ${Boost_LIBRARIES})
target_link_libraries(max_pooling_layer_test LINK_PUBLIC ${ARMADILLO_LIBRARIES})

add_executable(
  softmax_layer_test
  tests/softmax_layer_test.cpp
  )

target_link_libraries(softmax_layer_test LINK_PUBLIC ${Boost_LIBRARIES})
target_link_libraries(softmax_layer_test LINK_PUBLIC ${ARMADILLO_LIBRARIES})

add_executable(
  relu_layer_test
  tests/relu_layer_test.cpp
  )

target_link_libraries(relu_layer_test LINK_PUBLIC ${Boost_LIBRARIES})
target_link_libraries(relu_layer_test LINK_PUBLIC ${ARMADILLO_LIBRARIES})

add_executable(
  cross_entropy_loss_layer_test
  tests/cross_entropy_loss_layer_test.cpp
  )

target_link_libraries(cross_entropy_loss_layer_test LINK_PUBLIC ${Boost_LIBRARIES})
target_link_libraries(cross_entropy_loss_layer_test LINK_PUBLIC ${ARMADILLO_LIBRARIES})

add_executable(
  integration_test
  tests/integration_test.cpp
  )

target_link_libraries(integration_test LINK_PUBLIC ${Boost_LIBRARIES})
target_link_libraries(integration_test LINK_PUBLIC ${ARMADILLO_LIBRARIES})

add_executable(
  mnist_util_test
  tests/mnist_test.cpp
  )


target_link_libraries(mnist_util_test LINK_PUBLIC ${Boost_LIBRARIES})
target_link_libraries(mnist_util_test LINK_PUBLIC ${ARMADILLO_LIBRARIES})

add_executable(
  le_net
  le_net.cpp
  )


target_link_libraries(le_net LINK_PUBLIC ${Boost_LIBRARIES})
target_link_libraries(le_net LINK_PUBLIC ${ARMADILLO_LIBRARIES})

file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# file(COPY ${CMAKE_SOURCE_DIR}/data_medium DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# file(COPY ${CMAKE_SOURCE_DIR}/data_small DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

