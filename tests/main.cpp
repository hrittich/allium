#include <gtest/gtest.h>

#include <chive/main/init.hpp>

int main(int argc, char **argv) {
  chive::Init init(argc, argv);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

