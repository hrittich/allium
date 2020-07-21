// File: main.cpp
#include "apple_tree.hpp"
#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]) {
  AppleTree tree("in the backyard");
  tree.shake();
  std::cout << "The tree is " << tree.location() << "." << std::endl;
  tree.shake_strongly();

  return EXIT_SUCCESS;
}
