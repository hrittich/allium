// File: apple_tree.cpp
#include "apple_tree.hpp"
#include <iostream>

AppleTree::AppleTree(std::string location)
  : m_location(location)
{}

void AppleTree::shake() {
  std::cout << "An apple falls down." << std::endl;
}

void AppleTree::shake_strongly() {
  std::cout << "All apples fall down. The tree looses all its leaves."
            << std::endl;
}
