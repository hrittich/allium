// File: apple_tree.hpp
#ifndef APPLE_TREE_HPP
#define APPLE_TREE_HPP

#include <string>

// Classes
class AppleTree {
  public:
    AppleTree(std::string location = "unknown");

    // methods
    void shake();
    void shake_strongly();

    // getter/setter
    std::string location() { return m_location; }
    void location(std::string location) { m_location = location; }

  private:
    // member variables
    std::string m_location;
};

#endif
