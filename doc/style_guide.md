# Style Guide

## Synopsis

    #ifndef APPLE_TREE_HPP
    #define APPLE_TREE_HPP

    // File: apple_tree.hpp
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

## Stub Generator

The script `scripts/stub.py` can be used to generate new files.

## General Style

- Lines should be no longer than 80 characters.
- Use 2 spaces to indent.
- Curly braces add and remove *one* level of indentation, i.e., write

      if (a == b) {
        x = 42;

  or

      if (a == b)
      {
        x = 42;

  do **not** write

      if (a == b)
        {
          x = 42;

## Naming Conventions


## Function Parameters

Output arguments should be before the input arguments in the parameter list.
The idea being that `compute(x, a, b)` reads like `x = compute(a, b)`.

