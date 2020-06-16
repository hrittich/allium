#ifndef CHIVE_MAIN_INIT_HPP
#define CHIVE_MAIN_INIT_HPP

#include <chive/mpi/init.hpp>

namespace chive {
  class Init {
    public:
      Init(const Init&) = delete;
      Init& operator=(const Init&) = delete;

      Init(int& argc, char** &argv);
      ~Init();

    private:
      MpiInit mpi;
  };
}

#endif
