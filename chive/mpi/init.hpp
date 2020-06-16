#ifndef CHIVE_MPI_INIT_HPP
#define CHIVE_MPI_INIT_HPP

namespace chive {
  class MpiInit {
    public:
      MpiInit(int& argc, char** &argv);
      ~MpiInit();

    private:
      static MpiInit* instance;
  };
}

#endif
