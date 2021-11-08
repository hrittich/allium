
#ifndef ALLIUM_LA_CUDA_ARRAY_HPP
#define ALLIUM_LA_CUDA_ARRAY_HPP

#include <allium/config.hpp>
#ifdef ALLIUM_USE_CUDA

namespace allium {
  
  /**
   * Stores an array on the GPU. Memory is automatically managed.
   */
  template <typename T>
  class CudaArray final {
    public:
      CudaArray(size_t element_count=0);

      CudaArray(const CudaArray&) = delete;
      CudaArray& operator= (const CudaArray&) = delete;

      CudaArray(CudaArray&& other);
      CudaArray& operator= (CudaArray&& other);

      ~CudaArray();

      void resize(size_t element_count);

      T* ptr() { return m_ptr; }
      const T* ptr() const { return m_ptr; }
    private:
      T* m_ptr;
  };

}

#endif
#endif