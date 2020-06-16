#ifndef CHIVE_LA_VECTOR_HPP
#define CHIVE_LA_VECTOR_HPP

#include <memory>
#include <complex>
#include <chive/util/types.hpp>
#include <chive/mpi/comm.hpp>

namespace chive {
  template <typename NumberT, typename StorageT>
  class VectorSlice;

  class VectorSpec : public std::enable_shared_from_this<VectorSpec> {
    public:
      VectorSpec(const VectorSpec&) = default;
      VectorSpec(VectorSpec&&) = default;
      VectorSpec(MpiComm comm, global_size_t global_size, size_t local_size);

      VectorSpec& operator= (const VectorSpec&) = default;
      VectorSpec& operator= (VectorSpec&&) = default;

      MpiComm get_comm() { return comm; }
      global_size_t get_global_size() { return global_size; }
      size_t get_local_size() { return local_size; }
    private:
      MpiComm comm;
      global_size_t global_size;
      size_t local_size;
  };

#if 1
  template <typename T> struct real_part {};
  template <> struct real_part<float> { typedef float type; };
  template <> struct real_part<double> { typedef double type; };
  template <typename T> struct real_part<std::complex<T>> { typedef T type; };

  template <typename T>
  using real_part_t = typename real_part<T>::type;
#endif

  template <typename NumberT>
  class VectorStorage {
    public:
      template <typename N, typename S> friend class VectorSlice;

      using Number = NumberT;
      using Real = real_part_t<Number>;

      VectorStorage(VectorSpec spec) : spec(spec) {}

      virtual void add(const VectorStorage& rhs) = 0;
      virtual void scale(const Number& factor) = 0;
      virtual Real l2_norm() const = 0;

      VectorSpec get_spec() { return spec; }
    protected:
      virtual Number* aquire_data_ptr() = 0;
      virtual void release_data_ptr(Number* data) = 0;

    private:
      VectorSpec spec;
  };

  template <typename NumberT, typename StorageT = VectorStorage<NumberT>>
  class Vector {
    public:
      using Number = NumberT;
      using Real = real_part_t<Number>;
      using Slice = VectorSlice<NumberT, StorageT>;

      Vector(std::shared_ptr<StorageT>& ptr) : ptr(ptr) {}

      Vector(const Vector&) = default;
      Vector& operator= (const Vector&) = default;
      Vector(Vector&&) = default;
      Vector& operator= (Vector&&) = default;

      std::shared_ptr<StorageT> get_storage() { return ptr; }
    private:
      std::shared_ptr<StorageT> ptr;
  };

  template <typename NumberT, typename StorageT = VectorStorage<NumberT>>
  class VectorSlice {
    public:
      using Number = typename Vector<NumberT, StorageT>::Number;
      using Real = typename Vector<NumberT, StorageT>::Real;

      VectorSlice(std::shared_ptr<StorageT> ptr) : ptr(ptr) {
        data = ptr->aquire_data_ptr();
        size = ptr->get_spec().get_local_size();
      }

      ~VectorSlice() {
        ptr->release_data_ptr(data);
      }

      Number& operator[] (size_t index) {
        #ifdef CHIVE_BOUND_CHECKS
          if (index >= size) {
            throw std::logic_error("Index out of bounds.");
          }
        #endif

        return data[index];
      }

      Number operator[] (size_t index) const {
        return (*const_cast<VectorSlice>(this))[index];
      }

      Number* get_data() { return data; }
    protected:
      Number* data;
      size_t size;
    private:
      std::shared_ptr<StorageT> ptr;
  };

  template <typename StorageT>
    VectorSlice<typename StorageT::Number, StorageT>
    local_slice(std::shared_ptr<StorageT> ptr)
  {
    return VectorSlice<typename StorageT::Number, StorageT>(ptr);
  }

  template <typename StorageT>
    VectorSlice<typename StorageT::Number, StorageT>
    local_slice(Vector<typename StorageT::Number, StorageT> vec)
  {
    return local_slice(vec->get_storage());
  }

}

#endif
