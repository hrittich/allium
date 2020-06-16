#ifndef CHIVE_LA_VECTOR_HPP
#define CHIVE_LA_VECTOR_HPP

#include <memory>
#include <complex>
#include <chive/util/types.hpp>
#include <chive/mpi/comm.hpp>

namespace chive {
  template <typename NumberT>
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

  template <typename T> struct real_part_t {};
  template <> struct real_part_t<float> { typedef float type; };
  template <> struct real_part_t<double> { typedef double type; };
  template <typename T> struct real_part_t<std::complex<T>> { typedef T type; };

  template <typename NumberT>
  class Vector {
    public:
      using Number = NumberT;
      using Real = typename real_part_t<Number>::type;

      virtual void add(const Vector& rhs) = 0;
      virtual void scale(const Number& factor) = 0;
      virtual Real l2_norm() const = 0;

      virtual std::unique_ptr<VectorSlice<Number>> local_slice() = 0;
  };

  template <typename StoragePtrT>
  class Vector_ {
    public:
      template <typename OtherPtrT>
      friend class Vector_;

      Vector_(const StoragePtrT& ptr) : ptr(ptr) {}
      Vector_(StoragePtrT&& ptr) : ptr(std::move(ptr)) {}

      template <typename OtherPtrT>
      Vector_(const Vector_<OtherPtrT>& vec) : ptr(vec.ptr) {}

      template <typename OtherPtrT>
      Vector_& operator= (const Vector_<OtherPtrT>& vec) {
        ptr = vec->ptr;
        return *this;
      }

      template <typename OtherPtrT>
      Vector_(Vector_<OtherPtrT>&& vec) : ptr(std::move(vec.ptr)) {}

      template <typename OtherPtrT>
      Vector_& operator= (Vector_<OtherPtrT>&& vec) {
        ptr = std::move(vec->ptr);
        return *this;
      }

    private:
      StoragePtrT ptr;
  };

  template <typename NumberT>
  class VectorSlice {
    public:
      using Number = typename Vector<NumberT>::Number;
      using Real = typename Vector<NumberT>::Real;

      VectorSlice();

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
  };

  template <typename NumberT>
  VectorSlice<NumberT>::VectorSlice()
    : data(nullptr), size(0)
  {}

}

#endif
