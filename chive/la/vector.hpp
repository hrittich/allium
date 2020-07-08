#ifndef CHIVE_LA_VECTOR_HPP
#define CHIVE_LA_VECTOR_HPP

#include <memory>
#include <complex>
#include <cassert>
#include <chive/util/types.hpp>
#include <chive/mpi/comm.hpp>

namespace chive {
  template <typename StorageT>
  class VectorSlice;

  class VectorSpec final { //: public std::enable_shared_from_this<VectorSpec> {
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
  class VectorStorage
    : public std::enable_shared_from_this<VectorStorage<NumberT>> {
    public:
      template <typename S> friend class VectorSlice;

      using Number = NumberT;
      using Real = real_part_t<Number>;

      using std::enable_shared_from_this<VectorStorage<NumberT>>::shared_from_this;

      VectorStorage(VectorSpec spec) : m_spec(spec) {}

      virtual void add(const VectorStorage& rhs) = 0;
      virtual void scale(const Number& factor) = 0;
      virtual Real l2_norm() const = 0;

      virtual void assign(const VectorStorage& rhs);

      VectorSpec spec() const { return m_spec; }
    protected:
      virtual Number* aquire_data_ptr() = 0;
      virtual void release_data_ptr(Number* data) = 0;

    private:
      VectorSpec m_spec;
  };

  template <typename StorageT>
  class VectorBase {
    public:
      using Storage = StorageT;
      using Number = typename Storage::Number;
      using Real = real_part_t<Number>;
      using Slice = VectorSlice<Storage>;

      VectorBase() = default;

      template <typename S2>
      VectorBase(VectorBase<S2>& other) : ptr(other.storage()) {}

      explicit VectorBase(VectorSpec spec) : ptr(std::make_shared<StorageT>(spec)) {};
      explicit VectorBase(const std::shared_ptr<StorageT>& ptr) : ptr(ptr) {}

      template <typename S2>
      void assign(const VectorBase<S2>& other);

      VectorSpec spec() const { return ptr->spec(); }

      std::shared_ptr<StorageT> storage() { return ptr; }
      std::shared_ptr<const StorageT> storage() const { return ptr; }
    private:
      std::shared_ptr<StorageT> ptr;
  };

  template <typename N>
  using Vector = VectorBase<VectorStorage<N>>;

  template <typename S>
  class VectorSlice final {
    public:
      using Storage = S;
      using Number = typename S::Number;
      using Real = typename S::Real;

      explicit VectorSlice(const std::shared_ptr<S>& ptr) : ptr(ptr) {
        data = ptr->aquire_data_ptr();
        m_size = ptr->spec().get_local_size();
      }

      ~VectorSlice() {
        ptr->release_data_ptr(data);
      }

      Number& operator[] (size_t index) {
        #ifdef CHIVE_BOUND_CHECKS
          if (index >= m_size) {
            throw std::logic_error("Index out of bounds.");
          }
        #endif

        return data[index];
      }

      Number operator[] (size_t index) const {
        return (*const_cast<VectorSlice>(this))[index];
      }

      size_t size() const { return m_size; }

      Number* get_data() { return data; }
    protected:
      Number* data;
      size_t m_size;
    private:
      std::shared_ptr<S> ptr;
  };

  template <typename T, typename F>
  T vector_cast(const F& other) {
    auto ptr = std::dynamic_pointer_cast<typename T::Storage>(other.storage());

    if (ptr) {
      return T(ptr);
    } else {
      T vec;
      vec.assign(other);
      return vec;
    }
  }

  template <typename S>
    VectorSlice<S> local_slice(VectorBase<S> vec)
  {
    return VectorSlice<S>(vec.storage());
  }

  template <typename N>
  void VectorStorage<N>::assign(const VectorStorage& other)
  {
    auto ptr = shared_from_this();
    auto this_loc = VectorSlice<VectorStorage<N>>(ptr);
    auto other_loc
      = VectorSlice<VectorStorage<N>>(const_cast<VectorStorage&>(other).shared_from_this());

    assert(this_loc.size() == other_loc.size());

    for (size_t i_element=0; i_element < other_loc.size(); ++i_element) {
      this_loc[i_element] = other_loc[i_element];
    }
  }

  template <typename S>
  template <typename S2>
  void VectorBase<S>::assign(const VectorBase<S2>& other)
  {
    // allocate new memory
    ptr = std::make_shared<S>(other.spec());

    ptr->assign(*(other.storage()));
  }

}

#endif
