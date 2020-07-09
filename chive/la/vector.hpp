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

      bool operator!= (const VectorSpec& other) {
        return (m_comm != other.m_comm)
               || (m_global_size != other.m_global_size)
               || (m_local_size != other.m_local_size);
      }

      MpiComm comm() { return m_comm; }
      global_size_t global_size() { return m_global_size; }
      size_t local_size() { return m_local_size; }
    private:
      MpiComm m_comm;
      global_size_t m_global_size;
      size_t m_local_size;
  };

  template <typename T> struct real_part {};
  template <> struct real_part<float> { typedef float type; };
  template <> struct real_part<double> { typedef double type; };
  template <typename T> struct real_part<std::complex<T>> { typedef T type; };

  template <typename T>
  using real_part_t = typename real_part<T>::type;

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
      virtual Number dot(const VectorStorage& rhs) = 0;
      virtual std::shared_ptr<VectorStorage> allocate(VectorSpec spec) = 0;

      virtual void assign(std::shared_ptr<const VectorStorage> rhs);

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
      VectorBase(VectorBase<S2> other) : ptr(other.storage()) {}

      explicit VectorBase(VectorSpec spec) : ptr(std::make_shared<StorageT>(spec)) {};
      explicit VectorBase(std::shared_ptr<StorageT> ptr) : ptr(ptr) {}

      template <typename S2>
      void assign(const VectorBase<S2>& other);

      VectorSpec spec() const { return ptr->spec(); }

      std::shared_ptr<StorageT> storage() { return ptr; }
      std::shared_ptr<const StorageT> storage() const { return ptr; }

      VectorBase uninitialized_like() const {
        return VectorBase(std::dynamic_pointer_cast<Storage>(ptr->allocate(spec())));
      }

      VectorBase<VectorStorage<Number>> zeros_like() const {
        auto v = uninitialized_like();
        v.set_zero();
        return v;
      }

      void set_zero();

      VectorBase operator+ (const VectorBase& rhs) const {
        VectorBase aux(rhs.uninitialized_like());
        aux.assign(*this);
        aux.ptr->add(*(rhs.ptr));
        return aux;
      }

      VectorBase operator- (const VectorBase& rhs) const {
        // important todo: does this change rhs???
        VectorBase aux(rhs.uninitialized_like());
        aux.assign(rhs);
        aux.ptr->scale(-1);
        aux.ptr->add(*ptr);
        return aux;
      }

      Number dot(const VectorBase& rhs) const {
        return ptr->dot(*(rhs.storage()));
      }

      Real l2_norm() const {
        return ptr->l2_norm();
      }
    private:
      std::shared_ptr<StorageT> ptr;
  };

  template <typename S>
  VectorBase<S> operator* (typename S::Number s, const VectorBase<S>& v) {
    VectorBase<S> result = v.uninitialized_like();
    result.assign(v);
    result.storage()->scale(s);
    return result;
  }

  template <typename N>
  using Vector = VectorBase<VectorStorage<N>>;

  template <typename S>
  class VectorSlice final {
    public:
      using Storage = S;
      using Number = typename S::Number;
      using Real = typename S::Real;
      using Reference = typename std::conditional<std::is_const<S>::value,
                                                  Number,
                                                  Number&>::type;
      using MutableStorage = typename std::remove_const<S>::type;

      explicit VectorSlice(const std::shared_ptr<S>& ptr) : ptr(ptr) {
        data = std::const_pointer_cast<MutableStorage>(ptr)
               ->aquire_data_ptr();
        m_size = ptr->spec().local_size();
      }

      ~VectorSlice() {
        std::const_pointer_cast<MutableStorage>(ptr)->release_data_ptr(data);
      }

      Reference operator[] (size_t index) {
        #ifdef CHIVE_BOUND_CHECKS
          if (index >= m_size) {
            throw std::logic_error("Index out of bounds.");
          }
        #endif

        return data[index];
      }

      size_t size() const { return m_size; }

      //Number* get_data() { return data; }
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
      T vec(other.spec());
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
  void VectorStorage<N>::assign(std::shared_ptr<const VectorStorage> other)
  {
    auto ptr = shared_from_this();
    auto this_loc = VectorSlice<VectorStorage<N>>(ptr);
    auto other_loc = VectorSlice<const VectorStorage<N>>(other);

    assert(this_loc.size() == other_loc.size());

    for (size_t i_element=0; i_element < other_loc.size(); ++i_element) {
      this_loc[i_element] = other_loc[i_element];
    }
  }

  template <typename S>
  template <typename S2>
  void VectorBase<S>::assign(const VectorBase<S2>& other)
  {
    assert(ptr);
    assert(other.ptr);

    // check that the vectors have the same size allocation
    if (spec() != other.spec()) {
      throw std::logic_error("Cannot assign a vector with a different "
                             "specification");
    }
    ptr->assign(other.storage());
  }

  template <typename S>
  void VectorBase<S>::set_zero()
  {
    auto loc = local_slice(*this);
    for (size_t i_loc=0; i_loc < loc.size(); ++i_loc) {
      loc[i_loc] = 0;
    }
  }

}

#endif
