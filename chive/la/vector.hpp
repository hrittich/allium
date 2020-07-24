#ifndef CHIVE_LA_VECTOR_HPP
#define CHIVE_LA_VECTOR_HPP

#include <memory>
#include <complex>
#include <cassert>
#include <chive/util/types.hpp>
#include <chive/util/extern.hpp>
#include <chive/mpi/comm.hpp>
#include "vector_spec.hpp"

namespace chive {
  template <typename, bool> class VectorSlice;

  class BaseTag {};

  template <typename T> struct real_part {};
  template <> struct real_part<float> { typedef float type; };
  template <> struct real_part<double> { typedef double type; };
  template <typename T> struct real_part<std::complex<T>> { typedef T type; };

  template <typename T>
  using real_part_t = typename real_part<T>::type;

  /** Abstract base class for any vector storage. */
  template <typename N>
  class VectorStorage : public std::enable_shared_from_this<VectorStorage<N>> {
    public:
      template <typename, bool> friend class VectorSlice;

      // type traits
      using Number = N;
      using Real = real_part_t<Number>;

      VectorStorage(VectorSpec spec) : m_spec(spec) {}
      virtual ~VectorStorage() {}

      // Public virtual vector interface
      virtual void add(const VectorStorage& rhs) = 0;
      virtual void scale(const Number& factor) = 0;
      virtual Real l2_norm() const = 0;
      virtual Number dot(const VectorStorage& rhs) = 0;

      virtual void assign(std::shared_ptr<const VectorStorage> rhs) = 0;
      virtual std::unique_ptr<VectorStorage> allocate(VectorSpec spec, BaseTag = {}) = 0;

      VectorSpec spec() const { return m_spec; }
    protected:
      // Private virtual interface
      virtual Number* aquire_data_ptr() = 0;
      virtual void release_data_ptr(Number* data) = 0;

    private:
      VectorSpec m_spec;
  };

  template <typename N, bool is_const=false>
  class VectorSlice final {
    public:
      using Storage = typename std::conditional<is_const,
                                                const VectorStorage<N>,
                                                VectorStorage<N>>::type;
      using Number = typename Storage::Number;
      using Real = typename Storage::Real;
      using Reference = typename std::conditional<is_const,
                                                  Number,
                                                  Number&>::type;
      using MutableStorage = typename std::remove_const<Storage>::type;

      explicit VectorSlice(const std::shared_ptr<Storage>& ptr) : ptr(ptr) {
        data = std::const_pointer_cast<MutableStorage>(ptr)
               ->aquire_data_ptr();
        m_size = ptr->spec().local_size();
      }

      ~VectorSlice() {
        std::const_pointer_cast<MutableStorage>(ptr)->release_data_ptr(data);
      }

      VectorSlice& operator= (const std::initializer_list<N>& rhs) {
        #ifdef CHIVE_BOUND_CHECKS
          if (rhs.size() != m_size) {
            throw std::logic_error("Invalid length of initilizer list.");
          }
        #endif
        size_t i_element = 0;
        for (auto element : rhs) {
          (*this)[i_element] = element;
          ++i_element;
        }
        return *this;
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
      std::shared_ptr<Storage> ptr;
  };

  template <typename Derived, typename N>
  class VectorStorageBase
    : public VectorStorage<N> {
    public:
      using typename VectorStorage<N>::Number;
      using typename VectorStorage<N>::Real;

      using VectorStorage<N>::spec;
      using VectorStorage<N>::shared_from_this;

      VectorStorageBase(VectorSpec spec) : VectorStorage<N>(spec) {}

      std::unique_ptr<Derived> allocate(VectorSpec spec) {
        return std::make_unique<Derived>(spec);
      }
      std::unique_ptr<VectorStorage<N>> allocate(VectorSpec spec, BaseTag) override {
        return allocate(spec);
      }

      void assign(std::shared_ptr<const VectorStorage<N>> other)
      {
        auto ptr = shared_from_this();
        auto this_loc = VectorSlice<N>(ptr);
        auto other_loc = VectorSlice<N, true>(other);

        assert(this_loc.size() == other_loc.size());

        for (size_t i_element=0; i_element < other_loc.size(); ++i_element) {
          this_loc[i_element] = other_loc[i_element];
        }
      }
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
      void assign(const VectorBase<S2>& other)
      {
        ptr->assign(other.storage());
      }

      VectorSpec spec() const { return ptr->spec(); }

      std::shared_ptr<StorageT> storage() { return ptr; }
      std::shared_ptr<const StorageT> storage() const { return ptr; }

      VectorBase uninitialized_like() const {
        return VectorBase(std::shared_ptr<StorageT>(ptr->allocate(spec())));
      }

      void set_zero()
      {
        auto loc = VectorSlice<Number>(ptr);
        for (size_t i_loc=0; i_loc < loc.size(); ++i_loc) {
          loc[i_loc] = 0;
        }
      }

      VectorBase<VectorStorage<Number>> zeros_like() const {
        auto v = uninitialized_like();
        v.set_zero();
        return v;
      }

      VectorBase operator+ (const VectorBase& rhs) const {
        VectorBase aux(uninitialized_like());
        aux.assign(*this);
        aux.ptr->add(*(rhs.ptr));
        return aux;
      }

      VectorBase operator- (const VectorBase& rhs) const {
        VectorBase aux(uninitialized_like());
        aux.assign(rhs);
        aux.ptr->scale(-1);
        aux.ptr->add(*ptr);
        return aux;
      }

      VectorBase operator/ (Number rhs) const {
        VectorBase aux(uninitialized_like());
        aux.assign(*this);
        aux.ptr->scale(Number(1.0) / rhs);
        return aux;
      }

      VectorBase& operator+= (const VectorBase& rhs) {
        ptr->add(*rhs.ptr);
        return *this;
      }

      VectorBase& operator-= (const VectorBase& rhs) {
        VectorBase aux(rhs.uninitialized_like());
        aux.assign(rhs);
        aux.ptr->scale(-1);
        (*this) += aux;
        return *this;
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
    VectorSlice<typename S::Number> local_slice(VectorBase<S> vec)
  {
    return VectorSlice<typename S::Number>(vec.storage());
  }

  template <typename N>
  Vector<N> make_vector(VectorSpec spec);

  #define CHIVE_LA_VECTOR_DECL(T, N) \
    T Vector<N> make_vector(VectorSpec);
  CHIVE_EXTERN(CHIVE_LA_VECTOR_DECL)
}

#endif
