// Copyright 2020 Hannah Rittich
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ALLIUM_LA_VECTOR_HPP
#define ALLIUM_LA_VECTOR_HPP

#include <memory>
#include <allium/util/types.hpp>
#include <allium/util/cloneable.hpp>
#include <allium/util/assert.hpp>
#include <allium/util/cloning_ptr.hpp>
#include "vector_trait.hpp"
#include "vector_storage.hpp"

namespace allium {

  template <typename T>
  struct make {
    template <typename ...Args>
    std::unique_ptr<T> operator() (Args&& ...args) {
      return std::make_unique<T>(std::forward<Args>(args)...);
    }
  };

  template <typename Storage_,
            typename Allocator = make<Storage_>>
  class VectorBase
    : public VectorTrait<VectorBase<Storage_, Allocator>,
                         typename Storage_::Number>
  {
    public:
      using Storage = Storage_;
      using Number = typename Storage::Number;
      using Real = real_part_t<Number>;
      using Slice = LocalSlice<Storage>;

      VectorBase() = default;

      // Create
      template <typename S2, typename D2>
      VectorBase(const VectorBase<S2, D2>& other)
        : ptr(clone(other.storage()))
      {}

      explicit VectorBase(VectorSpec spec)
        : ptr(make_storage(spec))
      {}

      explicit VectorBase(std::unique_ptr<Storage> ptr)
        : ptr(std::move(ptr))
      {}

      VectorSpec spec() const { return ptr->spec(); }

      Storage& storage() { return *ptr; }
      const Storage& storage() const { return *ptr; }

      VectorBase uninitialized_like() const {
        return VectorBase(allocate_like(*ptr));
      }

      void set_zero()
      {
        auto loc = LocalSlice<Storage*>(ptr.get());
        for (size_t i_loc=0; i_loc < loc.size(); ++i_loc) {
          loc[i_loc] = 0;
        }
      }

      VectorBase zeros_like() const {
        auto v = uninitialized_like();
        v.set_zero();
        return v;
      }

      template <typename Other>
      VectorBase& operator+= (const Other& rhs) {
        *ptr += *rhs.ptr;
        return *this;
      }

      VectorBase& operator*= (Number factor) {
        *ptr *= factor;
        return *this;
      }

      template <typename Other>
      Number dot(const Other& rhs) const {
        return ptr->dot(rhs.storage());
      }

      Real l2_norm() const {
        return ptr->l2_norm();
      }
    private:
      CloningPtr<Storage> ptr;
      Allocator make_storage;
  };

  template <typename N>
  struct make_default_vector_storage {
    std::unique_ptr<VectorStorage<N>> operator() (VectorSpec spec);
  };

  template <typename N>
  using Vector = VectorBase<VectorStorage<N>, make_default_vector_storage<N>>;

  template <typename S, typename D>
    LocalSlice<S*> local_slice(VectorBase<S, D>& vec)
  {
    return LocalSlice<S*>(&vec.storage());
  }

  template <typename N>
  Vector<N> make_vector(VectorSpec spec);

  #define ALLIUM_LA_VECTOR_DECL(T, N) \
    T class VectorStorage<N>; \
    T struct make_default_vector_storage<N>;
  ALLIUM_EXTERN(ALLIUM_LA_VECTOR_DECL)
}



#endif
