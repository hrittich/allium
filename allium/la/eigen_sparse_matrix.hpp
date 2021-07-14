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

#ifndef ALLIUM_LA_EIGEN_SPARSE_MATRIX_HPP
#define ALLIUM_LA_EIGEN_SPARSE_MATRIX_HPP

#include "sparse_matrix.hpp"
#include "eigen_vector.hpp"
#include "linear_operator.hpp"
#include <allium/util/except.hpp>
#include <Eigen/Sparse>

namespace allium {
  /** @cond INTERNAL */
  template <typename N>
  class triplet_iterator final {
    public:
      typedef typename std::vector<MatrixEntry<N>>::iterator base_iterator;

      typedef typename base_iterator::difference_type difference_type;
      typedef Eigen::Triplet<N> value_type;
      typedef const Eigen::Triplet<N> *pointer;
      typedef const Eigen::Triplet<N> &reference;
      typedef std::forward_iterator_tag iterator_category;

      explicit triplet_iterator(base_iterator base) : base(base) {};

      bool operator!= (const triplet_iterator& other) const {
        return base != other.base;
      }

      reference operator* () {
        current = Eigen::Triplet<N>(base->row(),
                                    base->col(),
                                    base->value());
        return current;
      }

      pointer operator-> () {
        return &(**this);
      }

      triplet_iterator& operator++ () {
        ++base;
        return *this;
      }
    private:
      base_iterator base;
      Eigen::Triplet<N> current;
  };
  /** @endcond */

  /**
    @brief A sparse matrix implementation based on Eigen.
   */
  template <typename N>
  class EigenSparseMatrixStorage final
      : public SparseMatrixStorage<EigenVectorStorage<N>>
		{
    public:
      using Vector = EigenVectorStorage<N>;
      using DefaultVector = EigenVectorStorage<N>;
      using SparseMatrixStorage<Vector>::Number;
      using SparseMatrixStorage<Vector>::Real;
      using SparseMatrixStorage<Vector>::row_spec;
      using SparseMatrixStorage<Vector>::col_spec;

      EigenSparseMatrixStorage(VectorSpec rows, VectorSpec cols)
        : SparseMatrixStorage<Vector>(rows, cols),
          m_mat(rows.global_size(), cols.global_size()) {}

      void set_entries(LocalCooMatrix<N> lmat) override {
        auto entries = std::move(lmat).entries();

        m_mat.setFromTriplets(
          triplet_iterator<N>(entries.begin()),
          triplet_iterator<N>(entries.end()));
      };

      LocalCooMatrix<N> get_entries() override {
        LocalCooMatrix<N> lmat;

        for (long k=0; k < m_mat.outerSize(); ++k) {
          for (typename Eigen::SparseMatrix<N>::InnerIterator it(m_mat,k); it; ++it)
          {
            lmat.add(it.row(), it.col(), it.value());
          }
        }

        return lmat;
      }

      void apply(EigenVectorStorage<N>& result, const EigenVectorStorage<N>& arg) {
        result.native() = m_mat * arg.native();
      }

    private:
      Eigen::SparseMatrix<N> m_mat;
  };
}

#endif
