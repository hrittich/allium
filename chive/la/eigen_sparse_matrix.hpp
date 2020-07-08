#ifndef CHIVE_LA_EIGEN_SPARSE_MATRIX_HPP
#define CHIVE_LA_EIGEN_SPARSE_MATRIX_HPP

#include "sparse_matrix.hpp"
#include "eigen_vector.hpp"
#include <Eigen/Sparse>

namespace chive {
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
        current = Eigen::Triplet<N>(base->get_row(),
                                    base->get_col(),
                                    base->get_value());
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


  template <typename N>
  class EigenSparseMatrixStorage final : public SparseMatrixStorage<N> {
    public:
      using NativeVector = EigenVector<N>;
      using SparseMatrixStorage<N>::row_spec;
      using SparseMatrixStorage<N>::col_spec;

      EigenSparseMatrixStorage(VectorSpec rows, VectorSpec cols)
        : SparseMatrixStorage<N>(rows, cols),
          mat(rows.get_global_size(), cols.get_global_size()) {}

      void set_entries(LocalCooMatrix<N> lmat) override {
        auto entries = std::move(lmat).get_entries();

        mat.setFromTriplets(
          triplet_iterator<N>(entries.begin()),
          triplet_iterator<N>(entries.end()));
      };

      LocalCooMatrix<N> get_entries() override {
        LocalCooMatrix<N> lmat;

        for (size_t k=0; k < mat.outerSize(); ++k) {
          for (typename Eigen::SparseMatrix<N>::InnerIterator it(mat,k); it; ++it)
          {
            lmat.add(it.row(), it.col(), it.value());
          }
        }

        return lmat;
      }

      Vector<N> vec_mult(const Vector<N>& v) override {
        auto ptr = std::dynamic_pointer_cast<const EigenVectorStorage<N>> (v.storage());
        if (!ptr)
          throw std::logic_error("Not implemented");

        NativeVector ret(row_spec());
        ret.storage()->native()
          = mat * (std::const_pointer_cast<EigenVectorStorage<N>>(ptr)->native());
        return ret;
      }

    private:
      Eigen::SparseMatrix<N> mat;
  };

  template <typename N>
  using EigenSparseMatrix = SparseMatrixBase<EigenSparseMatrixStorage<N>>;
}

#endif
