#ifndef CHIVE_LA_EIGEN_SPARSE_MATRIX_HPP
#define CHIVE_LA_EIGEN_SPARSE_MATRIX_HPP

#include "sparse_matrix.hpp"
#include <Eigen/Sparse>

namespace chive {
  template <typename NumberT>
  class EigenSparseMatrixStorage : public SparseMatrixStorage<NumberT> {
    public:
      void add(LocalCooMatrix<NumberT> lmat) override {
        std::cerr << "Warning, add currently overrides entries" << std::endl;

        auto entries = std::move(lmat).get_entries();

        std::vector<Eigen::Triplet<NumberT>> triplets(entries.size());
        std::transform(entries.begin(), entries.end(),
                       triplets.begin(),
                       [] (typename LocalCooMatrix<NumberT>::Entry e)
                         { return Eigen::Triplet<NumberT>(e.row, e.col, e.value); });

        mat.setFromTriplets(triplets.begin(), triplets.end());
      };

    private:
      Eigen::SparseMatrix<NumberT> mat;
  };

}

#endif
