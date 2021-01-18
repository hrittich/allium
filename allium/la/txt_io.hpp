// Copyright 2021 Hannah Rittich
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

#ifndef ALLIUM_LA_GNUPLOT_HPP
#define ALLIUM_LA_GNUPLOT_HPP

#include "local_vector.hpp"
#include <string>
#include <vector>

namespace allium {

  /// @addtogroup io
  /// @{

  /**
    Write a set of columns into a text file.

    The numbers of each row are written as ASCII encoded text, separated by
    spaces. Each row is terminated by a new-line character.

    The generated file can be loaded into Numpy using the `numpy.loadtxt`
    method of plotted using the Gnuplot `plot` command.
  */
  void write_txt(std::string filename, std::vector<LocalVector<double>> columns);

  /// @}
}

#endif
