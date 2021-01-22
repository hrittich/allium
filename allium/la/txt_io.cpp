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

#include "txt_io.hpp"
#include <fstream>

namespace allium {

void write_txt(std::string filename, std::vector<LocalVector<double>> columns)
{
  // check that the columns have equal length
  for (size_t i = 1; i < columns.size(); ++i) {
    if (columns.at(0).rows() != columns.at(i).rows()) {
      throw std::runtime_error("Columns have different lengths.");
    }
  }

  std::fstream fp(filename, std::ios_base::out);

  for (size_t i_row = 0; i_row < columns.at(0).rows(); ++i_row)
  {
    bool first = true;
    for (size_t i_col = 0; i_col < columns.size(); ++i_col) {
      if (first) first = false;
      else fp << " ";

      fp << columns[i_col][i_row];
    }
    fp << std::endl;
  }

}

}

