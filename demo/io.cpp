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

#include <allium/la/txt_io.hpp>

using namespace allium;

void write_sin_table() {
  const double a = -5;
  const double b = 5;
  const int row_count = 1024;

  LocalVector<double> xs(row_count), ys(row_count);

  for (int i_row = 0; i_row < row_count; ++i_row) {
    double t = static_cast<double>(i_row) / (row_count-1);
    double x = (1-t) * a + t * b;
    double y = sin(x);
    xs[i_row] = x;
    ys[i_row] = y;
  }

  write_txt("sin_table.txt", {std::move(xs), std::move(ys)});
}

int main(int argc, char* argv[]) {
  write_sin_table();

  return 0;
}

