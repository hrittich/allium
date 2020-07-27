#!/usr/bin/env python3
#  Copyright 2020 Hannah Rittich
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import re
from datetime import datetime

year = datetime.now().year
module_name = input('Module name: ')
class_name = input('Class name: ')
author = input('Your name: ')

cn_words = re.findall('[A-Z][^A-Z]*', class_name)
cn_lower = "_".join([ w.lower() for w in cn_words ])
cn_upper = "_".join([ w.upper() for w in cn_words ])

fn_source = f'allium/{module_name}/{cn_lower}.cpp'
fn_header = f'allium/{module_name}/{cn_lower}.hpp'
include_guard = f'ALLIUM_{module_name.upper()}_{cn_upper}_HPP'

license_header = f'''// Copyright {year} {author}
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
// limitations under the License.'''

print(cn_words)
print(cn_lower)
print(fn_source)
print(fn_header)

with open(fn_source, 'x') as fp:
    fp.write(f'''{license_header}

#include "{cn_lower}.hpp"

namespace allium {{
}}
''')

with open(fn_header, 'x') as fp:
    fp.write(f'''{license_header}

#ifndef {include_guard}
#define {include_guard}

namespace allium {{
  class {class_name} {{
    public:

  }};
}}

#endif
''')

