# -*- coding: utf-8 -*-

# Copyright 2019-2024 Jean-Baptiste Delisle
# Licensed under the EUPL-1.2 or later

import numpy as np
from setuptools import Extension, setup

setup(
  ext_modules=[
    Extension(
      'spleaf.libspleaf',
      sources=['src/spleaf/pywrapspleaf.c', 'src/spleaf/libspleaf.c'],
      language='c',
    )
  ],
  include_dirs=[np.get_include()],
)
