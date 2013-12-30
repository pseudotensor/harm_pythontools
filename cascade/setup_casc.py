from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("casc", ["casc.pyx"], include_dirs=[np.get_include()], extra_compile_args=["-O3"])]

setup(
  name = 'Jet Cascade app',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
