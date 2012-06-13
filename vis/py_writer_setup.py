from distutils.core import setup, Extension 
module1 = Extension('visit_writer',
                     include_dirs= ['.'],
                     sources = ['visit_writer.c', 
                                 'py_visit_writer.c']) 
setup (name = 'visit_writer',
       version = '1.0', 
       description = 'This module lets us write VTK files.', 
       ext_modules = [module1])

