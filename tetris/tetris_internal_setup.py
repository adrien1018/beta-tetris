#!/usr/bin/env python3

from setuptools import Extension, setup
import numpy

#args = ['-DDEBUG_METHODS']
args = ['-DDEBUG_METHODS', '-DMIRROR_PIECES_COMPAT']
#args = ['-DDEBUG_METHODS', '-fsanitize=address', '-fsanitize=undefined']

name = 'tetris'
module = Extension(name, sources = ['tetris.cpp'],
        include_dirs = [numpy.get_include()],
        extra_compile_args = args,
        extra_link_args = args)
setup(name = name, ext_modules = [module])
