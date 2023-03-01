#!/usr/bin/env python3

from setuptools import Extension, setup
import numpy

sources = ['tetris.cpp', 'train_py.cpp', 'train.cpp', 'game_py.cpp', 'game.cpp', 'params.cpp', 'rng.cpp']
args = ['-DDEBUG_METHODS', '-std=c++20']
#args = ['-DDEBUG_METHODS', '-DMIRROR_PIECES_COMPAT']
#args = ['-DDEBUG_METHODS', '-std=c++20', '-fsanitize=address', '-fsanitize=undefined']

name = 'tetris'
module = Extension(
        name,
        sources = sources,
        include_dirs = [numpy.get_include()],
        extra_compile_args = args,
        extra_link_args = args)
setup(name = name, ext_modules = [module])
