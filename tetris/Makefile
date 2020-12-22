all:
	python tetris_internal_setup.py build_ext --inplace
cpp:
	g++ -O2 -o tetris -I /usr/include/python3.9 -DDEBUG -std=c++14 tetris.cpp
clean:
	rm -rf build *.so
