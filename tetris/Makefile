all:
	python tetris_internal_setup.py build_ext --inplace
	chmod 755 *.so
cpp: tetris

tetris: tetris.cpp
	g++ -Og -g -Wall -Wextra -o tetris -I /usr/include/python3.9 -DDEBUG -std=c++14 tetris.cpp

clean:
	rm -rf build *.so
