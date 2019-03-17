1. copy matrixMul.cubin from ../matrixMul_cuda/build/CMakeFiles/...

2. maxas.pl -i matrixMul_64x64.sass matrixMul.cubin matrixMul_64x64.cubin

3. mkdir build & compile code
	
	mkdir build;cd build;cmake ..;make
