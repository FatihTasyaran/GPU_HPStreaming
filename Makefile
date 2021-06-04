all:
	g++ -c main.cpp mmio.c
	nvcc -c gpu_partition.cu -I /home/users/ftasyaran/thrust/
	nvcc -o trier gpu_partition.o main.o mmio.o -Xcompiler -fopenmp -O3 -I /home/users/ftasyaran/thrust/ 

