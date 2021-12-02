nvcc -c ./include/kmeans.cu 
g++ -c main.cpp -fopenmp
g++ -o kmeans.out kmeans.o main.o -L/usr/local/cuda/lib64 -lcudart
rm kmeans.o main.o
./kmeans.out