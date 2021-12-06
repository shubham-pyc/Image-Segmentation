nvcc -arch=sm_60 -c ./include/kmeans.cu 
mpic++ -c main.cpp -fopenmp
mpic++ -o kmeans.out kmeans.o main.o -L/usr/local/cuda/lib64 -lcudart -fopenmp
rm kmeans.o main.o
# ./kmeans.out