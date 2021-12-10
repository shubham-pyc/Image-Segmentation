nvcc -arch=sm_60 -c ./include/kmeans.cu 
nvcc -arch=sm_60 -c ./include/median_filter.cu 
mpic++ -c main.cpp -fopenmp
mpic++ -o kmeans.out median_filter.o kmeans.o main.o -L/usr/local/cuda/lib64 -lcudart -fopenmp
rm kmeans.o main.o median_filter.o
# ./kmeans.out