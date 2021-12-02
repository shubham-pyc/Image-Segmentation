# Image Segmentation Using K-Means Clustering

This is the implementation for Image segmentation using k-means clustering and subtractive clustering algorithms. The research paper can be found here [Link](https://www.sciencedirect.com/science/article/pii/S1877050915014143?via%3Dihub)


## Theory
Image segmentation: It's a classification of the image into different groups. There are various methods to do image segmentation. One of the popular machine learning algorithms to do image segmentation is K-Means clustering.

K-Means Clustering: Is an unsupervised learning algoritm which would assign the data into k clusters with the nearest mean.

## Installations
This is a stand alone project implemented from scratch.

```shell
git clone
```

```shell
g++ main.cpp
./a.out
```


## Linking Cuda file to cPP

```shell
$ nvcc -arch=sm_20 -c file1.cu
$ g++ -c file2.cpp
$ g++ -o test file1.o file2.o -L/usr/local/cuda/lib64 -lcudart
$ ./test
```