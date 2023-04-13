TARGET=main
OBJECTS=main.o util.o namegen.o

# CPPFLAGS=-std=c++14 -O3 -Wall -march=native -mavx2 -mfma -fopenmp -mno-avx512f -I/usr/local/cuda/include -I/home/gyuseonglee/opt/ompi/include
CPPFLAGS=-std=c++14 -O3 -Wall -march=native -mavx2 -mfma -mno-avx512f -I/usr/local/cuda/include 

CUDA_CPPFLAGS:=$(foreach option, $(CPPFLAGS),-Xcompiler=$(option))

LDFLAGS=-pthread -L/usr/local/cuda/lib64
LDLIBS=-lstdc++ -lcudart -lm

CXX=g++
CUX=/usr/local/cuda/bin/nvcc

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CPPFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c -o $@ $^

%.o: %.cu
	$(CUX) $(CUDA_CPPFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)

