TARGET=main
OBJECTS=main.o util.o uNet.o tensor.o

CFLAGS=-std=c++14 -O3 -Wall -march=native -mavx2 -mfma -fopenmp -I/usr/local/cuda/include
CUDA_CFLAGS:=$(foreach option, $(CFLAGS),-Xcompiler=$(option))

LDFLAGS=-pthread -L/usr/local/cuda/lib64
LDLIBS= -lstdc++ -lcudart -lm

CXX=g++
CUX=/usr/local/cuda/bin/nvcc

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c -o $@ $^

%.o: %.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)
	
run: main
	sbatch run.sh model/weights.bin img/bin/img1.bin output1.bin 1

run2: main
	sbatch run.sh model/weights.bin img/bin/img2N.bin output2N.bin 2

run4: main
	sbatch run.sh model/weights.bin img/bin/img4N.bin output4N.bin 4
