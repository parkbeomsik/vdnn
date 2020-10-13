NVCC=/home/beomsik/cuda/bin/nvcc
all:
	$(NVCC) -lcudnn -lcublas -lcnmem vdnn_sample.cpp -o vdnn_sample

clean:
	-rm -f vdnn_sample
