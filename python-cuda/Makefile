
CC=nvcc


_fun.so:  fun.o
#gcc -shared fun.o -o _fun.so
	gcc -bundle -flat_namespace -undefined suppress -o _fun.so fun.o -L/Developer/NVIDIA/CUDA-7.0/lib/ -lcuda -lcudart -lcublas

fun.o:  fun.c
	nvcc  -c fun.c -I/usr/include/python2.7 -I/usr/local/lib/python2.7/site-packages/numpy/core/include/numpy -L/Developer/NVIDIA/CUDA-7.0/lib/ -lcuda -lcudart -lcublas

clean:
	rm *.o *.so


	
# _fun.so:  fun.o 
# 	#gcc -shared fun.o -o _fun.so  
# 	gcc -bundle -flat_namespace -undefined suppress -o _fun.so fun.o

# fun.o:  fun.c
# 	nvcc  -c fun.c -I/usr/include/python2.7 -I/usr/local/lib/python2.7/site-packages/numpy/core/include/numpy

# clean: 
# 	rm *.o *.so

