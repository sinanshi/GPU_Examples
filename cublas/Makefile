cc=nvcc
LIBS=-lcublas
OBJ=dot.o


%.o: %.cu
	$(cc) -c -o $@ $^ $(LIBS)
dot: $(OBJ)
	$(cc) -o $@ $^ $(LIBS)

clean: 
	rm -f $(OBJ) dot



