CFLAGS_INT=-g -O3 -Wall ${CFLAGS}
LDFLAGS_INT=-g -lm -lrt -O3 -llapack -lgsl  -lgslcblas  ${LDFLAGS}

all: danny_interface

danny_interface: danny_interface.o
	g++ ${LDFLAGS_INT} $< -o $@

.cpp.o:
	g++ ${CFLAGS_INT} -c $< -o $@

clean:
	rm danny_interface danny_interface.o
