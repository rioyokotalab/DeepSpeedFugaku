CC=mpifccpx
CCFLAGS=-Nclang -Kfast -Wall -Knolargepage -ffj-no-fjprof

only_override: only_override.c
	${CC} ${CCFLAGS} -shared -fPIC only_override.c -o only_override.so
