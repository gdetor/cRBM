IDIR := ./include
SRC := ./src
BIN := ./bin
DATADIR := ./data

CC=gcc
CFLAGS= -Ofast -flto -fstrict-aliasing -march=native -mtune=native \
        -faggressive-loop-optimizations -Wall -funroll-loops -Wno-unused-result \
	-floop-parallelize-all -I $(IDIR)
# CFLAGS:= -Ofast -flto -march=native -mtune=native -faggressive-loop-optimizations -I $(IDIR)

LIBS:=-lm
ODIR:=obj

_DEPS := rbm.h pcg_basic.h
DEPS := $(patsubst %,$(IDIR)/%,$(_DEPS))

_OBJ := main.o examples.o rbm.o pcg_basic.o functions.o load_data.o
OBJ := $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: $(SRC)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

rbm: $(OBJ)
	gcc -o $(BIN)/$@ $^ $(CFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ \
	rm -f $(BIN)/* \
	rm -f $(DATADIR)/*.dat
