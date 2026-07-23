#   make                    ./fluid
#   make DEFS=DEBUG         diagnostics: Reynolds/omega, per-solve report, FPS
#   make DEFS=OUTPUT        dump frames/*.txt for src/visualize.py
#   make OPT="-O0 -g"       unoptimized build with symbols
#   make clean				clean up build artifacts

CC     := gcc
OPT    := -O2
DEFS   :=
CFLAGS := -std=gnu11 -Wall -Wextra -Werror -fopenmp
LDLIBS := -lm
SRCS   := $(wildcard src/*.c)

fluid: $(SRCS)
	$(CC) $(CFLAGS) $(OPT) $(addprefix -D,$(DEFS)) -o $@ $(SRCS) $(LDLIBS)

clean:
	rm -f fluid

.PHONY: clean
