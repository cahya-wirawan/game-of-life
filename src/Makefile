CXX = g++
NVCC=nvcc
CXXFLAGS = -O3 -Wall -std=c++11
LDFLAGS = 
OMP_FLAGS = -fopenmp
CL_CFLAGS = 
CL_LDFLAGS = 
MT_FLAGS = 
MT_LDFLAGS = 

# Detect OS
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    # macOS specific flags for OpenMP with Clang
    LIBOMP_PREFIX := $(shell brew --prefix libomp)
    CXXFLAGS += -I$(LIBOMP_PREFIX)/include
    LDFLAGS += -L$(LIBOMP_PREFIX)/lib -lomp
    OMP_FLAGS = -Xpreprocessor -fopenmp
    
    # macOS OpenCL
    CL_LDFLAGS = -framework OpenCL
    
    # macOS Metal
    MT_FLAGS = -x objective-c++
    MT_LDFLAGS = -framework Metal -framework Foundation -framework OpenCL
    
    # Targets for macOS (including Metal)
    TARGETS = gol gol-baseline
else
    # Linux OpenCL (assuming standard installation)
    CL_LDFLAGS = -lOpenCL

    # Targets for Linux (excluding Metal)
    TARGETS = gol gol-baseline gol-cuda
endif
OBJECTS = Timing.o

.PHONY: all clean

all: $(TARGETS)

Timing.o: Timing.cpp Timing.h
	$(CXX) $(CXXFLAGS) -c Timing.cpp -o Timing.o

gol: gol.cpp Timing.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OMP_FLAGS) gol.cpp Timing.o -o gol

gol-baseline: gol-baseline.cpp Timing.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OMP_FLAGS) gol-baseline.cpp Timing.o -o gol-baseline

gol-cuda: gol-cuda.cu Timing.o
	$(NVCC) -std=c++17 gol-cuda.cu Timing.o -o gol-cuda

clean:
	rm -f $(TARGETS) $(OBJECTS)
