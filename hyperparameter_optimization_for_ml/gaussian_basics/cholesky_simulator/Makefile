CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -O2
TARGET := cholesky_simulator.exe
SOURCES := backend/Matrix.cpp backend/Cholesky.cpp frontend/main.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SOURCES) backend/Matrix.hpp backend/Cholesky.hpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)
