CC = nvcc
PROJECT = main
SRC = main.cu
LIBS = `pkg-config --cflags --libs opencv4`

$(PROJECT): $(SRC)
	$(CC) $(SRC) -o $(PROJECT) $(LIBS) -diag-suppress 611

clean:
	rm -rf main
