CC = gcc

# Compiler flags
CFLAGS = -Wall -g3 -std=c99

# Target executable name
TARGET = main_program

# Automatically find all .c files in the current directory
SRCS = $(wildcard *.c)

# Object files (derived from source files)
OBJS = $(SRCS:.c=.o)

# Command to remove files
RM = rm -f

# Default target
all: $(TARGET)

# Rule to link the object files into the final executable
$(TARGET): $(OBJS)
	echo "Linking and producing the final application ..."
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) -lm

# Rule to compile source files into object files
%.o: %.c
	echo "Compiling source file $< into object file $@ ..."
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule to remove generated files
clean:
	echo "Cleaning the project ..."
	$(RM) $(TARGET) $(OBJS)