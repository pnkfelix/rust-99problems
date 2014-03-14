default: test-all


test-all: all.bin
	./all.bin

all.bin: *.rs
	rustc -o $@ all.rs
