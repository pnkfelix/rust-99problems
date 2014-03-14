default: test-all

test-all: all.bin
	./all.bin --test

all.bin: *.rs
	rustc -o $@ --test all.rs
