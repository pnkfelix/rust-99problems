default: test-all

test-all: all.bin
	./all.bin --test --bench

all.bin: *.rs Makefile
	rustc -o $@ --test all.rs
