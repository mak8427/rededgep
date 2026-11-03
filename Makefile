.PHONY: help native test bench clean install

help:
	@echo "make native   build the optional C kernel"
	@echo "make test     run the unit suite"
	@echo "make bench    run the benchmarks"
	@echo "make clean    remove build artefacts"

native:
	@command -v gcc >/dev/null || { echo "gcc not found"; exit 1; }
	gcc -O3 -march=native -shared -fPIC -o src/rededgep/librededgep.so src/rededgep/_native.c -lm
	@echo "built src/rededgep/librededgep.so"

test:
	python3 -m unittest discover -s tests

bench: native
	python3 benchmarks/bench_core.py

install:
	pip install -e .

clean:
	rm -f src/rededgep/librededgep.so
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
