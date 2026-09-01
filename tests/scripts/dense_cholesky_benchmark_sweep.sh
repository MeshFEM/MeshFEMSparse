echo 'scalar,size,method,time'
# for size in 1024 2048 4096 8192 16384; do
for size in 128 152 182 216 256 304 362 430 512 608 724 862 1024 1218 1448 1722 2048 2436 2896 3444 4096 4870 5792 6888 8192 9742 11586 13778 16384 19484 23170 27554 32768; do
    for tile_size in 64 128 256 512; do
        # NOTE: this binary is built in `build/tests/`.
        ./benchmark_dense_cholesky 14 $size $tile_size | grep -v '#'
    done
done
