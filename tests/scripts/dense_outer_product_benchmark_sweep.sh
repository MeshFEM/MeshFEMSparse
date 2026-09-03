echo 'scalar,height,rank,method,time'

for height in 128 152 182 216 256 304 362 430 512 608 724 862 1024 1218 1448 1722 2048 2436 2896 3444 4096 4870 5792 6888 8192; do
    for ratio in 0.2 0.275 0.35 0.425 0.5; do
        rank=$(awk -v h="$height" -v r="$ratio" 'BEGIN { printf "%d", 8 * int((h * r + 4) / 8) }')

        for tile_size in 64 128 256 512; do
            ./benchmark_dense_outer_product 14 "$height" "$rank" 100 "$tile_size" | grep -v '#'
        done
    done
done
