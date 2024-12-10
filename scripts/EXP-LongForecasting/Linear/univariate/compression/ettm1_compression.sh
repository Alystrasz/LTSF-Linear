#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

# Store all experiment results in a common directory
timestamp=$(date +%s)
mkdir ./logs/LongForecasting/$timestamp

model_name=NLinear

# Run compression for multiple powers of 2
if test "$#" -ne 1
then
    array=$(seq 0 20)
    debug=false
else
    echo "Multiple parameters detected, debugging mode enabled."
    value=("${@: -1}")
    array=$(seq $value $value)
    debug=true
    echo "Compressing data set with value $value."
fi

for p in $array
do
    ratio=$((2 ** $p))
    filename=(logs/LongForecasting/$timestamp/$model_name'_'fS_ETTm1_336_720_ratio$ratio.log)

    echo "Started at" $(date) >$filename
    echo "" >>$filename

    if $debug; then
        python -u run_longExp.py \
        --is_training 1 \
        --root_path ./dataset/ \
        --data_path ETTm1.csv \
        --model_id ETTm1_336_720 \
        --model $model_name \
        --data ETTm1_compression \
        --seq_len 336 \
        --pred_len 720 \
        --enc_in 1 \
        --des 'Exp' \
        --enable_compression \
        --preserve_ratio $ratio \
        --itr 1 --batch_size 8 --learning_rate 0.0001 --feature S
    else
        python -u run_longExp.py \
            --is_training 1 \
            --root_path ./dataset/ \
            --data_path ETTm1.csv \
            --model_id ETTm1_336_720 \
            --model $model_name \
            --data ETTm1_compression \
            --seq_len 336 \
            --pred_len 720 \
            --enc_in 1 \
            --des 'Exp' \
            --enable_compression \
            --preserve_ratio $ratio \
            --itr 1 --batch_size 8 --learning_rate 0.0001 --feature S >>$filename
    fi

    echo "" >>$filename
    echo "Finished at" $(date) >>$filename
done
