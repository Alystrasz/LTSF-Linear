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

# Simple run
#
# value=("${@: -1}")
# echo "Compressing data set using FLI with tolerated error: $value."

# filename=(logs/LongForecasting/$timestamp/$model_name'_'fS_ETTm1_336_720_fli-$value.log)

# echo "Started at" $(date) >$filename
# echo "" >>$filename

# python -u run_longExp.py \
#     --is_training 1 \
#     --root_path ./dataset/ \
#     --data_path ETTm1.csv \
#     --model_id ETTm1_336_720 \
#     --model $model_name \
#     --data ETTm1_compression \
#     --seq_len 336 \
#     --pred_len 720 \
#     --enc_in 1 \
#     --des 'Exp' \
#     --enable_compression \
#     --tolerated_error $value \
#     --itr 1 --batch_size 8 --learning_rate 0.0001 --feature S >>$filename

# echo "" >>$filename
# echo "Finished at" $(date) >>$filename


# Batch run
#
array=$(seq 0 15)
for p in $array
do
    ratio=$(echo "scale=2;0.001*2^$p" | bc)
    echo $ratio
    filename=(logs/LongForecasting/$timestamp/$model_name'_'fS_ETTm1_336_720_error$ratio.log)

    echo "Started at" $(date) >$filename
    echo "" >>$filename

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
        --tolerated_error $ratio \
        --itr 1 --batch_size 8 --learning_rate 0.0001 --feature S >>$filename
    
    echo "" >>$filename
    echo "Finished at" $(date) >>$filename
done
