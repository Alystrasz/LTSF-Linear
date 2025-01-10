if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi
model_name=NLinear

# Store all experiment results in a common directory
timestamp=$(date +%s)
mkdir ./logs/LongForecasting/$timestamp

model_name=NLinear
array=$(seq 0 10)

for p in $array
do

# Compute power of 2
p=$((2 ** $p))

filename=(logs/LongForecasting/$timestamp/$model_name'_'fS_ETTm1_336_96_keep_$p.log)

echo "Started at" $(date) >$filename
echo "" >>$filename

# ETTm1, univariate results, pred_len= 96 192 336 720
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_336_96 \
  --model $model_name \
  --data ETTm1 \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 1 \
  --des 'Exp' \
  --divide_dataset_size $p \
  --itr 1 --batch_size 8 --learning_rate 0.0001 --feature S >>$filename

#--keep_one_datum_out_of $p \

echo "" >>$filename
echo "Finished at" $(date) >>$filename

done