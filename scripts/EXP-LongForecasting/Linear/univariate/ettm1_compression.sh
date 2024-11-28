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
  --preserve_ratio 2 \
  --itr 1 --batch_size 8 --learning_rate 0.0001 --feature S >logs/LongForecasting/$model_name'_'fS_ETTm1_336_720.log
