
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net sparsemax_vertical_sams --expert dnn\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  \
    --moe_hid_layer_len 32 --data_nemb 10  \
    --hid_layer_len 32 --sql_nemb 5\
    --dropout 0.3   \
    --K 16 --alpha 2.5 --max_filter_col 8\
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --seed 2023 --beta=0.005 \
    --train_dir dnn_K16_alpha2-5  --report_freq 5