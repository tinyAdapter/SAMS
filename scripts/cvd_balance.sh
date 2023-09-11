


~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 32 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 --beta 0.001\
    --train_dir sparsemax_vertical_sams_balance_K32_alpha_2  --report_freq 10



~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 8 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 --beta 0.001\
    --train_dir sparsemax_vertical_sams_balance_K8_alpha_2  --report_freq 10




~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 16 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 --beta 0.005\
    --train_dir sparsemax_vertical_sams_balance_K16_alpha_2  --report_freq 10



~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 16 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 3 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 --beta 0.005\
    --train_dir sparsemax_vertical_sams_balance_K16_alpha_3  --report_freq 10


~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 16 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 3 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 --beta 0.01\
    --train_dir sparsemax_vertical_sams_balance_K16_alpha_2-5_beta_001  --report_freq 10