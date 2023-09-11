# default
# k_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sams\
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sams_K_4  --report_freq 5


~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 8 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sams_K_8  --report_freq 5



~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 4 --C 1 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir default_vertical_sams_K_1  --report_freq 5


~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --net vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 4 --C 2 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir default_vertical_sams_K_2  --report_freq 5


~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --net vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 4 --C 3 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir default_vertical_sams_K_3  --report_freq 5

~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --net vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 4 --C 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir default_vertical_sams_K_4  --report_freq 5

~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --net dnn\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir "baseline_dnn" 


# -------------------------------------------------------

# SparseMax - Vertical MoE

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 2 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir Sparsemax_vertical_sams_K_2  --report_freq 5


~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 4  --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir Sparsemax_vertical_sams_K_4  --report_freq 5


~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 8 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir default_vertical_sams_K_8  --report_freq 5

~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 16 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir default_vertical_sams_K_16  --report_freq 5

# alpha_2

~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 2 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_K_2_alpha_2  --report_freq 5

~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_K_4_alpha_2  --report_freq 5


~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 8 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_K_8_alpha_2  --report_freq 5

~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 16 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_K_16_alpha_2  --report_freq 5


~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 32 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_K_32_alpha_2  --report_freq 5


~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset adult --nfeat 140 --nfield 13  --num_labels 1  \
    --K 64 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4  \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_K_64_alpha_2  --report_freq 5