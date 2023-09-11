# default
# k_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir Default  --report_freq 5



~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 1 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir Default_K1  --report_freq 5




~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --net dnn\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir "baseline_dnn_1" 

~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --net dnn\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.001 \
    --train_dir "baseline_dnn" 

# -------------------------------------------------------
# sparse MoE : just choose C expert

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 4 --C 2 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir default_vertical_sams_K_2  --report_freq 10

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 4 --C 3 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir default_vertical_sams_K_3  --report_freq 10

~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --net vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 4 --C 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir default_vertical_sams_K_4  --report_freq 10

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir default_vertical_sams  --report_freq 10

## K = 16
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 16 --C 1 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir default_vertical_sams_K16_C1  --report_freq 10


~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 16 --C 2 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir default_vertical_sams_K16_C2  --report_freq 10

~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --net vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 16 --C 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir default_vertical_sams_K16_C4  --report_freq 10

~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --net vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 16 --C 8 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir default_vertical_sams_K16_C8  --report_freq 10

# ------------------ split MoE and predict layer ----------------- #
# v4

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net vertical_predict_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 16 --C 1 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir v4_vertical_MoE_predict_K16_C1  --report_freq 10


~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net vertical_predict_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 16 --C 2 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir v4_vertical_MoE_predict_K16_C2  --report_freq 10

~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --net vertical_predict_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 16 --C 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir v4_vertical_MoE_predict_K16_C4  --report_freq 10

~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --net vertical_predict_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 16 --C 8 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir v4_vertical_MoE_predict_K16_C8  --report_freq 10


# K=32 C = 16
~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --net vertical_predict_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 16 --C 8 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir v4_vertical_MoE_predict_K32_C16  --report_freq 10
# -------------------------------------------------------
# Sparsemax  Vertical MoE

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 2 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_K_2  --report_freq 10

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_K_4  --report_freq 10

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 8 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_K_8  --report_freq 10


~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 16 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_K_16  --report_freq 10



# ---alpha = 2.0

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 2 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_K_2_alpha_2  --report_freq 10


~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_K_4_alpha_2  --report_freq 10

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 8 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_K_8_alpha_2  --report_freq 10

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 16 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_K_16_alpha_2  --report_freq 10


~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 32 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_K_32_alpha_2  --report_freq 10


~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 64 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_K_64_alpha_2  --report_freq 10



#----------------------------Version 3 -------------------------------

# Vertical MoE + Predict Layer

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_plus_predict_layer \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 2 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir verticalMoE+Predict_layer_K_2_alpha_2  --report_freq 10


~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_plus_predict_layer \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir verticalMoE+Predict_layer_K_4_alpha_2  --report_freq 10

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_plus_predict_layer \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 8 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir verticalMoE+Predict_layer_K_8_alpha_2  --report_freq 10

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_plus_predict_layer \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 16 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir verticalMoE+Predict_layer_K_16_alpha_2  --report_freq 10


~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_plus_predict_layer \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 32 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir verticalMoE+Predict_layer_K_32_alpha_2  --report_freq 10


~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --net sparsemax_plus_predict_layer \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 64 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir verticalMoE+Predict_layer_K_64_alpha_2  --report_freq 10



# -------------------------------- V5


# importance loss

~/anaconda3/bin/python3 main.py \
    --device cuda:0 --log_folder sams_logs --net sparsemax_vertical_sams \
    --data_dir /hdd1/sams/data/ --exp /hdd1/sams/tensor_log\
    --dataset cvd --nfeat 110 --nfield 11  --num_labels 1  \
    --K 16 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2 --max_filter_col 4   \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir sparsemax_vertical_sams_balance_K16_alpha_2  --report_freq 10