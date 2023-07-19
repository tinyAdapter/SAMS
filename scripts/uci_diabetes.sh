# default
# k_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:6 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random Default:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# Check data_emb 30
# k_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_30
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:7 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 30 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 30 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random Check#data_emb_30:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_30 data_emb_30" > /dev/null 2>&1&


# Check data_emb 50
# k_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_30
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:5 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 50 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 50 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random Check#data_emb_50:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_50 data_emb_50" > /dev/null 2>&1&


# Check K_1
# k_1 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:3 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 1 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random Check#K_1:K_1 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# Check K_8
# k_8 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 8 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random Check#K_8:K_8 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# Check K_16
# k_16 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 16 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random Check#K_16:K_16 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# Check MoEHidden 64
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 64 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random MoEHidden_64:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_64 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# Check MoEHidden 256
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:2 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 256 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random MoEHidden_256:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_256 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&

# Check MoEHidden 512
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:3 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 512 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random MoEHidden_512:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_512 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# Check MoE Layer 1 (2 layer)
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:4 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 4 --moe_num_layers 1 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random MoELayer_2:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# Check MoE Layer 3 (4 layer)
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:5 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 4 --moe_num_layers 3 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random MoELayer_4:K_4 MoeLayer_3 hyperLayer_2 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&

# Check MoE Layer 4 (5 layer)
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:6 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 4 --moe_num_layers 4 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random MoELayer_5:K_4 MoeLayer_3 hyperLayer_5 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&

# Check MoE Layer 5 (6 layer)
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:6 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 4 --moe_num_layers 5 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random MoELayer_6:K_4 MoeLayer_3 hyperLayer_5 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# Check Sparse Alpha = 1
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:7 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random #Alpha 1:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# Check Sparse Alpha = 1.3
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:7 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.3 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random #Alpha 1.3:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# Check Sparse Alpha = 2.0
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:4 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2.0 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random #Alpha 2.0:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# Check Sparse Alpha = 2.3
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:5 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2.3 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random #Alpha 2.3:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# Check Sparse Alpha = 2.7
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:6 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset uci_diabetes --nfeat 369 --nfield 43  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 2.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random #Alpha 2.7:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&
