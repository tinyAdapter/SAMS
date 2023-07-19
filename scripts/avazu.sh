# default
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset avazu --nfeat 1600000 --nfield 22  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 128 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random Default:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_128 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&


# default
# check  moeHidden 256
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset avazu --nfeat 1600000 --nfield 22  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 256 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random MoEHidden_256:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_256 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&

# check  moeHidden 512
nohup ~/anaconda3/bin/python3 main.py \
    --device cuda:1 --log_folder sams_logs --data_dir "/hdd1/sams/data/" \
    --dataset avazu --nfeat 1600000 --nfield 22  --num_labels 1  \
    --K 4 --moe_num_layers 2 --moe_hid_layer_len 512 --sql_nemb 10 \
    --hyper_num_layers 2 --hid_layer_len 128 --data_nemb 10 \
    --dropout 0.3 --alpha 1.7 --max_filter_col 4   \
    --epoch 100 --batch_size 1024 --lr 0.002 \
    --train_dir "Random MoEHidden_512:K_4 MoeLayer_3 hyperLayer_3 MoeHidden_512 HyperHidden_128 sql_emb_10 data_emb_10" > /dev/null 2>&1&