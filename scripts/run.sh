


# CPU
python main.py --device cpu --log_folder sams_logs --K 4 --moe_num_layers 4 --moe_hid_layer_len 10 --num_layers 4 --hid_layer_len 10 --data_nemb 10 --sql_nemb 10 --dropout 0.0 --max_filter_col 4 --nfeat 369 --nfield 43 --epoch 3 --batch_size 128 --lr 0.002 --iter_per_epoch 10 --report_freq 30 --data_dir "/hdd1/sams/data/" --dataset uci_diabetes --num_labels 1


# GPU
python main.py --device cuda:0 --log_folder sams_logs --K 4 --moe_num_layers 4 --moe_hid_layer_len 10 --num_layers 4 --hid_layer_len 10 --data_nemb 10 --sql_nemb 10 --dropout 0.0 --alpha 1.7 --max_filter_col 4 --nfeat 369 --nfield 43 --epoch 40 --batch_size 1024 --lr 0.002 --iter_per_epoch 200 --report_freq 30 --data_dir "/hdd1/sams/data/" --dataset uci_diabetes --num_labels 1