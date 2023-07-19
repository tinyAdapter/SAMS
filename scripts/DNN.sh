

# Frappe Dataset  DNN MoE w/o Gated Network
~/anaconda3/bin/python3 main.py \
    --net="moe_dnn" \
    --device=cuda:3 --log_folder=sams_logs --data_dir="/hdd1/sams/data/" \
    --dataset=frappe --nfeat=5500 --nfield=10  --num_labels=1  \
    --K=4 --moe_num_layers=2 --hid_layer_len=128 --data_nemb=10 \
    --dropout=0.2 --max_filter_col=0  \
    --epoch=200 --batch_size=1024 --lr=0.002 --save_best_model=False\
    --exp="/hdd1/sams/tensor_log" \
    --train_dir="Default MoE DNN w/o GatedNetwork"


# Frappe Dataset Only DNN
~/anaconda3/bin/python3 main.py \
    --net="dnn" \
    --device=cuda:1 --log_folder=sams_logs --data_dir="/hdd1/sams/data/" \
    --dataset=frappe --nfeat=5500 --nfield=10  --num_labels=1  \
    --K=1 --moe_num_layers=2 --hid_layer_len=128 --data_nemb=10 \
    --dropout=0.2 --max_filter_col=0  \
    --epoch=200 --batch_size=1024 --lr=0.002 --save_best_model=False\
    --exp="/hdd1/sams/tensor_log" \
    --train_dir="Default DNN"


# uci DNN MoE w/o Gated Network
nohup ~/anaconda3/bin/python3 main.py \
    --net="moe_dnn" \
    --device=cuda:3 --log_folder=sams_logs --data_dir="/hdd1/sams/data/" \
    --dataset=uci_diabetes --nfeat=369 --nfield=43  --num_labels=1  \
    --K=4 --moe_num_layers=2 --hid_layer_len=128 --data_nemb=10 \
    --dropout=0.2 --max_filter_col=0  \
    --epoch=100 --batch_size=1024 --lr=0.002 --save_best_model=False\
    --exp="/hdd1/sams/tensor_log" \
    --train_dir="Default MoE DNN w/o GatedNetwork" > /dev/null 2>&1&


# uci Dataset / Only DNN
nohup ~/anaconda3/bin/python3 main.py \
    --net="moe_dnn" \
    --device=cuda:4 --log_folder=sams_logs --data_dir="/hdd1/sams/data/" \
    --dataset=uci_diabetes --nfeat=369 --nfield=43  --num_labels=1  \
    --K=4 --moe_num_layers=2 --hid_layer_len=128 --data_nemb=10 \
    --dropout=0.2 --max_filter_col=0  \
    --epoch=100 --batch_size=1024 --lr=0.002 --save_best_model=False\
    --exp="/hdd1/sams/tensor_log" \
    --train_dir="Default DNN" > /dev/null 2>&1&

# uci 
nohup ~/anaconda3/bin/python3 main.py \
    --net="moe_dnn" \
    --device=cuda:5 --log_folder=sams_logs --data_dir="/hdd1/sams/data/" \
    --dataset=uci_diabetes --nfeat=369 --nfield=43  --num_labels=1  \
    --K=8 --moe_num_layers=2 --hid_layer_len=128 --data_nemb=10 \
    --dropout=0.2 --max_filter_col=0  \
    --epoch=100 --batch_size=1024 --lr=0.002 --save_best_model=False\
    --exp="/hdd1/sams/tensor_log" \
    --train_dir="Default MoE DNN without GatedNetwork [K = 8]" > /dev/null 2>&1&



# uci Dataset / Only DNN
nohup ~/anaconda3/bin/python3 main.py \
    --net="moe_dnn" \
    --device=cuda:6 --log_folder=sams_logs --data_dir="/hdd1/sams/data/" \
    --dataset=uci_diabetes --nfeat=369 --nfield=43  --num_labels=1  \
    --K=8 --moe_num_layers=2 --hid_layer_len=128 --data_nemb=10 \
    --dropout=0.2 --max_filter_col=0  \
    --epoch=100 --batch_size=1024 --lr=0.002 --save_best_model=False\
    --exp="/hdd1/sams/tensor_log" \
    --train_dir="Default DNN [K = 8]" > /dev/null 2>&1&