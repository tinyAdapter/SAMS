# census
~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net afn\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log_baseline\
    --dataset census --nfeat 540 --nfield 41  \
    --moe_hid_layer_len 32 --nhid 16 --data_nemb 10 \
    --dropout 0 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir "afn_hidden32" 


#credit
~/anaconda3/bin/python3 main.py \
    --device cuda:2 --net afn\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log_baseline\
    --dataset credit --nfeat 350 --nfield 23  \
    --moe_hid_layer_len 16 --nhid 16 --data_nemb 10 \
    --dropout 0 \
    --epoch 30 --batch_size 1024 --lr 0.002 \
    --train_dir "afn_hidden32" 

# diabetes
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net afn\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log_baseline\
    --dataset diabetes --nfeat 850 --nfield 48 \
    --moe_hid_layer_len 32 --nhid 32 --data_nemb 10 \
    --dropout 0.2 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir "afn_hidden64" 

# hcdr
~/anaconda3/bin/python3 main.py \
    --device cuda:0 --net afn\
    --data_dir "/hdd1/sams/data/" --exp /hdd1/sams/tensor_log_baseline\
    --dataset hcdr --nfeat 550 --nfield 69  \
    --moe_hid_layer_len 32 --nhid 32 --data_nemb 10 \
    --dropout 0 \
    --epoch 50 --batch_size 1024 --lr 0.002 \
    --train_dir "afn_hidden64" 