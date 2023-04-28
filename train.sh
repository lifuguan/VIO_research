python train.py --gpu_ids 0 --batch_size 16 --seq_len 11 --workers 48 --data_dir ./data --experiment_name vanilla_no_gt --model_type vanilla_transformer --pretrain results/vanilla_no_gt/checkpoints/049.pth
python train.py --gpu_ids 1 --batch_size 16 --seq_len 11 --workers 48 --data_dir ./data --experiment_name vanilla_with_gt --model_type vanilla_transformer --gt_visibility
python train.py --gpu_ids 2 --batch_size 16 --seq_len 11 --workers 48 --data_dir ./data --experiment_name original_deepvio --model_type originalDeepVIO 
python train.py --gpu_ids 3 --batch_size 16 --seq_len 11 --workers 48 --data_dir ./data --experiment_name only_encoder --model_type vanilla_transformer --only_encoder


python test.py --gpu_ids 2  --seq_len 11 --data_dir ./data --experiment_name debug --model_type vanilla_transformer  --model results/vanilla/checkpoints/best_4.69_new.pth

################
#train
python train.py --gpu_ids 1 --batch_size 16 --seq_len 11 --workers 48 --data_dir ./data --experiment_name one_by_one_predict2 --model_type transformer_emb
#test
python test.py --gpu_ids 0 --seq_len 11 --data_dir ./data/data --experiment_name test --model_type transformer_emb --model ./results/full_transformer/checkpoints/best_4.72.pth

#2023/4/27  decoder_layer为1，测试效果(跑了1，3两个实验，结果1的好很多)

#2023/4/27晚测试(只需要修改experiment_name和encoder_layer_num, 跑1，3吧。还有就是decoderlayer的数量，建议两个实验都选1，如果卡多3也可以跑一下，这样就是四个实验)
python train.py --gpu_ids 0 --batch_size 16 --seq_len 11 --workers 48 --data_dir ./data --experiment_name debug --model_type vanilla_transformer --decoder_layer_num 3 --encoder_layer_num 3


python train.py --gpu_ids 7 --batch_size 16 --seq_len 11 --workers 48 --data_dir ./data --experiment_name decoder1_with_mask --model_type vanilla_transformer --only_encoder --with_src_mask --encoder_layer_num 1
