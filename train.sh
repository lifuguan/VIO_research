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

python train.py --gpu_ids 5 --batch_size 16 --workers 48 --data_dir ./data --experiment_name encoder3_per_pe --model_type vanilla_transformer --only_encoder --per_pe --encoder_layer_num 3 --decoder_layer_num 3

python train.py --gpu_ids 4 --batch_size 16 --workers 48 --data_dir ./data --experiment_name encoder3_per_pe_with_mask --model_type vanilla_transformer --only_encoder --with_src_mask --zero_input --per_pe --encoder_layer_num 3

python train.py --gpu_ids 7 --batch_size 16 --workers 48 --data_dir ./data --experiment_name encoder1_with_mask --model_type vanilla_transformer --only_encoder --with_src_mask --encoder_layer_num 1

python train.py --gpu_ids 6 --batch_size 16 --workers 48 --data_dir ./data --experiment_name e1_d1_zeroinput --model_type vanilla_transformer --encoder_layer_num 1 --decoder_layer_num 1

python train.py --gpu_ids 7 --workers 48 --experiment_name e1_d1_zero_mask --model_type vanilla_transformer --encoder_layer_num 1 --decoder_layer_num 1 --with_src_mask


python train.py --gpu_ids 5 --workers 48 --experiment_name mask_cross_first --model_type vanilla_transformer --encoder_layer_num 1 --decoder_layer_num 1 --with_src_mask --cross_first

python train.py --gpu_ids 4 --workers 48 --experiment_name zero_mask_cross_first --model_type vanilla_transformer --encoder_layer_num 1 --decoder_layer_num 1 --with_src_mask --cross_first --zero_input

#2023/5/10
#采取like dino的方式，在decoder mask attention部分tgt也改为memory
python train.py --gpu_ids 0 --data_dir ./data/data --seq_len 11 --batch_size 16 --workers 12 --experiment_name tgt=memory --model_type vanilla_transformer --encoder_layer_num 1 --decoder_layer_num 1 --per_pe #--with_src_mask False --cross_first False,其他参数都是false
python test.py --gpu_ids 0 --seq_len 11 --data_dir ./data/data --experiment_name test --model_type vanilla_transformer --model ./results/tgt=memory/checkpoints/071.pth --val_seq 00
#验证1，1是否过拟合？模型复杂能否缓解  在训练集上特别差，在测试集上反而好。。。
python train.py --gpu_ids 0 --data_dir ./data/data --seq_len 11 --batch_size 16 --workers 12 --experiment_name tgt=memory --model_type vanilla_transformer --encoder_layer_num 3 --decoder_layer_num 3 --per_pe #--with_src_mask False --cross_first False,其他参数都是false

#加入vit融合部分
python train.py --gpu_ids 0 --data_dir ./data/data --seq_len 11 --batch_size 16 --workers 8 --experiment_name like_vit_fusion --model_type transfusionodom --encoder_layer_num 1 --decoder_layer_num 1 --only_encoder --patch_size 64 --imu_height 256 --imu_width 512 --out_dim 128 #--with_src_mask False --cross_first False,其他参数都是false
#上一个对照试验，只有一层encoder没有transfusion的融合部分，还是之前的方法
python train.py --gpu_ids 1 --data_dir ./data/data --seq_len 11 --batch_size 16 --workers 8 --experiment_name only_encoder_no_transfusion --model_type vanilla_transformer --encoder_layer_num 1 --decoder_layer_num 1 --only_encoder --patch_size 0 --imu_height 0 --imu_width 0 --out_dim 0 #--with_src_mask False --cross_first False,其他参数都是false
