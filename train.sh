# time series
python train.py --gpu_ids 2 --experiment_name try_have_time_series4 --model time_series --batch_size 16 --workers 48 --data_dir ./data 
# vanilla transformer
python train.py --gpu_ids 3 --experiment_name vanilla_transformer --model vanilla_transformer --batch_size 16 --workers 48 --data_dir ./data 


#----2023/4/8
python train.py --gpu_ids 2 --experiment_name try_have_time_series4 --transformer --seq2seq --batch_size 16 --workers 48 --data_dir ./data --pretrain results/try_have_time_series4/checkpoints/081.pth
#接着080.pth继续train
python train.py --gpu_ids 1 --data_dir ./data --experiment_name full_transformer --batch_size 16 --pretrain ./results/full_transformer/checkpoints/080.pth
python test.py --gpu_ids 2 --data_dir ./data --experiment_name debug --transformer --seq2seq --model results/try_have_time_series4/checkpoints/065.pth

#修改为2023/4/12开会时所说的加时序的结果，跑一下看看结果！！！！try_have_time_series  in test.py ; in train.py is_first is all true 
python train.py --gpu_ids 4,5 --data_dir ./data --experiment_name try_have_time_series --batch_size 16 --transformer --seq2seq --pretrain results/try_have_time_series/checkpoints/081.pth

#调参2023/4/14
python train.py --gpu_ids 0 --data_dir ./data --pretrain_flownet ./model_zoo/flownets_bn_EPE2.459.pth.tar --optimizer AdamW --experiment_name adapt_para_optimizer_AdamW --batch_size 16 --transformer --seq2seq

python train.py --gpu_ids 0 --data_dir ./data --pretrain_flownet ./model_zoo/flownets_bn_EPE2.459.pth.tar --optimizer Adam --experiment_name adapt_para_optimizer_Adam_lr_fine_2e-6 --batch_size 16 --transformer --seq2seq --lr_fine 2e-6
python train.py --gpu_ids 0 --data_dir ./data --pretrain_flownet ./model_zoo/flownets_bn_EPE2.459.pth.tar --optimizer Adam --experiment_name adapt_para_optimizer_Adam_lr_fine_5e-7 --batch_size 16 --transformer --seq2seq --lr_fine 5e-7
python train.py --gpu_ids 0 --data_dir ./data --pretrain_flownet ./model_zoo/flownets_bn_EPE2.459.pth.tar --optimizer Adam --experiment_name adapt_para_optimizer_Adam_WD_1e-5 --batch_size 16 --transformer --seq2seq --weight_decay 1e-5
python train.py --gpu_ids 0 --data_dir ./data --pretrain_flownet ./model_zoo/flownets_bn_EPE2.459.pth.tar --optimizer Adam --experiment_name adapt_para_optimizer_Adam_WD_ --batch_size 16 --transformer --seq2seq --weight_decay 2.5e-6
