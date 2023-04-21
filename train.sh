python train.py --gpu_ids 0 --batch_size 16 --seq_len 11 --workers 48 --data_dir ./data --experiment_name vanilla_no_gt --model_type vanilla_transformer
python train.py --gpu_ids 1 --batch_size 16 --seq_len 11 --workers 48 --data_dir ./data --experiment_name vanilla_with_gt --model_type vanilla_transformer --gt_visibility


python test.py --gpu_ids 0  --seq_len 11 --data_dir ./data --experiment_name vanilla_no_gt --model_type vanilla_transformer --model results/vanilla/checkpoints/best_4.69_new.pth