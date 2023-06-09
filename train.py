import argparse
import os
import torch
import logging
from path import Path
from utils import custom_transform
from dataset.KITTI_dataset import KITTI
from model import DeepVIO, DeepVIO2, DeepVIOVanillaTransformer, DeepVIOOldTransformer, DeepVIOTransformer, TransFusionOdom, TransFusionOdom_CNN
from collections import defaultdict
from utils.kitti_eval import KITTI_tester
import numpy as np
import wandb 
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_dir', type=str, default='./data', help='path to the dataset')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the result')

parser.add_argument('--train_seq', type=list, default=['00', '01', '02', '04', '06', '08', '09'], help='sequences for training')
parser.add_argument('--val_seq', type=list, default=['05', '07', '10'], help='sequences for validation')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--img_w', type=int, default=512, help='image width')
parser.add_argument('--img_h', type=int, default=256, help='image height')
parser.add_argument('--v_f_len', type=int, default=512, help='visual feature length')
parser.add_argument('--i_f_len', type=int, default=256, help='imu feature length')
parser.add_argument('--fuse_method', type=str, default='cat', help='fusion method [cat, soft, hard]')
parser.add_argument('--imu_dropout', type=float, default=0, help='dropout for the IMU encoder')

parser.add_argument('--rnn_hidden_size', type=int, default=1024, help='size of the LSTM latent')
parser.add_argument('--rnn_dropout_out', type=float, default=0.2, help='dropout for the LSTM output layer')
parser.add_argument('--rnn_dropout_between', type=float, default=0.2, help='dropout within LSTM')

parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for the optimizer')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--seq_len', type=int, default=11, help='sequence length for LSTM')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--epochs_warmup', type=int, default=40, help='number of epochs for warmup')
parser.add_argument('--epochs_joint', type=int, default=40, help='number of epochs for joint training')
parser.add_argument('--epochs_fine', type=int, default=20, help='number of epochs for finetuning')
parser.add_argument('--lr_warmup', type=float, default=5e-4, help='learning rate for warming up stage')
parser.add_argument('--lr_joint', type=float, default=5e-5, help='learning rate for joint training stage')
parser.add_argument('--lr_fine', type=float, default=1e-6, help='learning rate for finetuning stage')
parser.add_argument('--eta', type=float, default=0.05, help='exponential decay factor for temperature')
parser.add_argument('--temp_init', type=float, default=5, help='initial temperature for gumbel-softmax')
parser.add_argument('--Lambda', type=float, default=3e-5, help='penalty factor for the visual encoder usage')

parser.add_argument('--experiment_name', type=str, default='transformer_nopolicy', help='experiment name')
parser.add_argument('--optimizer', type=str, default='Adam', help='type of optimizer [Adam, SGD]')

# parser.add_argument('--pretrain_flownet',type=str, default=None, help='wehther to use the pre-trained flownet')
parser.add_argument('--pretrain_flownet',type=str, default='./model_zoo/flownets_bn_EPE2.459.pth.tar', help='wehther to use the pre-trained flownet')
parser.add_argument('--pretrain', type=str, default=None, help='path to the pretrained model')
parser.add_argument('--hflip', default=False, action='store_true', help='whether to use horizonal flipping as augmentation')
parser.add_argument('--color', default=False, action='store_true', help='whether to use color augmentations')

parser.add_argument('--print_frequency', type=int, default=10, help='print frequency for loss values')
parser.add_argument('--weighted', default=False, action='store_true', help='whether to use weighted sum')

parser.add_argument('--patch_size', type=int, default=16, help='patch size')
parser.add_argument('--imu_height', type=int, default=256, help='imu2image height size')
parser.add_argument('--imu_width', type=int, default=512, help='imu2image width size')
parser.add_argument('--model_type', type=str, default='transformer_emb', help='type of optimizer [vanilla_transformer, time_series, transformer_emb, originalDeepVIO]')
parser.add_argument('--gt_visibility', default=False, action='store_true', help='')
parser.add_argument('--decoder_layer_num', default=3, type=int, help='the number of transformer’s decoder layer')
parser.add_argument('--encoder_layer_num', default=3, type=int, help='the number of transformer’s encoder layer')
parser.add_argument('--only_encoder', default=False, action='store_true', help='')
parser.add_argument('--with_src_mask', default=False, action='store_true', help='')
parser.add_argument('--zero_input', default=False, action='store_true', help='')
parser.add_argument('--per_pe', default=False, action='store_true', help='')
parser.add_argument('--cross_first', default=False, action='store_true', help='')
parser.add_argument('--out_dim', type=int, default=64, help='vit each patch feature dim')



args = parser.parse_args()

if args.experiment_name != 'debug':
    wandb.init(
        # set the wandb project where this run will be logged
        entity="vio-research",
        project="VIO Research",
        name=args.experiment_name,
        
        # track hyperparameters and run metadata
        config={
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "lr_warmup": args.lr_warmup,
        "lr_joint": args.lr_joint,
        "lr_fine": args.lr_fine,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "epochs_warmup": args.epochs_warmup,
        "epochs_joint": args.epochs_joint,
        "epochs_fine": args.epochs_fine,
        "model_type": args.model_type,
        "gt_visibility": args.gt_visibility,
        "encoder_layer_num": args.encoder_layer_num,
        "decoder_layer_num": args.decoder_layer_num if args.only_encoder is False else 0,
        "only_encoder": args.only_encoder,
        "with_src_mask": args.with_src_mask,
        "zero_input": args.zero_input,
        "per_pe": args.per_pe,
        "cross_first": args.cross_first,
        "patch_size": args.patch_size,
        "imu_height": args.imu_height,
        "imu_width": args.imu_width,
        "out_dim": args.out_dim
        }
    )

# Set the random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def update_status(ep, args, model):
    if ep < args.epochs_warmup:  # Warmup stage
        lr = args.lr_warmup
        selection = 'gumbel-softmax'
    elif ep >= args.epochs_warmup and ep < args.epochs_warmup + args.epochs_joint: # Joint training stage
        lr = args.lr_joint
        selection = 'gumbel-softmax'
    elif ep >= args.epochs_warmup + args.epochs_joint: # Finetuning stage
        lr = args.lr_fine
        selection = 'gumbel-softmax'
    return lr, selection

def train(model, optimizer, train_loader, selection, logger, ep, p=0.5, weighted=False):
    
    mse_losses = []
    penalties = []
    data_len = len(train_loader)

    pre_tgt = None
    for i, (imgs, imus, gts, rot, weight) in enumerate(train_loader):#the given Numpy array is not writable

        imgs = imgs.cuda().float() #[16, 11, 3, 256, 512]
        imus = imus.cuda().float() #[16, 101, 6]
        gts = gts.cuda().float() #[16, 10, 6]
        weight = weight.cuda().float()

        optimizer.zero_grad()

        poses, _ = model(imgs, imus, is_training=True, selection=selection, history_out = pre_tgt, gt_pose = gts) # [10,16,6]

        if not weighted:
            angle_loss = torch.nn.functional.mse_loss(poses[:,:,:3], gts[:, :, :3])
            translation_loss = torch.nn.functional.mse_loss(poses[:,:,3:], gts[:, :, 3:])
        else:
            weight = weight/weight.sum()
            angle_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,:3] - gts[:, :, :3]) ** 2).mean()
            translation_loss = (weight.unsqueeze(-1).unsqueeze(-1) * (poses[:,:,3:] - gts[:, :, 3:]) ** 2).mean()
        
        pose_loss = 100 * angle_loss + translation_loss        
        loss = pose_loss
        
        loss.backward()
        optimizer.step()
        
        if i % args.print_frequency == 0: 
            message = f'Epoch: {ep}, iters: {i}/{data_len}, angle loss: {angle_loss.item():.6f}, translation loss: {pose_loss.item():.6f}, loss: {loss.item():.6f}'
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"Epoch": ep, "iters": i, "angle loss": angle_loss.item(), "translation loss": translation_loss.item(), "loss": loss.item()})
            logger.info(message)

        mse_losses.append(pose_loss.item())

    return np.mean(mse_losses)


def main():

    # Create Dir
    experiment_dir = Path('./results')
    experiment_dir.mkdir() if not experiment_dir.exists() else 0
    file_dir = experiment_dir.joinpath('{}/'.format(args.experiment_name))
    file_dir.mkdir() if not file_dir.exists() else 0
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir() if not checkpoints_dir.exists() else 0
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir() if not log_dir.exists() else 0
    
    # Create logs
    logger = logging.getLogger(args.experiment_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s.txt'%args.experiment_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)
    
    # Load the dataset
    transform_train = [custom_transform.ToTensor(),
                       custom_transform.Resize((args.img_h, args.img_w))]
    if args.hflip:
        transform_train += [custom_transform.RandomHorizontalFlip()]
    if args.color:
        transform_train += [custom_transform.RandomColorAug()]
    transform_train = custom_transform.Compose(transform_train)

    train_dataset = KITTI(args.data_dir,
                        sequence_length=args.seq_len,
                        train_seqs=args.train_seq,
                        transform=transform_train
                        )
    logger.info('train_dataset: ' + str(train_dataset))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # GPU selections
    str_ids = args.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
    
    # Initialize the tester
    args.seq_len = 11
    tester = KITTI_tester(args)

    # Model initialization
    if args.model_type == 'time_series':
        model = DeepVIO2(args)
    elif args.model_type == 'vanilla_transformer':
        model = DeepVIOVanillaTransformer(args)
    elif args.model_type == 'old_transformer':
        model = DeepVIOOldTransformer(args)
    elif args.model_type == 'originalDeepVIO':
        model = DeepVIO(args)
    elif args.model_type == 'transformer_emb':
        model = DeepVIOTransformer(args)
    elif args.model_type == 'transfusionodom':
        model = TransFusionOdom(args)
    elif args.model_type == 'transfusionodom_cnn':
        model = TransFusionOdom_CNN(args)

    # Continual training or not
    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain), strict=False)
        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')
    
    # Use the pre-trained flownet or not
    if args.pretrain_flownet and args.pretrain is None:
        pretrained_w = torch.load(args.pretrain_flownet, map_location='cpu')
        model_dict = model.Feature_net.state_dict()
        update_dict = {k: v for k, v in pretrained_w['state_dict'].items() if k in model_dict}
        model_dict.update(update_dict)
        model.Feature_net.load_state_dict(model_dict)

    # Feed model to GPU
    model.cuda(gpu_ids[0])
    model = torch.nn.DataParallel(model, device_ids = gpu_ids)

    pretrain = args.pretrain 
    init_epoch = int(pretrain[-7:-4])+1 if args.pretrain is not None else 0    
    
    # Initialize the optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), 
                                     eps=1e-08, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.999), 
                                     eps=1e-8, weight_decay=args.weight_decay)
    
    best = 10000

    for ep in range(init_epoch, args.epochs_warmup+args.epochs_joint+args.epochs_fine):
        
        lr, selection = update_status(ep, args, model)
        optimizer.param_groups[0]['lr'] = lr
        message = f'Epoch: {ep}, lr: {lr}, selection: {selection}'
        print(message)
        logger.info(message)

        model.train()
        avg_pose_loss = train(model, optimizer, train_loader, selection, logger, ep, p=0.5)

        # Save the model after training
        if(os.path.isfile(f'{checkpoints_dir}/{(ep-1):003}.pth')):
            os.remove(f'{checkpoints_dir}/{(ep-1):003}.pth')
        torch.save(model.module.state_dict(), f'{checkpoints_dir}/{ep:003}.pth')
        message = f'Epoch {ep} training finished, pose loss: {avg_pose_loss:.6f}, model saved'
        print(message)
        logger.info(message)

        if ep > args.epochs_warmup+args.epochs_joint or (ep > args.epochs_warmup and ep % 2 == 0):
            # Evaluate the model
            print('Evaluating the model')
            logger.info('Evaluating the model')
            with torch.no_grad(): 
                model.eval()
                errors = tester.eval(model, selection='gumbel-softmax', num_gpu=len(gpu_ids))
        
            t_rel = np.mean([errors[i]['t_rel'] for i in range(len(errors))])
            r_rel = np.mean([errors[i]['r_rel'] for i in range(len(errors))])
            t_rmse = np.mean([errors[i]['t_rmse'] for i in range(len(errors))])
            r_rmse = np.mean([errors[i]['r_rmse'] for i in range(len(errors))])

            if t_rel < best:
                best = t_rel 
                if best < 10:
                    torch.save(model.module.state_dict(), f'{checkpoints_dir}/best_{best:.2f}.pth')

            message = "Epoch {} evaluation Seq. 05 , t_rel: {}, r_rel: {}, t_rmse: {}, r_rmse: {}" .format(ep, round(errors[0]['t_rel'], 4), round(errors[0]['r_rel'], 4), round(errors[0]['t_rmse'], 4), round(errors[0]['r_rmse'], 4))
            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"5. t_rel": round(errors[0]['t_rel'], 4), "5. r_rel": round(errors[0]['r_rel'], 4), "5. t_rmse": round(errors[0]['t_rmse'], 4), "5. r_rmse": round(errors[0]['r_rmse'], 4)})

            message = "Epoch {} evaluation Seq. 07 , t_rel: {}, r_rel: {}, t_rmse: {}, r_rmse: {}" .format(ep, round(errors[1]['t_rel'], 4), round(errors[1]['r_rel'], 4), round(errors[1]['t_rmse'], 4), round(errors[1]['r_rmse'], 4))
            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"7. t_rel": round(errors[1]['t_rel'], 4), "7. r_rel": round(errors[1]['r_rel'], 4), "7. t_rmse": round(errors[1]['t_rmse'], 4), "7. r_rmse": round(errors[1]['r_rmse'], 4)})


            message = "Epoch {} evaluation Seq. 10 , t_rel: {}, r_rel: {}, t_rmse: {}, r_rmse: {}" .format(ep, round(errors[2]['t_rel'], 4), round(errors[2]['r_rel'], 4), round(errors[2]['t_rmse'], 4), round(errors[2]['r_rmse'], 4))
            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"10. t_rel": round(errors[2]['t_rel'], 4), "10. r_rel": round(errors[2]['r_rel'], 4), "10. t_rmse": round(errors[2]['t_rmse'], 4), "10. r_rmse": round(errors[2]['r_rmse'], 4)})

            logger.info(message)
            print(message)
            if args.experiment_name != 'debug':
                wandb.log({"Epoch": ep, "t_rel": round(t_rel,4), "r_rel": round(r_rel,4), "t_rmse": round(t_rmse,4), "r_rmse": round(r_rmse,4), "best t_rel": round(best,4)})
    
    message = f'Training finished, best t_rel: {best:.4f}'
    wandb.finish()
    logger.info(message)
    print(message)

if __name__ == "__main__":
    main()
