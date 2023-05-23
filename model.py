import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_
import numpy as np
import torch.nn.functional as F
import math
from transformer import TemporalTransformer, TokenEmbedding, PositionalEncoding
from utils.utils import pair, pre_IMU, Transformer, ImageCNN, normalize_imagenet, IMUEncoder, SelfAttention, Block, GPT, Encoder
from utils import config
import torchvision.models as models
from PIL import Image
from einops.layers.torch import Rearrange
from torch.nn.init import kaiming_normal_, constant_

from utils.utils import create_mask, generate_square_subsequent_mask
config = config.GlobalConfig()

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, dropout=0):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout)  # , inplace=True)
        )

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )
def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)

def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]
# The inertial encoder for raw imu data
class Inertial_encoder(nn.Module):
    def __init__(self, opt):
        super(Inertial_encoder, self).__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(opt.imu_dropout))
        self.proj = nn.Linear(256 * 1 * 11, opt.i_f_len)

    def forward(self, x):
        # x: (N, seq_len, 11, 6)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        x = x.view(batch_size * seq_len, x.size(2), x.size(3))    # x: (N x seq_len, 11, 6)
        x = self.encoder_conv(x.permute(0, 2, 1))                 # x: (N x seq_len, 64, 11)
        out = self.proj(x.view(x.shape[0], -1))                   # out: (N x seq_len, 256)
        return out.view(batch_size, seq_len, 256)

class flownet(nn.Module):
    def __init__(self, batchNorm=True):
        super(flownet, self).__init__()
        # CNN
        self.batchNorm = batchNorm
        self.conv1 = conv(True, 6, 64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2 = conv(True, 64, 128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3 = conv(True, 128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(True, 256, 256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4 = conv(True, 256, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv4_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv5 = conv(True, 512, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv5_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv6 = conv(True, 512, 1024, kernel_size=3, stride=2, dropout=0.5)
        
        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)
        
        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)
        
        
    def forward(self, v):
        # batch_size, seq_len, img_channel, image_height, image_width = img.shape
        batch_size, img_channel, image_height, image_width = v.shape
        v = self.encode_image(v) #[16,6,64,128]
        # out = []
        out = F.interpolate(v, (image_height,image_width),mode='bilinear') #[16,6,256,512]
        return out
    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        
        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)
        
        return flow2 #[16,2,64,128]
        

class VisualIMUEncoder(nn.Module):
    def __init__(self, opt):
        super(VisualIMUEncoder, self).__init__()
        # CNN
        self.opt = opt
        self.conv1 = conv(True, 6, 64, kernel_size=7, stride=2, dropout=0.2)
        self.conv2 = conv(True, 64, 128, kernel_size=5, stride=2, dropout=0.2)
        self.conv3 = conv(True, 128, 256, kernel_size=5, stride=2, dropout=0.2)
        self.conv3_1 = conv(True, 256, 256, kernel_size=3, stride=1, dropout=0.2)
        self.conv4 = conv(True, 256, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv4_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv5 = conv(True, 512, 512, kernel_size=3, stride=2, dropout=0.2)
        self.conv5_1 = conv(True, 512, 512, kernel_size=3, stride=1, dropout=0.2)
        self.conv6 = conv(True, 512, 1024, kernel_size=3, stride=2, dropout=0.5)
        # Comput the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, opt.img_w, opt.img_h))
        __tmp = self.encode_image(__tmp)

        self.visual_head = nn.Linear(int(np.prod(__tmp.size())), opt.v_f_len)
        self.inertial_encoder = Inertial_encoder(opt)
        # self.patch_height, self.patch_width = pair(opt.patch_size)
        # img_resnet = models.resnet34(pretrained=True)
        # img_resnet = nn.Sequential(*list(img_resnet.children())[:-1])#移除最后一层全连接
        # self.img_resnet34 = img_resnet
        
        # self.out_dim = 768
        # # img_channel = 3
        # imu_patch_dim = img_channel * self.patch_height * self.patch_width
        # self.img_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width),
        #     nn.LayerNorm(imu_patch_dim),
        #     nn.Linear(imu_patch_dim, self.out_dim),
        #     nn.LayerNorm(self.out_dim),
        # )
        # imu_channel = 1
        # imu_patch_dim = imu_channel * self.patch_height * self.patch_width
        # self.img_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width),
        #     nn.LayerNorm(imu_patch_dim),
        #     nn.Linear(imu_patch_dim, self.out_dim),
        #     nn.LayerNorm(self.out_dim),
        # )
        
        # imu_resnet = models.resnet18(pretrained=True)
        # imu_resnet = nn.Sequential(*list(imu_resnet.children())[:-1])#移除最后一层全连接
        # self.imu_resnet18 = imu_resnet

    def forward(self, img, imu):
        # batch_size, seq_len, img_channel, image_height, image_width = img.shape
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2) # 构建 t->t+1 对
        batch_size, seq_len, img_channel, image_height, image_width = v.shape
        # for i in range(seq_len):
        #     img_feature[i] = self.img_resnet34(v[:,i,:,:,:])
        # f = self.img_resnet34(img[:,0,:,:,:])
        
        # #提取视觉特征
        # self.img_resnet34.eval()
        # img_feature = self.img_resnet34(img)
        # patch_dim = img_channel * self.patch_height * self.patch_width
        # #提取imu特征
        # self.imu_resnet18.eval()
        # imu_photo = pre_IMU(imu)
        # imu_feature = self.imu_resnet18(imu_photo)
        # patch_dim = imu_channels * self.patch_height * self.patch_width
        
        # image CNN
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v = self.encode_image(v)
        v = v.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        v = self.visual_head(v)  # (batch, seq_len, 512)
        
        # IMU CNN
        imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
        imu = self.inertial_encoder(imu)
        
        return v, imu

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6


# The fusion module
class FusionModule(nn.Module):
    def __init__(self, opt):
        super(FusionModule, self).__init__()
        self.fuse_method = opt.fuse_method
        self.f_len = opt.i_f_len + opt.v_f_len
        if self.fuse_method == 'soft':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, self.f_len))
        elif self.fuse_method == 'hard':
            self.net = nn.Sequential(
                nn.Linear(self.f_len, 2 * self.f_len))

    def forward(self, v, i):
        if self.fuse_method == 'cat':
            return torch.cat((v, i), -1)
        elif self.fuse_method == 'soft':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            return feat_cat * weights
        elif self.fuse_method == 'hard':
            feat_cat = torch.cat((v, i), -1)
            weights = self.net(feat_cat)
            weights = weights.view(v.shape[0], v.shape[1], self.f_len, 2)
            mask = F.gumbel_softmax(weights, tau=1, hard=True, dim=-1)
            return feat_cat * mask[:, :, :, 0]


# The pose estimation network# The pose estimation network
class Pose_RNN(nn.Module):
    def __init__(self, opt):
        super(Pose_RNN, self).__init__()

        # The main RNN network
        f_len = opt.v_f_len + opt.i_f_len
        self.rnn = nn.LSTM(
            input_size=f_len,
            hidden_size=opt.rnn_hidden_size,
            num_layers=2,
            dropout=opt.rnn_dropout_between,
            batch_first=True)

        self.fuse = FusionModule(opt)

        # The output networks
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
        self.regressor = nn.Sequential(
            nn.Linear(opt.rnn_hidden_size, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 6))

    def forward(self, fv, fv_alter, fi, dec, prev=None):
        if prev is not None:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())
        
        # Select between fv and fv_alter
        v_in = fv * dec[:, :, :1] + fv_alter * dec[:, :, -1:] if fv_alter is not None else fv
        fused = self.fuse(v_in, fi)
        
        out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev)
        out = self.rnn_drop_out(out)
        pose = self.regressor(out)

        hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())
        return pose, hc

class DeepVIO(nn.Module):
    def __init__(self, opt):
        super(DeepVIO, self).__init__()

        self.Feature_net = VisualIMUEncoder(opt)
        self.Pose_net = Pose_RNN(opt)
        self.Policy_net = FusionModule(opt)
        self.opt = opt
        
        initialization(self)

    def forward(self, img, imu, is_training=True, history_out=None, selection='gumbel-softmax', gt_pose=None):

        fv, fi = self.Feature_net(img, imu)
        seq_len = fv.shape[1]
        fv_alter = torch.zeros_like(fv) # zero padding in the paper, can be replaced by other 
        poses = []
        for i in range(seq_len):
            decision = torch.ones(fv.shape[0], 1, 2, device=fv.device)
            if i == 0 and is_training:
                # The first relative pose is estimated by both images and imu by default
                pose, history_out = self.Pose_net(fv[:, i:i+1, :], None, fi[:, i:i+1, :], None, history_out)
            else:
                pose, history_out = self.Pose_net(fv[:, i:i+1, :], fv_alter[:, i:i+1, :], fi[:, i:i+1, :], decision, history_out)
            poses.append(pose)

        poses = torch.cat(poses, dim=1)

        return poses, history_out

class DeepVIO2(nn.Module):
    def __init__(self, opt):
        super(DeepVIO2, self).__init__()

        self.opt = opt
        self.latent_dim = self.opt.v_f_len + self.opt.i_f_len
        self.feature_extractor = VisualIMUEncoder(opt)
        self.fuse_net = FusionModule(opt)
        self.linear = nn.Linear(6, self.latent_dim)    # tgt feature

        self.query_emb = nn.Embedding(20, self.latent_dim)  # 20 和seq_len相关

        self.positional_encoding = PositionalEncoding(emb_size=self.latent_dim, dropout=0.1)
        self.temporal_transformer = TemporalTransformer(opt, batch_first=False)

        self.generator = nn.Linear(self.latent_dim, 6) # 这里是6维

        initialization(self)

    def forward(self, img, imu, is_training=True, selection='gumbel-softmax', history_out = None, gt_pose = None):
        fv, fi = self.feature_extractor(img, imu)
        
        device = fv.device
        batch_size, seq_len = fv.shape[0], fv.shape[1] 

        fused_feat = self.fuse_net(fv, fi)
        fused_feat = fused_feat.transpose(1, 0)
        # query_emb = self.query_emb.weight.unsqueeze(1).repeat(1, batch_size, 1)
        # target = torch.zeros_like(query_emb)
        target = torch.zeros((seq_len, batch_size, self.latent_dim), device=device)

        if is_training:
            pos_fused_feat = self.positional_encoding(fused_feat) # seq = 20, [0:10] = history, [10:20] = current
            pos_target = self.positional_encoding(target)         # seq = 20, [0:10] = history, [10:20] = current

            pos_fused_feat0, pos_fused_feat1 = pos_fused_feat.chunk(chunks=2, dim=0)
            pos_target0, pos_target1 = pos_target.chunk(chunks=2, dim=0)

            out0 = self.temporal_transformer(pos_fused_feat0, pos_target0, history_out = None)
            out1 = self.temporal_transformer(pos_fused_feat1, pos_target1, history_out = out0)
            out = torch.concat([out0, out1], dim=0)

        else:
            _, target = target.chunk(chunks=2, dim=0)
            pad_fused_feat = torch.cat([torch.zeros_like(fused_feat, device=device), fused_feat], dim=0)
            pad_target = torch.cat([torch.zeros_like(target, device=device), target], dim=0)
            
            pos_fused_feat_ = self.positional_encoding(pad_fused_feat) # seq = 20, [0:10] = zero_pad, [10:20] = current
            pos_target_ = self.positional_encoding(pad_target)         # seq = 20, [0:10] = zero_pad, [10:20] = current
            
            _, pos_fused_feat = pos_fused_feat_.chunk(chunks=2, dim=0)
            _, pos_target = pos_target_.chunk(chunks=2, dim=0)
            
            if history_out is not None:   # 第一次在inference中进行迭代，没有history_out
                pad_his_out = torch.cat([history_out, torch.zeros_like(history_out, device=device)], dim=0)
                pos_his_out_ = self.positional_encoding(pad_his_out)       # seq = 20, [0:10] = zero_pad, [10:20] = current
                pos_his_out, _ = pos_his_out_.chunk(chunks=2, dim=0)
                out = self.temporal_transformer(pos_fused_feat, pos_target, history_out = pos_his_out)
            else:
                out = self.temporal_transformer(pos_fused_feat, pos_target, history_out = history_out)

        # 输出出来的out应该是[10,1,768]
        pose = self.generator(out)
        pose = pose.transpose(1, 0) 
        return pose, history_out



def initialization(net):
    #Initilization
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'bias_ih' in name:
                    param.data.fill_(0)
                elif 'bias_hh' in name:
                    param.data.fill_(0)
                    n = param.size(0)
                    start, end = n//4, n//2
                    param.data[start:end].fill_(1.)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


# 这是one by one finished test结果15左右，显然不能用，测试一版原本的transformer
# add tgt_mask, tgt_embedding, change test pipeline
class DeepVIOTransformer(nn.Module):
    def __init__(self, opt):
        super(DeepVIOTransformer, self).__init__()

        self.opt = opt
        self.latent_dim = self.opt.v_f_len + self.opt.i_f_len
        self.Feature_net = VisualIMUEncoder(opt)
        self.fuse_net = FusionModule(opt)

        self.positional_encoding = PositionalEncoding(emb_size=self.latent_dim, dropout=0.1)
        # self.transformer = TemporalTransformer(opt, 
        #                                d_model=self.latent_dim,
        #                                nhead=8,
        #                                num_encoder_layers=3,
        #                                num_decoder_layers=3, dropout=0.1, 
        #                                dim_feedforward=512,  batch_first=True)
        self.transformer = torch.nn.Transformer(d_model=self.latent_dim,
                                       nhead=8,
                                       num_encoder_layers=opt.encoder_layer_num,
                                       num_decoder_layers=opt.decoder_layer_num,
                                       dim_feedforward=512,
                                       dropout=0.1, batch_first=True)

        self.generator = nn.Linear(self.latent_dim, 6) # 这里是6维
        self.linear = nn.Linear(6, self.latent_dim) 
        self.tgt_to_emb = TokenEmbedding(opt.seq_len - 1, self.latent_dim)

        initialization(self)

    def forward(self, img, imu, is_training=True, selection='gumbel-softmax', history_out = None, gt_pose = None, ys=None):
        fv, fi = self.Feature_net(img, imu)
        fused_feat = self.fuse_net(fv, fi)
        DEVICE = fused_feat.device

        fused_feat = fused_feat.transpose(1, 0)
        pos_fused_feat = self.positional_encoding(fused_feat).transpose(1, 0) 
        src_mask = torch.zeros((fused_feat.shape[0], fused_feat.shape[0]), device=DEVICE).type(torch.bool)
        memory = self.transformer.encoder(pos_fused_feat, None, None)    # [10,16,768]

        if is_training is True:
            src_mask, tgt_mask = create_mask(src=fused_feat, tgt=gt_pose)
            # target = self.tgt_to_emb(gt_pose)
            target = self.linear(gt_pose).transpose(1, 0) # [10,16,768]
            pos_target = self.positional_encoding(target).transpose(1, 0)         # seq = 20, [0:10] = history, [10:20] = current

            # out = self.transformer.decoder(pos_target, memory, history_out=None, tgt_mask=None) # [10,16,768]
            out = self.transformer.decoder(pos_target, memory, tgt_mask=tgt_mask) # [10,16,768]
            pose = self.generator(out)  # 输出出来的out应该是[10,1,768]

        if not is_training:
            for i in range(fused_feat.shape[0]):# fused_feat.shape[0] - 1 -> bug
                tgt_mask = generate_square_subsequent_mask(ys.size(0), DEVICE)
                target = self.linear(ys)# seq, batch, feat_size
                target = target.transpose(1, 0)# batch, seq, feat_size
                # out = self.transformer.decoder(target, memory, None, tgt_mask)
                out = self.transformer.decoder(target, memory, tgt_mask=tgt_mask)
                out = out.transpose(0, 1)
                pred_pose = self.generator(out[-1])

                ys = torch.cat([ys, pred_pose.unsqueeze(0)], dim=0)
            # pose = ys
            pose = ys[1:,:,:]
            # history_out = pose[-1].unsqueeze(0)
        return pose, history_out
#针对transfusionodometry的模型
class TransFusionOdom(nn.Module):
    def __init__(self, opt):
        super(TransFusionOdom, self).__init__()

        self.opt = opt
        self.patch_height, self.patch_width = pair(opt.patch_size)
        #
        # img_resnet = models.resnet34(pretrained=True)
        # img_resnet = nn.Sequential(*list(img_resnet.children())[:-1])#移除最后一层全连接
        # self.img_resnet34 = img_resnet
        
        # imu_resnet = models.resnet18(pretrained=True)
        # imu_resnet = nn.Sequential(*list(imu_resnet.children())[:-1])#移除最后一层全连接
        # self.imu_resnet18 = imu_resnet
        img_channel = 6   #光流是2，图片是6
        assert self.opt.img_h % self.patch_height == 0 and self.opt.img_w % self.patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert self.opt.imu_height % self.patch_height == 0 and self.opt.imu_width % self.patch_width == 0, 'IMU2image dimensions must be divisible by the patch size.'
        
        self.img_num_patches = (self.opt.img_h // self.patch_height) * (self.opt.img_w // self.patch_width)
        self.imu_num_patches = (self.opt.imu_height // self.patch_height) * (self.opt.imu_width // self.patch_width)
        img_patch_dim = img_channel * self.patch_height * self.patch_width
        
        self.out_dim = self.opt.out_dim
        imu_channel = 1
        imu_patch_dim = imu_channel * self.patch_height * self.patch_width
        self.img_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width),
            nn.LayerNorm(img_patch_dim),
            nn.Linear(img_patch_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )
        imu_patch_dim = imu_channel * self.patch_height * self.patch_width
        #img采用可学习的
        self.imu_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width),
            nn.LayerNorm(imu_patch_dim),
            nn.Linear(imu_patch_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
        )# self.opt.batch_size
        self.img_pos_embedding = nn.Parameter(torch.randn(16, self.img_num_patches, self.out_dim))
        self.imu_pos_embedding = nn.Parameter(torch.randn(16, self.imu_num_patches, self.out_dim))
        self.dropout = nn.Dropout(p=0.1)
        # 没必要对imu加入位置编码吧
        # self.imu_pos_embedding = PositionalEncoding(emb_size=self.out_dim, dropout=0.1)
        #fusion transformer
        self.transformer_fusion = Transformer(self.out_dim, 1, 8, self.out_dim / 8, self.out_dim, 0.1)
        #inference transformer encoder
        self.gt_visibility = opt.gt_visibility
        self.with_src_mask = opt.with_src_mask
        self.only_encoder = opt.only_encoder
        self.zero_input = opt.zero_input
        self.feature_dim = (self.imu_num_patches + self.img_num_patches) * self.out_dim  #进入transformer的维度
        # self.feature_dim = self.img_num_patches * self.out_dim
        if self.zero_input is not True:
            self.linear = nn.Linear(6, 1024) #这里之前也是self.feature_dim，这里为了测试改为这个
        self.transformer = TemporalTransformer(opt, 
                                       d_model=1024,# 这里之前也是self.feature_dim，这里为了测试改为这个
                                       nhead=8, 
                                       num_encoder_layers=opt.encoder_layer_num, 
                                       num_decoder_layers=opt.decoder_layer_num, dropout=0.1, 
                                       dim_feedforward=512,  batch_first=True)#dim_feedforward可以改，一般大于out_dim
        self.generator = nn.Linear(1024, 6) # 这里是6维
        self.positional_encoding = PositionalEncoding(emb_size=1024, dropout=0.1)
        #上一个人为的改为1024，不用时可以恢复为之前的维度-》emb_size=self.feature_dim
        #计算光流
        self.Feature_net = flownet()#输出[16,2,64,128]
        
        #变成一维特征向量
        # self.covert1D = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', ),
        #     nn.Conv2d(4096, N, kernel_size=1, stride=1, padding=0),
            
        # )
        self.convert = nn.Linear((self.img_num_patches + self.imu_num_patches) * self.out_dim, 1024)
        initialization(self)

    def forward(self, img, imu, is_training=True, selection='gumbel-softmax', history_out = None, gt_pose = None):
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2) # 构建 t->t+1 对
        device = img.device
        # 这部分是用于光流做patch的。
        # flow = []
        # for i in range(v.shape[1]):
        #     flow.append(self.Feature_net(v[:,i,:,:,:]))
        # flow = torch.stack(flow, dim=0).transpose(1,0)
        flow = v
        batch_size, seq_len, img_channel, image_height, image_width = flow.shape
        imu2image = pre_IMU(imu, self.opt.imu_height, self.opt.imu_width)#b,q,h,w
        imu2image = imu2image.unsqueeze(2).to(device)#b,q,c=1,h,w
        img_patch_feature = torch.zeros((batch_size, seq_len, self.img_num_patches, self.out_dim)).to(device)
        imu_patch_feature = torch.zeros((batch_size, seq_len, self.imu_num_patches, self.out_dim)).to(device)
        for i in range(seq_len):
            img_patch_feature[:,i,:,:] = self.img_patch_embedding(flow[:,i,:,:,:])
            img_patch_feature[:,i,:,:] = img_patch_feature[:,i,:,:] + self.img_pos_embedding[:batch_size]
            img_patch_feature[:,i,:,:] = self.dropout(img_patch_feature[:,i,:,:])
            
            imu_patch_feature[:,i,:,:] = self.imu_patch_embedding(imu2image[:batch_size,i,:,:,:])
            imu_patch_feature[:,i,:,:] = imu_patch_feature[:,i,:,:] + self.imu_pos_embedding[:batch_size]
            imu_patch_feature[:,i,:,:] = self.dropout(imu_patch_feature[:batch_size,i,:,:])
        #--这是patch=16的结果--img_feature [16,10,512,768] imu同样维度
        
        all_feature = torch.cat((img_patch_feature, imu_patch_feature), dim=2)
        #去掉imu，再试一下利用率会不会有变化
        # all_feature = img_patch_feature
        out_feature = torch.zeros_like(all_feature).to(device)
        for i in range(seq_len):
            out_feature[:,i,:,:] = self.transformer_fusion(all_feature[:,i,:,:])            
        pre_fused_feat = torch.flatten(out_feature, start_dim=2)
        fused_feat = self.convert(pre_fused_feat)
        #跟以前一样
        src_seq_len = seq_len
        feature_dim = fused_feat.shape[2]
        if self.gt_visibility is True:
            src_mask, tgt_mask = create_mask(src=fused_feat, tgt=gt_pose)
            target = self.linear(gt_pose)
            
            fused_feat = fused_feat.transpose(1, 0)
            target = target.transpose(1, 0)
        else:
            src_mask, tgt_mask = create_mask(src=fused_feat, tgt=gt_pose)
            if self.zero_input:
                target = torch.zeros((seq_len, batch_size, self.feature_dim), device=device)
            else:
                target = self.linear(torch.ones((seq_len, batch_size, 6), device=device))
            fused_feat = fused_feat.transpose(1, 0)

        pos_fused_feat = self.positional_encoding(fused_feat) # seq = 20, [0:10] = history, [10:20] = current
        pos_target = self.positional_encoding(target)         # seq = 20, [0:10] = history, [10:20] = current

        pos_fused_feat = pos_fused_feat.transpose(1, 0)
        pos_target = pos_target.transpose(1, 0)

        src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

        if self.with_src_mask:
            memory = self.transformer.encoder(pos_fused_feat, tgt_mask, None)    # [10,16,768]
        else:
            memory = self.transformer.encoder(pos_fused_feat, src_mask, None)    # [10,16,768]

        if self.only_encoder is True:
            pose = self.generator(memory)
            return pose, history_out
        out = self.transformer.decoder(memory, pos_target, history_out=None, tgt_mask=tgt_mask) # [10,16,768]  decoder中像dino一样用memory做SA
        pose = self.generator(out)  # 输出出来的out应该是[10,1,768]
        return pose, history_out

class TransFusionOdom_CNN(nn.Module):
    def __init__(self, opt):
        super(TransFusionOdom_CNN, self).__init__()

        self.opt = opt
        #初始化resent提特征以及transformer融合
        self.image_imu_encoder = Encoder(config)
        img_channel = 6   #光流是2，图片是6
        
        self.out_dim = self.opt.out_dim
        imu_channel = 1
        self.dropout = nn.Dropout(p=0.1)
        self.positional_encoding = PositionalEncoding(emb_size=1024, dropout=0.1)
        self.transformer = TemporalTransformer(opt, 
                               d_model=1024,# 这里之前也是self.feature_dim，这里为了测试改为这个
                               nhead=8, 
                               num_encoder_layers=opt.encoder_layer_num, 
                               num_decoder_layers=opt.decoder_layer_num, dropout=0.1, 
                               dim_feedforward=512,  batch_first=True)#dim_feedforward可以改，一般大于out_dim
        self.gt_visibility = opt.gt_visibility
        self.with_src_mask = opt.with_src_mask
        self.only_encoder = opt.only_encoder
        self.zero_input = opt.zero_input
        if self.zero_input is not True:
            self.linear = nn.Linear(6, 1024) #这里之前也是self.feature_dim，这里为了测试改为这个
        self.generator = nn.Linear(1024, 6) # 这里是6维
        self.Feature_net = flownet()#输出[16,2,64,128]
        initialization(self)

    def forward(self, img, imu, is_training=True, selection='gumbel-softmax', history_out = None, gt_pose = None):
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2) # 构建 t->t+1 对
        device = img.device
        # 这部分是用于光流做patch的。
        # flow = []
        # for i in range(v.shape[1]):
        #     flow.append(self.Feature_net(v[:,i,:,:,:]))
        # flow = torch.stack(flow, dim=0).transpose(1,0)
        flow = v
        batch_size, seq_len, img_channel, image_height, image_width = flow.shape
        imu2image = pre_IMU(imu, self.opt.imu_height, self.opt.imu_width)#b,q,h,w
        imu2image = imu2image.unsqueeze(2).to(device)#b,q,c=1,h,w
        fused_feat = self.image_imu_encoder(flow.transpose(1,0), imu2image.transpose(1,0))
        #跟以前一样
        src_seq_len = seq_len
        feature_dim = fused_feat.shape[2]
        if self.gt_visibility is True:
            src_mask, tgt_mask = create_mask(src=fused_feat, tgt=gt_pose)
            target = self.linear(gt_pose)
            
            fused_feat = fused_feat.transpose(1, 0)
            target = target.transpose(1, 0)
        else:
            src_mask, tgt_mask = create_mask(src=fused_feat, tgt=gt_pose)
            if self.zero_input:
                target = torch.zeros((seq_len, batch_size, self.feature_dim), device=device)
            else:
                target = self.linear(torch.ones((seq_len, batch_size, 6), device=device))
            fused_feat = fused_feat.transpose(1, 0)

        pos_fused_feat = self.positional_encoding(fused_feat) # seq = 20, [0:10] = history, [10:20] = current
        pos_target = self.positional_encoding(target)         # seq = 20, [0:10] = history, [10:20] = current

        pos_fused_feat = pos_fused_feat.transpose(1, 0)
        pos_target = pos_target.transpose(1, 0)

        src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

        if self.with_src_mask:
            memory = self.transformer.encoder(pos_fused_feat, tgt_mask, None)    # [10,16,768]
        else:
            memory = self.transformer.encoder(pos_fused_feat, src_mask, None)    # [10,16,768]

        if self.only_encoder is True:
            pose = self.generator(memory)
            return pose, history_out
        out = self.transformer.decoder(memory, pos_target, history_out=None, tgt_mask=tgt_mask) # [10,16,768]  decoder中像dino一样用memory做SA
        pose = self.generator(out)  # 输出出来的out应该是[10,1,768]
        return pose, history_out

class DeepVIOVanillaTransformer(nn.Module):
    def __init__(self, opt):
        super(DeepVIOVanillaTransformer, self).__init__()

        self.opt = opt
        self.latent_dim = self.opt.v_f_len + self.opt.i_f_len
        self.Feature_net = VisualIMUEncoder(opt)
        self.fuse_net = FusionModule(opt)

        self.positional_encoding = PositionalEncoding(emb_size=self.latent_dim, dropout=0.1)
        self.transformer = TemporalTransformer(opt, 
                                       d_model=self.latent_dim,
                                       nhead=8, 
                                       num_encoder_layers=opt.encoder_layer_num, 
                                       num_decoder_layers=opt.decoder_layer_num, dropout=0.1, 
                                       dim_feedforward=512,  batch_first=True)

        self.generator = nn.Linear(self.latent_dim, 6) # 这里是6维

        self.gt_visibility = opt.gt_visibility
        self.with_src_mask = opt.with_src_mask
        self.only_encoder = opt.only_encoder
        self.zero_input = opt.zero_input
        if self.zero_input is not True:
            self.linear = nn.Linear(6, self.latent_dim) 
        initialization(self)

    def forward(self, img, imu, is_training=True, selection='gumbel-softmax', history_out = None, gt_pose = None):
        batch_size, seq_len = img.shape[0], img.shape[1]
        fv, fi = self.Feature_net(img, imu)#[16, 11, 3, 256, 512]
        #[16, 101, 6]
        
        device = fv.device
        batch_size, seq_len = fv.shape[0], fv.shape[1]

        fused_feat = self.fuse_net(fv, fi)
        src_seq_len = fused_feat.shape[1]

        if self.gt_visibility is True:
            src_mask, tgt_mask = create_mask(src=fused_feat, tgt=gt_pose)
            target = self.linear(gt_pose)

            fused_feat = fused_feat.transpose(1, 0)
            target = target.transpose(1, 0)
        else:
            src_mask, tgt_mask = create_mask(src=fused_feat, tgt=gt_pose)
            if self.zero_input:
                target = torch.zeros((seq_len, batch_size, 768), device=device)
            else:
                target = self.linear(torch.ones((seq_len, batch_size, 6), device=device))
            fused_feat = fused_feat.transpose(1, 0)

        pos_fused_feat = self.positional_encoding(fused_feat) # seq = 20, [0:10] = history, [10:20] = current
        pos_target = self.positional_encoding(target)         # seq = 20, [0:10] = history, [10:20] = current

        pos_fused_feat = pos_fused_feat.transpose(1, 0)
        pos_target = pos_target.transpose(1, 0)

        src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

        if self.with_src_mask:
            memory = self.transformer.encoder(pos_fused_feat, tgt_mask, None)    # [10,16,768]
        else:
            memory = self.transformer.encoder(pos_fused_feat, src_mask, None)    # [10,16,768]

        if self.only_encoder is True:
            pose = self.generator(memory)
            return pose, history_out
        out = self.transformer.decoder(memory, pos_target, history_out=None, tgt_mask=tgt_mask) # [10,16,768]  decoder中像dino一样用memory做SA
        pose = self.generator(out)  # 输出出来的out应该是[10,1,768]
        return pose, history_out


class DeepVIOOldTransformer(nn.Module):
    def __init__(self, opt):
        super(DeepVIOOldTransformer, self).__init__()

        self.Feature_net = VisualIMUEncoder(opt)
        self.opt = opt
        initialization(self)

        # The main RNN network
        self.d_model = opt.v_f_len + opt.i_f_len
        self.positional_encoding = PositionalEncoding(emb_size=self.d_model, dropout=0.1)
        self.transformer = torch.nn.Transformer(d_model=self.d_model,
                                       nhead=8,
                                       num_encoder_layers=opt.encoder_layer_num,
                                       num_decoder_layers=opt.encoder_layer_num,
                                       dim_feedforward=512,
                                       dropout=0.1)
        self.generator = nn.Linear(self.d_model, 6)#这里是6维
        self.linear = nn.Linear(6, self.d_model)# tgt feature
        self.fuse = FusionModule(opt)

    def forward(self, img, imu, is_training=True, selection='gumbel-softmax', history_out = None, gt_pose = None):
        tgt = gt_pose
        fv, fi = self.Feature_net(img, imu)
        if tgt.dim() == 3:
            tgt = tgt.transpose(1,0)
        if tgt.dim() == 3:
            tgt = tgt.transpose(1,0)
        fused = self.fuse(fv, fi)
        fused = torch.transpose(fused, 1, 0)
        fused = self.positional_encoding(fused)
        if tgt.dim() == 2:
            tgt = tgt.unsqueeze(0)
        tgt = self.linear(tgt)
        tgt = self.positional_encoding(tgt)
        tgt = torch.transpose(tgt, 1, 0)    #[8,10,6]
        memory = self.transformer.encoder(fused, None, None)    # [10,16,768]
        out = self.transformer.decoder(tgt, memory, None, None) # [10,16,768]

        pose = self.generator(out)
        pose = pose.transpose(1,0)
        return pose, history_out