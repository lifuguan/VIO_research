import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_
import numpy as np
import torch.nn.functional as F
import math
from transformer import TemporalTransformer, TokenEmbedding


from utils.utils import create_mask, generate_square_subsequent_mask

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

    def forward(self, img, imu):
        v = torch.cat((img[:, :-1], img[:, 1:]), dim=2) # 构建 t->t+1 对
        batch_size = v.size(0)
        seq_len = v.size(1)

        # image CNN
        v = v.view(batch_size * seq_len, v.size(2), v.size(3), v.size(4))
        v = self.encode_image(v)
        v = v.view(batch_size, seq_len, -1)  # (batch, seq_len, fv)
        v = self.visual_head(v)  # (batch, seq_len, 256)
        
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

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

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
                                       num_encoder_layers=3,
                                       num_decoder_layers=3,
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
        src_mask = torch.zeros((fused_feat.shape[0], fused_feat.shape[0]),device=DEVICE).type(torch.bool)
        memory = self.transformer.encoder(pos_fused_feat, src_mask, None)    # [10,16,768]

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
                                       num_encoder_layers=3,
                                       num_decoder_layers=3, dropout=0.1, 
                                       dim_feedforward=512,  batch_first=True)

        self.generator = nn.Linear(self.latent_dim, 6) # 这里是6维
        self.linear = nn.Linear(6, self.latent_dim) 

        self.gt_visibility = opt.gt_visibility
        self.only_encoder = opt.only_encoder
        initialization(self)

    def forward(self, img, imu, is_training=True, selection='gumbel-softmax', history_out = None, gt_pose = None):
        fv, fi = self.Feature_net(img, imu)
        
        device = fv.device
        batch_size, seq_len = fv.shape[0], fv.shape[1] 

        fused_feat = self.fuse_net(fv, fi)

        if self.gt_visibility is True:
            src_mask, tgt_mask = create_mask(src=fused_feat, tgt=gt_pose)
            target = self.linear(gt_pose)

            fused_feat = fused_feat.transpose(1, 0)
            target = target.transpose(1, 0)
        else:
            target = self.linear(torch.ones((seq_len, batch_size, 6), device=device))
            fused_feat = fused_feat.transpose(1, 0)

        pos_fused_feat = self.positional_encoding(fused_feat) # seq = 20, [0:10] = history, [10:20] = current
        pos_target = self.positional_encoding(target)         # seq = 20, [0:10] = history, [10:20] = current

        pos_fused_feat = pos_fused_feat.transpose(1, 0)
        pos_target = pos_target.transpose(1, 0)

        memory = self.transformer.encoder(pos_fused_feat, None, None)    # [10,16,768]
        if self.only_encoder is True:
            pose = self.generator(memory)
            return pose, history_out
        out = self.transformer.decoder(pos_target, memory, history_out=None, tgt_mask=tgt_mask) # [10,16,768]
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
                                       num_encoder_layers=3,
                                       num_decoder_layers=3,
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