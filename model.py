import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_
import numpy as np
import torch.nn.functional as F
import math
from torch.nn import Transformer

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

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
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
class Fusion_module(nn.Module):
    def __init__(self, opt):
        super(Fusion_module, self).__init__()
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


# The pose estimation network
class Pose_RNN(nn.Module):
    def __init__(self, opt, transformer = False):
        super(Pose_RNN, self).__init__()

        # The main RNN network
        f_len = opt.v_f_len + opt.i_f_len
        if transformer == False:
            self.rnn = nn.LSTM(
                input_size=f_len,
                hidden_size=opt.rnn_hidden_size,
                num_layers=2,
                dropout=opt.rnn_dropout_between,
                batch_first=True)
            self.regressor = nn.Sequential(
                nn.Linear(opt.rnn_hidden_size, 128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(128, 6))
        elif transformer == True:
            self.decode_layer = nn.TransformerDecoderLayer(d_model=f_len, nhead=8, dropout=0.1, batch_first=True)
            self.encode_layer = nn.TransformerEncoderLayer(d_model=f_len, nhead=8, dropout=0.1, batch_first=True)
            self.regressor = nn.Sequential(
                nn.Linear(f_len, 128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(128, 6))
        self.transformer = transformer


        self.fuse = Fusion_module(opt)

        # The output networks
        self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
       

    def forward(self, fv, fv_alter, fi, dec, prev=None):
        if prev is not None and self.transformer is False:
            prev = (prev[0].transpose(1, 0).contiguous(), prev[1].transpose(1, 0).contiguous())
        
        # Select between fv and fv_alter
        v_in = fv * dec[:, :, :1] + fv_alter * dec[:, :, -1:] if fv_alter is not None else fv
        fused = self.fuse(v_in, fi)
        
        if self.transformer == False:
            out, hc = self.rnn(fused) if prev is None else self.rnn(fused, prev) # hc = (hn, cn)
        elif self.transformer == True:
            out = self.encode_layer(fused) if prev is None else self.decode_layer(fused, prev)
            hc = out.clone()

        out = self.rnn_drop_out(out)
        pose = self.regressor(out)

        if self.transformer == False:
            hc = (hc[0].transpose(1, 0).contiguous(), hc[1].transpose(1, 0).contiguous())

        return pose, hc



class DeepVIO(nn.Module):
    def __init__(self, opt):
        super(DeepVIO, self).__init__()

        self.Feature_net = Encoder(opt)
        self.Pose_net = Pose_RNN(opt, opt.transformer)
        self.opt = opt
        self.transformer = opt.transformer
        self.dense_connect = opt.dense_connect
        initialization(self)

    def forward(self, img, imu, is_first=True, hc=None, temp=5, selection='gumbel-softmax', p=0.5):

        fv, fi = self.Feature_net(img, imu)
        batch_size = fv.shape[0]
        seq_len = fv.shape[1]

        poses, decisions, logits= [], [], []
        if hc is None:
            hidden = torch.zeros(batch_size, self.opt.rnn_hidden_size).to(fv.device) 
        elif self.transformer is True:
            hidden = hc
        else:
            hidden = hc[0].contiguous()[:, -1, :]
        fv_alter = torch.zeros_like(fv) # zero padding in the paper, can be replaced by other 
        
        for i in range(seq_len):
            if i == 0 and is_first:
                # The first relative pose is estimated by both images and imu by default
                pose, hc = self.Pose_net(fv[:, i:i+1, :], None, fi[:, i:i+1, :], None, hc)
            else:
                if selection == 'gumbel-softmax':
                    # Otherwise, sample the decision from the policy network
                    if self.transformer is True:
                        p_in = torch.cat((fi[:, i, :], hidden[:, -1, :]), -1)
                    else:
                        p_in = torch.cat((fi[:, i, :], hidden), -1)
                    logit, decision = self.Policy_net(p_in.detach(), temp)
                    decision = decision.unsqueeze(1)
                    logit = logit.unsqueeze(1)
                    pose, hc = self.Pose_net(fv[:, i:i+1, :], fv_alter[:, i:i+1, :], fi[:, i:i+1, :], decision, hc)
                    decisions.append(decision)
                    logits.append(logit)
                elif selection == 'random':
                    decision = (torch.rand(fv.shape[0], 1, 2) < p).float()
                    decision[:,:,1] = 1-decision[:,:,0]
                    decision = decision.to(fv.device)
                    logit = 0.5*torch.ones((fv.shape[0], 1, 2)).to(fv.device)
                    pose, hc = self.Pose_net(fv[:, i:i+1, :], fv_alter[:, i:i+1, :], fi[:, i:i+1, :], decision, hc)
                    decisions.append(decision)
                    logits.append(logit)
            poses.append(pose)

            if self.dense_connect is True and self.transformer is True: 
                if i ==0 and is_first:
                    hidden = hc
                else:
                    hidden = torch.cat([hidden, hc], dim=1)
            else:
                hidden = hc[0].contiguous()[:, -1, :] if self.transformer is False else hc

        poses = torch.cat(poses, dim=1)
        decisions = torch.cat(decisions, dim=1)
        logits = torch.cat(logits, dim=1)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        return poses, decisions, probs, hc


# class PoseTransformer(nn.Module):
#     def __init__(self, opt):
#         super(PoseTransformer, self).__init__()

#         # The main RNN network
#         d_model = opt.v_f_len + opt.i_f_len
#         self.positional_encoding = PositionalEncoding(emb_size=d_model, dropout=0.1)
#         self.decode_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8, dropout=0.1, batch_first=True)
#         self.encode_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=0.1, batch_first=True)
#         self.regressor = nn.Sequential(
#             nn.Linear(d_model, 128),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Linear(128, 6))

#         self.fuse = Fusion_module(opt)

#         # The output networks
#         self.rnn_drop_out = nn.Dropout(opt.rnn_dropout_out)
       

#     def forward(self, fv, fv_alter, fi, dec, prev=None):
#         # Select between fv and fv_alter
#         v_in = fv * dec[:, :, :1] + fv_alter * dec[:, :, -1:] if fv_alter is not None else fv
#         fused = self.fuse(v_in, fi)
        
#         if prev is None:
#             out = self.encode_layer(fused)  
#         else:
#             prev = self.positional_encoding(prev)
#             out = self.decode_layer(fused, prev)
#         hc = out.clone()

#         out = self.rnn_drop_out(out)
#         pose = self.regressor(out)

#         return pose, hc

class PoseTransformer(nn.Module):
    def __init__(self, opt):
        super(PoseTransformer, self).__init__()

        # The main RNN network
        self.d_model = opt.v_f_len + opt.i_f_len
        self.positional_encoding = PositionalEncoding(emb_size=self.d_model, dropout=0.1)
        self.transformer = Transformer(d_model=self.d_model,
                                       nhead=8,
                                       num_encoder_layers=3,
                                       num_decoder_layers=3,
                                       dim_feedforward=512,
                                       dropout=0.1)
        self.generator = nn.Linear(self.d_model, 6)#这里是6维
        self.linear = nn.Linear(6, self.d_model)# tgt feature
        self.fuse = Fusion_module(opt)

    def forward(self, fv, fi, src_mask, tgt_mask, tgt, hc=None, is_first=False):
        fused = self.fuse(fv, fi)
        fused = torch.transpose(fused, 1, 0)
        fused = self.positional_encoding(fused)
        if tgt.dim() == 2:
            tgt = tgt.unsqueeze(0)
        tgt = self.linear(tgt)
        tgt = self.positional_encoding(tgt)
        tgt = torch.transpose(tgt, 1, 0)    #[8,10,6]
        memory = self.transformer.encoder(fused, src_mask, None)#[10,16,768]
        if is_first:
            tgt
        else:
            tgt = torch.cat([hc, tgt], 0)

        out = self.transformer.decoder(tgt, memory, tgt_mask)#[10,16,768]
        pose = self.generator(out)
        pose = pose.transpose(1,0)
        return pose, out


def generate_square_subsequent_mask(sz, device):
    mask1 = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask1 = mask1.float().masked_fill(mask1 == 0, float('-inf')).masked_fill(mask1 == 1, float(0.0))
    mask1 = mask1.transpose(1, 0)
    mask2 = (torch.triu(torch.ones((sz, sz), device=device), 1) == 1).transpose(0, 1)
    mask2 = mask2.float().masked_fill(mask2 == 0, float('-inf')).masked_fill(mask2 == 1, float(0.0))
    mask = torch.cat([mask1, mask2], 1)
    return mask

def create_mask(tgt_seq_len, device):
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=device)
    tgt_mask.transpose(1,0)
    return tgt_mask
class DeepVIO2(nn.Module):
    def __init__(self, opt):
        super(DeepVIO2, self).__init__()

        self.Feature_net = Encoder(opt)
        self.Pose_net = PoseTransformer(opt)
        self.opt = opt
        initialization(self)

    def forward(self, tgt, img, imu, is_first=True, selection='gumbel-softmax', hc = None):
        fv, fi = self.Feature_net(img, imu)
        device = tgt.device
        tgt_len = fv.shape[1]
        batch_size = fv.shape[0]
        seq_len = fv.shape[1]#[16,10,512]
        # src_mask, tgt_mask = create_mask(seq_len, tgt_len, device)
        src_mask = torch.zeros((seq_len, seq_len),device=device).type(torch.bool)
        if hc is None:
            hc = torch.zeros(seq_len, batch_size, self.opt.v_f_len + self.opt.i_f_len).to(fv.device) 
        if is_first:
            tgt_mask = generate_square_subsequent_mask(seq_len, device=device)
            tgt_mask = (torch.triu(torch.ones((seq_len, seq_len), device=device)) == 1).transpose(0, 1)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        else:
            tgt_mask = generate_square_subsequent_mask(seq_len, device=device)
        pose, hc = self.Pose_net(fv, fi, src_mask, tgt_mask, tgt, hc, is_first)#hc.shape [16,768]
        # else:
            # pose, hc = self.Pose_net(fv, fi, src_mask, tgt_mask, tgt, hc)
        return pose, hc

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
