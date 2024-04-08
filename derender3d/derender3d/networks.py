import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..derender3d.utils import IdentityLayer

EPS = 1e-7

AUTOENCODER_FIX = True


class PadSameConv2d(torch.nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        if isinstance(kernel_size, (tuple, list)):
            self.kernel_size_y = kernel_size[0]
            self.kernel_size_x = kernel_size[1]
        else:
            self.kernel_size_y = kernel_size
            self.kernel_size_x = kernel_size
        if isinstance(stride, (tuple, list)):
            self.stride_y = stride[0]
            self.stride_x = stride[1]
        else:
            self.stride_y = stride
            self.stride_x = stride

    def forward(self, x: torch.Tensor):
        _, _, height, width = x.shape
        padding_y = (self.stride_y * (math.ceil(height / self.stride_y) - 1) + self.kernel_size_y - height) / 2
        padding_x = (self.stride_x * (math.ceil(width / self.stride_x) - 1) + self.kernel_size_x - width) / 2
        padding = [math.floor(padding_x), math.ceil(padding_x), math.floor(padding_y), math.ceil(padding_y)]
        return F.pad(input=x, pad=padding)


class ConvReLU(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, leaky_relu_neg_slope=0.1):
        super().__init__()
        self.pad = PadSameConv2d(kernel_size=kernel_size, stride=stride)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_neg_slope)

    def forward(self, x: torch.Tensor):
        t = self.pad(x)
        t = self.conv(t)
        return self.leaky_relu(t)


class PaddedConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.pad = PadSameConv2d(kernel_size=kernel_size, stride=stride)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor):
        t = self.pad(x)
        t = self.conv(t)
        return t


class Encoder(nn.Module):
    def __init__(self, cin, cout, in_size=64, nf=64, activation=nn.Tanh):
        super(Encoder, self).__init__()

        max_channels = 8 * nf
        num_layers = int(math.log2(in_size)) - 1
        channels = [cin] + [min(nf * (2 ** i), max_channels) for i in range(num_layers)]

        self.layers = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(channels[i], channels[i+1], kernel_size=4, stride=2, padding=1 if i != num_layers - 1 else 0, bias=False),
                nn.ReLU(inplace=True)
            ) for i in range(num_layers)]
        )
        if activation is not None:
            self.out_layer = nn.Sequential(nn.Conv2d(max_channels, cout, kernel_size=1, stride=1, padding=0, bias=False), activation())
        else:
            self.out_layer = nn.Conv2d(max_channels, cout, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out_layer(x).reshape(x.size(0),-1)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad_(True)


class AutoEncoder(nn.Module):
    def __init__(self, cin, cout, nf=64, in_size=64, activation=nn.Tanh, depth=5, last_layer_relu=True):
        super().__init__()
        self.max_channels = 16 * nf
        # self.num_layers = min(int(math.log2(in_size)), 5)
        self.num_layers = int(math.log2(in_size))
        if depth is not None:
            self.num_layers = min(depth, self.num_layers)
        self.enc_channels = [cin] + [min(nf * (2 ** i), self.max_channels) for i in range(self.num_layers)]
        self.dec_channels = [min(nf * (2 ** i), self.max_channels) for i in reversed(range(self.num_layers))] + [cout]

        self.enc = nn.ModuleList([
            nn.Sequential(
                IdentityLayer() if i == 0 else nn.MaxPool2d(2),
                ConvReLU(in_channels=self.enc_channels[i], out_channels=self.enc_channels[i+1], kernel_size=3),
                ConvReLU(in_channels=self.enc_channels[i+1], out_channels=self.enc_channels[i+1], kernel_size=3))
            for i in range(self.num_layers)
        ])
        self.dec = nn.ModuleList([
            nn.Sequential(
                (ConvReLU if not (i == self.num_layers-1 and not last_layer_relu) else PaddedConv)(in_channels=(self.dec_channels[i] if i != 0 else 0) + self.enc_channels[-(i+1)], out_channels=self.dec_channels[i+1], kernel_size=3),
                (ConvReLU if not (i == self.num_layers-1 and AUTOENCODER_FIX) else PaddedConv)(in_channels=self.dec_channels[i+1], out_channels=self.dec_channels[i+1], kernel_size=3),
                IdentityLayer() if i == self.num_layers-1 else nn.Upsample(scale_factor=2))
            for i in range(self.num_layers)
        ])

        self.predictor = nn.Sequential(
            activation()
        )

    def forward(self, x):
        enc_feats = []
        for layer in self.enc:
            x = layer(x)
            enc_feats.append(x)
        for i, layer in enumerate(self.dec):
            if i == 0: x = enc_feats[-1]
            else: x = torch.cat([x, enc_feats[-(i+1)]], dim=1)
            x = layer(x)
        return [self.predictor(x)]

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad_(True)


class ConfNet(nn.Module):
    def __init__(self, cin, cout, zdim=128, nf=64, in_size=64, out_size=64):
        super(ConfNet, self).__init__()

        max_channels = 8 * nf
        enc_num_layers = int(math.log2(in_size)) - 1
        enc_channels = [cin] + [min(nf * (2 ** i), max_channels) for i in range(enc_num_layers)]
        enc_channels[-1] = zdim
        dec_num_layers = int(math.log2(out_size)) - 1
        dec_channels = [zdim] + [min(nf * (2 ** i), max_channels) for i in reversed(range(dec_num_layers))]

        self.enc_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(enc_channels[i], enc_channels[i + 1], kernel_size=4, stride=2,
                          padding=1 if i != enc_num_layers - 1 else 0, bias=False),
                nn.GroupNorm(16 * min(2 ** i, 8), nf * min(2 ** i, 8)) if i < enc_num_layers - 2 else IdentityLayer(),
                nn.LeakyReLU(0.2, inplace=True) if i != enc_num_layers - 1 else nn.ReLU()
            ) for i in range(enc_num_layers)]
        )

        self.dec_layers = nn.ModuleList(
            [nn.Sequential(
                nn.ConvTranspose2d(dec_channels[i], dec_channels[i + 1], kernel_size=4, stride=1 if i == 0 else 2, padding=0 if i == 0 else 1, bias=False),
                nn.GroupNorm(16 * (dec_channels[i + 1] // nf), dec_channels[i+1]) if i != 0 else IdentityLayer(),
                nn.ReLU(inplace=True)
            ) for i in reversed(range(dec_num_layers))]
        )

        self.predictors_l1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dec_channels[dec_num_layers - i], cout, kernel_size=5, stride=1, padding=2, bias=False),
                nn.Softplus()
            ) for i in range(dec_num_layers - 1)])
        self.predictors_percl = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dec_channels[dec_num_layers - i], cout, kernel_size=5, stride=1, padding=2, bias=False),
                nn.Softplus()
            ) for i in range(2, dec_num_layers, 1)])

    def forward(self, x):
        for layer in self.enc_layers:
            x = layer(x)
        outputs_l1 = []
        outputs_percl = []
        for i, layer in reversed(list(enumerate(self.dec_layers))):
            x = layer(x)
            if i < len(self.dec_layers)-1:
                outputs_l1 += [self.predictors_l1[i](x)]
            if i > 1 :
                outputs_percl += [self.predictors_percl[i-2](x)]
        return list(reversed(outputs_l1)), list(reversed(outputs_percl))


class PerceptualLoss(nn.Module):
    def __init__(self, requires_grad=False):
        super(PerceptualLoss, self).__init__()
        mean_rgb = torch.FloatTensor([0.485, 0.456, 0.406])
        std_rgb = torch.FloatTensor([0.229, 0.224, 0.225])
        self.register_buffer('mean_rgb', mean_rgb)
        self.register_buffer('std_rgb', std_rgb)

        vgg_pretrained_features = torchvision.models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def normalize(self, x):
        out = x/2 + 0.5
        out = (out - self.mean_rgb.view(1,3,1,1)) / self.std_rgb.view(1,3,1,1)
        return out

    def __call__(self, im1, im2, mask=None, conf_sigma=None):
        im = torch.cat([im1,im2], 0)
        im = self.normalize(im)  # normalize input

        ## compute features
        feats = []
        f = self.slice1(im)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice2(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice3(f)
        feats += [torch.chunk(f, 2, dim=0)]
        f = self.slice4(f)
        feats += [torch.chunk(f, 2, dim=0)]

        losses = []
        for f1, f2 in feats[2:3]:  # use relu3_3 features only
            loss = (f1-f2)**2
            if conf_sigma is not None:
                loss = loss / (2*conf_sigma**2 +EPS) + (conf_sigma +EPS).log()
            if mask is not None:
                b, c, h, w = loss.shape
                _, _, hm, wm = mask.shape
                sh, sw = hm//h, wm//w
                mask0 = nn.functional.avg_pool2d(mask, kernel_size=(sh,sw), stride=(sh,sw)).expand_as(loss)
                loss = (loss * mask0).sum() / mask0.sum()
            else:
                loss = loss.mean()
            losses += [loss]
        return sum(losses)


class DiscNet(nn.Module):
    def __init__(self, cin, cout, nf=64, in_size=64, norm=nn.InstanceNorm2d, activation=None):
        super(DiscNet, self).__init__()
        self.num_layers = int(math.log2(in_size)) - 1
        self.max_channels = 8 * nf
        self.enc_channels = [cin] + [min(nf * (2 ** i), self.max_channels) for i in range(self.num_layers)]
        network = [
            nn.Sequential(
                IdentityLayer() if i == 0 else nn.MaxPool2d(2),
                ConvReLU(in_channels=self.enc_channels[i], out_channels=self.enc_channels[i + 1], kernel_size=3),
                ConvReLU(in_channels=self.enc_channels[i + 1], out_channels=self.enc_channels[i + 1], kernel_size=3))
            for i in range(self.num_layers)
        ]
        network += [nn.Conv2d(nf*8, cout, kernel_size=4, stride=1, padding=0, bias=True)]
        if activation is not None:
            network += [activation()]
        self.network = nn.Sequential(*network)

    def forward(self, input):
        return self.network(input).reshape(input.size(0),-1)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad_(True)


class DoubleDiscNet(nn.Module):
    def __init__(self, cin, cout, nf=64, im_size=256, patch_size=64, norm=nn.InstanceNorm2d, activation=None, weighting=.5):
        super(DoubleDiscNet, self).__init__()
        self.disc_full = DiscNet(cin, cout, nf, in_size=im_size, norm=norm, activation=None)
        self.disc_patch = DiscNet(cin, cout, nf, in_size=patch_size, norm=norm, activation=None)
        self.activation = activation
        self.weighting = weighting

    def forward(self, input):
        input_full, input_patch = input
        out_full = self.disc_full(input_full)
        out_patch = self.disc_patch(input_patch)
        out = self.weighting * out_full + (1 - self.weighting) * out_patch
        if self.activation is not None:
            out = self.activation(out)
        return out.reshape(input_full.size(0),-1)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad_(True)


## copy from: https://github.com/taesungp/contrastive-unpaired-translation/blob/master/models/networks.py
class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'nonsaturating']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        bs = prediction.size(0)
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'nonsaturating':
            if target_is_real:
                loss = F.softplus(-prediction).view(bs, -1).mean(dim=1)
            else:
                loss = F.softplus(prediction).view(bs, -1).mean(dim=1)
        return loss
