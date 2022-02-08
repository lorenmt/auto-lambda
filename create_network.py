import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet


# --------------------------------------------------------------------------------
# Define DeepLab Modules
# --------------------------------------------------------------------------------
class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )]

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))

        # take pre-defined ResNet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu

        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# --------------------------------------------------------------------------------
# Define MTAN DeepLab
# --------------------------------------------------------------------------------
class MTANDeepLabv3(nn.Module):
    def __init__(self, tasks):
        super(MTANDeepLabv3, self).__init__()
        backbone = ResnetDilated(resnet.resnet50())
        ch = [256, 512, 1024, 2048]

        self.tasks = tasks
        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)

        # We will apply the attention over the last bottleneck layer in the ResNet.
        self.shared_layer1_b = backbone.layer1[:-1]
        self.shared_layer1_t = backbone.layer1[-1]

        self.shared_layer2_b = backbone.layer2[:-1]
        self.shared_layer2_t = backbone.layer2[-1]

        self.shared_layer3_b = backbone.layer3[:-1]
        self.shared_layer3_t = backbone.layer3[-1]

        self.shared_layer4_b = backbone.layer4[:-1]
        self.shared_layer4_t = backbone.layer4[-1]

        # Define task specific attention modules using a similar bottleneck design in residual block
        self.encoder_att_1 = nn.ModuleList([self.att_layer(ch[0], ch[0] // 4, ch[0]) for _ in self.tasks])
        self.encoder_att_2 = nn.ModuleList([self.att_layer(2 * ch[1], ch[1] // 4, ch[1]) for _ in self.tasks])
        self.encoder_att_3 = nn.ModuleList([self.att_layer(2 * ch[2], ch[2] // 4, ch[2]) for _ in self.tasks])
        self.encoder_att_4 = nn.ModuleList([self.att_layer(2 * ch[3], ch[3] // 4, ch[3]) for _ in self.tasks])

        # Define task shared attention encoders using residual bottleneck layers
        self.encoder_block_att_1 = self.conv_layer(ch[0], ch[1] // 4)
        self.encoder_block_att_2 = self.conv_layer(ch[1], ch[2] // 4)
        self.encoder_block_att_3 = self.conv_layer(ch[2], ch[3] // 4)

        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList([DeepLabHead(ch[-1], self.tasks[t]) for t in self.tasks])

    def att_layer(self, in_channel, intermediate_channel, out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=intermediate_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(intermediate_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=intermediate_channel, out_channels=out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid()
        )

    def conv_layer(self, in_channel, out_channel):
        downsample = nn.Sequential(nn.Conv2d(in_channel, 4 * out_channel, kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(4 * out_channel))
        return resnet.Bottleneck(in_channel, out_channel, downsample=downsample)

    def forward(self, x):
        _, _, im_h, im_w = x.shape

        # Shared convolution
        x = self.shared_conv(x)

        # Shared ResNet block 1
        u_1_b = self.shared_layer1_b(x)
        u_1_t = self.shared_layer1_t(u_1_b)

        # Shared ResNet block 2
        u_2_b = self.shared_layer2_b(u_1_t)
        u_2_t = self.shared_layer2_t(u_2_b)

        # Shared ResNet block 3
        u_3_b = self.shared_layer3_b(u_2_t)
        u_3_t = self.shared_layer3_t(u_3_b)

        # Shared ResNet block 4
        u_4_b = self.shared_layer4_b(u_3_t)
        u_4_t = self.shared_layer4_t(u_4_b)

        # Attention block 1 -> Apply attention over last residual block
        a_1_mask = [att_i(u_1_b) for att_i in self.encoder_att_1]  # Generate task specific attention map
        a_1 = [a_1_mask_i * u_1_t for a_1_mask_i in a_1_mask]  # Apply task specific attention map to shared features
        a_1 = [self.down_sampling(self.encoder_block_att_1(a_1_i)) for a_1_i in a_1]

        # Attention block 2 -> Apply attention over last residual block
        a_2_mask = [att_i(torch.cat((u_2_b, a_1_i), dim=1)) for a_1_i, att_i in zip(a_1, self.encoder_att_2)]
        a_2 = [a_2_mask_i * u_2_t for a_2_mask_i in a_2_mask]
        a_2 = [self.encoder_block_att_2(a_2_i) for a_2_i in a_2]

        # Attention block 3 -> Apply attention over last residual block
        a_3_mask = [att_i(torch.cat((u_3_b, a_2_i), dim=1)) for a_2_i, att_i in zip(a_2, self.encoder_att_3)]
        a_3 = [a_3_mask_i * u_3_t for a_3_mask_i in a_3_mask]
        a_3 = [self.encoder_block_att_3(a_3_i) for a_3_i in a_3]

        # Attention block 4 -> Apply attention over last residual block (without final encoder)
        a_4_mask = [att_i(torch.cat((u_4_b, a_3_i), dim=1)) for a_3_i, att_i in zip(a_3, self.encoder_att_4)]
        a_4 = [a_4_mask_i * u_4_t for a_4_mask_i in a_4_mask]

        # Task specific decoders
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](a_4[i]), size=[im_h, im_w], mode='bilinear', align_corners=True)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out

    def shared_modules(self):
        return [self.shared_conv,
                self.shared_layer1_b,
                self.shared_layer1_t,
                self.shared_layer2_b,
                self.shared_layer2_t,
                self.shared_layer3_b,
                self.shared_layer3_t,
                self.shared_layer4_b,
                self.shared_layer4_t,
                self.encoder_block_att_1,
                self.encoder_block_att_2,
                self.encoder_block_att_3]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()


# --------------------------------------------------------------------------------
# Define Split DeepLab
# --------------------------------------------------------------------------------
class MTLDeepLabv3(nn.Module):
    def __init__(self, tasks):
        super(MTLDeepLabv3, self).__init__()
        backbone = ResnetDilated(resnet.resnet50())
        ch = [256, 512, 1024, 2048]

        self.tasks = tasks

        self.shared_conv = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu1, backbone.maxpool)
        self.shared_layer1 = backbone.layer1
        self.shared_layer2 = backbone.layer2
        self.shared_layer3 = backbone.layer3
        self.shared_layer4 = backbone.layer4

        # Define task-specific decoders using ASPP modules
        self.decoders = nn.ModuleList([DeepLabHead(ch[-1], self.tasks[t]) for t in self.tasks])

    def forward(self, x):
        _, _, im_h, im_w = x.shape

        # Shared convolution
        x = self.shared_conv(x)
        x = self.shared_layer1(x)
        x = self.shared_layer2(x)
        x = self.shared_layer3(x)
        x = self.shared_layer4(x)

        # Task specific decoders
        out = [0 for _ in self.tasks]
        for i, t in enumerate(self.tasks):
            out[i] = F.interpolate(self.decoders[i](x), size=[im_h, im_w], mode='bilinear', align_corners=True)
            if t == 'normal':
                out[i] = out[i] / torch.norm(out[i], p=2, dim=1, keepdim=True)
        return out

    def shared_modules(self):
        return [self.shared_conv,
                self.shared_layer1,
                self.shared_layer2,
                self.shared_layer3,
                self.shared_layer4]

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()


# --------------------------------------------------------------------------------
# Define VGG-16 (for CIFAR-100 experiments)
# --------------------------------------------------------------------------------
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        self.bn_list = nn.ModuleList()

        for i in range(num_classes):
            self.bn_list.append(nn.BatchNorm2d(num_features))

    def forward(self, x, y):
        out = self.bn_list[y](x)
        return out


class MTLVGG16(nn.Module):
    def __init__(self, num_tasks):
        super(MTLVGG16, self).__init__()
        filter = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        self.num_tasks = num_tasks

        # define VGG-16 block
        network_layers = []
        channel_in = 3
        for ch in filter:
            if ch == 'M':
                network_layers += [nn.MaxPool2d(2, 2)]
            else:
                network_layers += [nn.Conv2d(channel_in, ch, kernel_size=3, padding=1),
                                   ConditionalBatchNorm2d(ch, num_tasks),
                                   nn.ReLU(inplace=True)]
                channel_in = ch

        self.network_block = nn.Sequential(*network_layers)

        # define classifiers here
        self.classifier = nn.ModuleList()
        for i in range(num_tasks):
            self.classifier.append(nn.Sequential(nn.Linear(filter[-1], 5)))

    def forward(self, x, task_id):
        for layer in self.network_block:
            if isinstance(layer, ConditionalBatchNorm2d):
                x = layer(x, task_id)
            else:
                x = layer(x)

        x = F.adaptive_avg_pool2d(x, 1)
        pred = self.classifier[task_id](x.view(x.shape[0], -1))
        return pred


