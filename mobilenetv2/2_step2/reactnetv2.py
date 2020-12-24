import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, groups=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.number_of_weights = in_chn//groups * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn//groups, kernel_size, kernel_size)
        self.weights = nn.Parameter(torch.rand((self.number_of_weights,1)) * 0.001, requires_grad=True)
        self.groups=groups

    def forward(self, x):
        real_weights = self.weights.view(self.shape)
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        #print(scaling_factor, flush=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding, groups=self.groups)

        return y

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes),
                LearnableBias(out_planes),
                nn.PReLU(),
                LearnableBias(out_planes)
                )


class BinaryConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(BinaryConvBNReLU, self).__init__(
                HardBinaryConv(in_planes, out_planes, kernel_size, stride, padding, groups=groups),
                nn.BatchNorm2d(out_planes),
                LearnableBias(out_planes),
                nn.PReLU(),
                LearnableBias(out_planes)
                )

class BasicBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(BasicBlock, self).__init__()
        self.stride = stride
        assert stride in [1,2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(BinaryConvBNReLU(inp,hidden_dim,kernel_size=1))
        layers.extend([
            BinaryConvBNReLU(hidden_dim,hidden_dim,stride=stride,groups=hidden_dim),
            HardBinaryConv(hidden_dim,oup,1,1,0),
            nn.BatchNorm2d(oup),
            ])

        self.conv = nn.Sequential(*layers)

    def forward(self,x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
        


class reactnetv2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super(reactnetv2, self).__init__()
        block = BasicBlock
        input_channel = 32
        last_channel = 1280

        self.cfgs = [
                # expansion factor, channel, repeat number, stride
                [1, 16, 1, 1],
                [6, 24, 2, 1],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        input_channel = int(input_channel*width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))

        features = [ConvBNReLU(3, input_channel, stride=1)]
        
        for t, c, n, s in self.cfgs:
            output_channel = int(c*width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        
        features.append(BinaryConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, num_classes),
                )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.mean([2,3])
        x = self.classifier(x)
        return x

