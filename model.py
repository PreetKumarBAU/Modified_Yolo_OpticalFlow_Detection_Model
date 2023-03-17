"""
Implementation of YOLOv3 architecture
"""

import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

num_frames = 2
n_conc = num_frames + 1   ## Where one represent, all_frames through this block

## Concatenations of CNN Features, at each layer, from Frame1 and Frame2 at each layer, to the Both Frames Extracted Featured 


config_both = [
(32  , 3, 1),
["Conc"],
(64 * n_conc, 3, 2),
["B", 1],
["Conc"],
((128 * n_conc )  , 3, 2 ),
["B", 2],
["Conc"],
((256 * n_conc )  , 3, 2 ),
["B", 8],
["Conc"],
(512 * n_conc, 3, 2 ),
["Conc"],
(512  * n_conc, 3, 1 ),
( 1024 , 1, 1),
["B", 8],
(1024 , 3, 2),
["B", 4],  # To this point is Darknet-53
(512 , 1, 1),
(1024, 3, 1),
"S",
(256, 1, 1),
"U",
(256, 1, 1),
(512, 3, 1),
"S",
(128, 1, 1),
"U",
(128, 1, 1),
(256, 3, 1),
"S",
]

config1 = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 4],
    (512, 3, 2)
]

config2 = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 4],
    (512, 3, 2)
]



class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

class ConcatenateModule(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        return x

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_frames = 2
        self.n_conc = self.num_frames + 1



        #self.layers = self._create_conv_layers()
        #
        self.layers_first_frame  = self._create_conv_layers1(   )
        self.layers_second_frame  = self._create_conv_layers2(  )
        self.layers_both_frames  = self._create_conv_layers_for_conc_frames(  )


## Input is a List of frames
    def forward(self, frame1_frame2_list ):
        
        x = torch.concat(frame1_frame2_list, dim= 1)    ## Both Frames
        frame1 = frame1_frame2_list[0]          ## First Frame
        frame2 = frame1_frame2_list[1]          ## Second Frame

        ## Block1 for Frame1 Features Extraction
        frame1_extract_features = []
        for layer in self.layers_first_frame:
            
            
            frame1 = layer(frame1)
            if not isinstance(layer, ResidualBlock):
                frame1_extract_features.append(frame1) 

        ## Block2 for Frame2 Features Extraction
        frame2_extract_features = []
        for layer in self.layers_second_frame:
            
            
            frame2 = layer(frame2)
            if  not isinstance(layer, ResidualBlock):
                frame2_extract_features.append(frame2) 
        

        ## For the Block3 with Both Frames and Extracted Features from Block1 and Block2
        outputs = []  # for each scale
        route_connections = []
        for i, layer in enumerate(self.layers_both_frames):
            

            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            if isinstance(layer, ConcatenateModule):
                x = torch.concat([x , frame1_extract_features[0], frame2_extract_features[0]], dim =1  )
                frame1_extract_features.pop(0)
                frame2_extract_features.pop(0)
            
            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        return outputs


    def _create_conv_layers_for_conc_frames(self ):
        layers = nn.ModuleList()
        in_channels = self.in_channels * 2
        previous_ch = [32 , 64 ,128 , 256 , 512 ]
        residual_channels_after_8_repeats = []
        for i,module in enumerate(config_both):
            
            if isinstance(module, tuple):
                
                if len(module) == 4:
                    if module[3] == "D":
                        out_channels, kernel_size, stride , _= module
                        layers.append(
                            CNNBlock(
                                in_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=1 if kernel_size == 3 else 0,
                            )
                        )
                        in_channels = out_channels + ( previous_ch[0] * num_frames)  
                        

                elif len(module) == 3:

                    out_channels, kernel_size, stride = module
                    layers.append(
                        CNNBlock(
                            in_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=1 if kernel_size == 3 else 0,
                        )
                    )
                    in_channels = out_channels 

                

            elif isinstance(module, list):
                if module[0] == 'B':
                    #print("ResidualBlockResidualBlockResidualBlockResidualBlockResidualBlock")
                    num_repeats = module[1]
                    if num_repeats == 8:
                        residual_channels_after_8_repeats.append(in_channels)
                    layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))
                elif module[0] == "Conc":
                    #print("ConcConcConcConcConcConcConc")
                    layers.append(ConcatenateModule())
                    in_channels = in_channels + ( previous_ch[0] * num_frames)

                    previous_ch.pop(0) 




            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    #in_channels = in_channels * 3
                    in_channels = in_channels + residual_channels_after_8_repeats[-1]
                    residual_channels_after_8_repeats.pop(-1)


        return layers

    def _create_conv_layers1(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config1:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

        return layers

    def _create_conv_layers2(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config2:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

        return layers
    '''
    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3

        return layers
    '''


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == "__main__":
    num_classes = 1
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = [ torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE)) ,  torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE)) ] 
    print(count_parameters(model))     ## 185301794  parameters requiring grad
    print(count_all_parameters(model))  ## 185301794 total parameters
    #out = model(x)
    #assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    #assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    #assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")

