"""
Implementation of YOLOv3 architecture
"""

def draw_hsv(flow):

    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)

    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
            #t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


import torch
import torch.nn as nn
import numpy as np
import cv2


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

config_flow = [
    (32, 3, 1),
    (32, 3, 2),
    ["B", 1],
    (32, 3, 2),
    ["B", 2],
    (32, 3, 2),
    ["B", 4],
    (32, 3, 2)
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

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

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
        #self.layers_second_frame  = self._create_conv_layers2(  )
        #self.layers_flow          = self._create_conv_layers_flow(  )
        self.layers_both_frames  = self._create_conv_layers_for_conc_frames(  )


    def optical_flow(self, FRAMES , save = False ):

        folder = "saved_images/"
        # FRAMES is a List of 2 Frames of shape (b, ch , h , w)
        img1_tensor = FRAMES[0].detach().clone()    ## (b, c , h , w)
        img2_tensor = FRAMES[1].detach().clone()    ## (b, c , h , w)

        frame1_tensors_second_copy = FRAMES[0].detach().clone()  ## (b, c , h , w) 
        frame2_tensors_second_copy = FRAMES[1].detach().clone()  ## (b, c , h , w) 

        ## Denormalize the Frames
        unorm = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #unorm = UnNormalize(mean=[0, 0, 0] , std=[1, 1, 1])
        #mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,
        img1_tensor = unorm(img1_tensor)   
        img2_tensor = unorm(img2_tensor)

        bs = img1_tensor.shape[0]
        flow_tensors = []
        flow_mask_tensors = []

        clustered_masked_frames = []
        contours_list = [] 

        for i,bs_index in enumerate( range(bs)):

            ## Tensors to Numpy array
            ## Multiply by 255 as tensor has 0 to 1 values
            before =  img1_tensor[bs_index].detach().clone().cpu().permute( 1, 2, 0).numpy() * 255
            after = img2_tensor[bs_index].detach().clone().cpu().permute( 1, 2, 0).numpy() * 255 
            #before =  img1_tensor[bs_index].detach().clone().cpu().permute( 1, 2, 0).numpy() 
            #after = img2_tensor[bs_index].detach().clone().cpu().permute( 1, 2, 0).numpy()  

            #if i <20:
            #    print("Image max:",np.max( before))
            #    print("Image min:",np.min(before))

            # Convert images to grayscale
            before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
            after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(before_gray, after_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            hsv_flow = draw_hsv(flow)

            ## hsv_flow min 0  and max 56
            #print("hsv_flow min {}  and max {}".format(np.min(hsv_flow) , np.max(hsv_flow)))
            #hsv_flow_copy = torch.clamp( (hsv_flow/np.max(hsv_flow.astype(int))).astype(float) , min=0.0, max=1.0).astype(float)

### Compute the Mask from Flow
            ## Apply otsu thresh to get values as 0 and 1 only for Flow Mask
            img = cv2.cvtColor(hsv_flow, cv2.COLOR_BGR2GRAY)
            ## Max value will be 255 and minimum will be 0 and only two values, 0 and 255
            ret4,flow_mask = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  ## Have 0 and 1 values only

            ## Convert values of Flow MASK to 0 and 1 values
            flow_mask = flow_mask/255.0

            ## flow_mask has (h, w) to (1, h , w)
            flow_mask_copy = flow_mask
            
            flow_mask_copy = [flow_mask_copy] * 1  

            flow_mask_copy = np.stack(flow_mask_copy, axis=0)     # (1, h , w)        ## List of Arrays

            flow_mask_tensors.append(torch.from_numpy(flow_mask_copy).float().to(DEVICE))


            ## Convert Flow Values to 0 and 1 range
            hsv_flow = hsv_flow/255.0   ##  min 0.0  and max 0.1843137254901961
            #print("After Dividing by 255, hsv_flow min {}  and max {}".format(np.min(hsv_flow) , np.max(hsv_flow)))

            # hsv_flow has (h, w, 3) to (3, h , w)
            # hsv_flow has (h, w, 3) to (3, h , w)
            if (hsv_flow.shape[2] ==1):
                hsv_flow = [hsv_flow] * 3         # (3, h , w)        ## List of Arrays
                hsv_flow = np.stack(hsv_flow, axis=0) 
                flow_tensors.append(torch.from_numpy(hsv_flow).float().to(DEVICE))
            
            else:
                hsv_flow = np.transpose( hsv_flow , (2, 0, 1))
                ## Array to tensor of shape (channels, h , w)
                flow_tensors.append(torch.from_numpy(hsv_flow).float().to(DEVICE))
            

## Masked Flow Tensor (b, 1, h , w)  
        flow_mask_tensors_stacked = torch.stack(flow_mask_tensors, dim = 0).float().to(DEVICE)

## Inverse Masked Flow Tensor (b , 1 , h , w)
        inverse_mask_tensors_stacked = ( 1.0 - flow_mask_tensors_stacked ).float().to(DEVICE)

        ## Flow Tensors
        flow_tensors_stacked = torch.stack(flow_tensors, dim = 0).float().to(DEVICE)
        
        ## Normalize the Flow Tensor
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #normalize = Normalize(mean=[0, 0, 0] , std=[1, 1, 1])
        
        normalized_flow_tensors_stacked = normalize(flow_tensors_stacked)
        ## Normalize and divide by 255.0, hsv_flow min -2.1179039478302  and max -1.0553220510482788
        #print("Normalize and divide by 255.0, hsv_flow min {}  and max {}".format(torch.min(normalized_flow_tensors_stacked) , torch.max(normalized_flow_tensors_stacked)))


        ## Compute the Indexes from "Flow_Mask" where it is 0
        b , ch, h , w = flow_mask_tensors_stacked.shape
        mask_indices = (flow_mask_tensors_stacked == 0).reshape(b , ch , h  , w)

        return flow_tensors_stacked , normalized_flow_tensors_stacked , flow_mask_tensors_stacked , inverse_mask_tensors_stacked  ,  mask_indices 



## Input is a List of frames
    def forward(self, frame1_frame2_list ):
        
        flow_tensors_stacked , normalized_flow_tensors_stacked , flow_mask_tensors_stacked , inverse_mask_tensors_stacked  ,  mask_indices = self.optical_flow( frame1_frame2_list , save = False )
        
        x = torch.concat(frame1_frame2_list, dim= 1)    ## Both Frames
        x= torch.concat([x, normalized_flow_tensors_stacked] , dim = 1)  ## Both Frames with Flow
        #print("X shape::", x.shape)     ## X shape:: torch.Size([2, 9, 416, 416])
        frame1 = frame1_frame2_list[0]          ## First Frame
        frame2 = frame1_frame2_list[1]          ## Second Frame

        all_frames = [ frame1 , frame2 , normalized_flow_tensors_stacked] 

        frames_features_dict = { "frame1": [] , "frame2" : [] , "normalized_flow_tensors_stacked" : [] }
        keys=list(frames_features_dict.keys())
        
        #frame1_extract_features , frame2_extract_features , flow_features_extractation =  [] ,  [] , []
        for i in range(len(all_frames)):
            #print(keys[i])
            for layer in self.layers_first_frame:
            
                all_frames[i] = layer(all_frames[i])
                if not isinstance(layer, ResidualBlock):
                    
                    frames_features_dict[keys[i]].append(all_frames[i]) 

        '''
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
        
        ## Block3 for Flow Features Extraction
        flow_features_extractation = []
        for layer in self.layers_flow:    
            normalized_flow_tensors_stacked = layer(normalized_flow_tensors_stacked)
            if  not isinstance(layer, ResidualBlock):
                flow_features_extractation.append(normalized_flow_tensors_stacked) 
        '''

        ## For the Block4 with Both Frames and Extracted Features from Block1 and Block2
        ## Passing the Frame1, Frame2, Both Frames and Flow to these Layers in "self.layers_both_frames"
        outputs = []  # for each scale
        route_connections = []
        for i, layer in enumerate(self.layers_both_frames):
            

            if isinstance(layer, ScalePrediction):
                outputs.append(layer(x))
                continue

            if isinstance(layer, ConcatenateModule):
                x = torch.concat([x , frames_features_dict["frame1"][0], frames_features_dict["frame2"][0] , frames_features_dict["normalized_flow_tensors_stacked"][0] ], dim =1  )
                frames_features_dict["frame1"].pop(0)
                frames_features_dict["frame2"].pop(0)
                frames_features_dict["normalized_flow_tensors_stacked"].pop(0)
            
            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop(-1)

        return outputs

    ## Forming Layers for the CONCATENATED Frames
    def _create_conv_layers_for_conc_frames(self ):
        layers = nn.ModuleList()
        in_channels = self.in_channels * self.n_conc
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
                    ## Where 32 is flow Channels and "previous_ch[0] * num_frames" are Frame1 and Frame2 Channels
                    #in_channels = in_channels + ( previous_ch[0] * num_frames) + 32 ## Where 32 is flow Channels
                    in_channels = in_channels + ( previous_ch[0] * n_conc)
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

    ## Forming Layers for the Frame1
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

    ## Forming Layers for the Frame2
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

    ## Forming Layers for the Fr 
    def _create_conv_layers_flow(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config_flow:
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
    #DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = "cpu"
    model = YOLOv3(num_classes=num_classes)
    ## (tensor(1.0000), tensor(3.5763e-07))
    x = [ torch.rand((2, 3, IMAGE_SIZE, IMAGE_SIZE))  ,  torch.rand((2, 3, IMAGE_SIZE, IMAGE_SIZE))   ] 
    #print("Torch max::",(torch.max(x[0]), torch.min(x[1]) ))
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # (tensor(2.4285), tensor(-2.1179))
    x = [ norm(i) for i in x ] 
    #print("Torch max::",(torch.max(x[0]), torch.min(x[1]) ))
    print(count_parameters(model))     ## 194012066  parameters requiring grad
    print(count_all_parameters(model))  ## 194012066 total parameters

    ### This model has 197155266 number of Parameters as Flow Features are also increased 32 , 64, 128 , 256 , 512
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")

