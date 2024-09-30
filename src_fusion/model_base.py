"""Defined functions for general classification model building
Essentially the functions from: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
"""
import torch
import torch.nn as nn
import math
from torchvision import models
from torchvision.models import resnet
from pretrainedmodels import se_resnext101_32x4d, inceptionresnetv2
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
import numpy as np



def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting == True:  # made sure to change to '== True' since just 'if feat:' is true for 'partial'
        for param in model.parameters():
            param.requires_grad = False


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x1, x2):

        # convert (32,8) to (32,1,8)
        #x1 = x1.unsqueeze(1)
        #x2 = x2.unsqueeze(1)


        batch_size, seq_length1, embed_dim1 = x1.size()
        seq_length2, embed_dim2 = x2.size()[1], x2.size()[2]


        # x1 and x2 = 32x8
        # Stack x1 and x2 together
        Y = torch.cat((x1, x2), dim=1)


        # Y = 32x16

        # Linear transformations
        QY = self.query(Y) # 32x16
        KY = self.key(Y)
        VY = self.value(Y)
    

        # Split into heads
        QY = QY.view(batch_size, seq_length1 + seq_length2, self.num_heads, self.head_dim).transpose(1, 2)
        KY = KY.view(batch_size, seq_length1 + seq_length2, self.num_heads, self.head_dim).transpose(1, 2)
        VY = VY.view(batch_size, seq_length1 + seq_length2, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(QY, KY.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, VY)

        # Concatenate heads

        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length1 + seq_length2, embed_dim1)

        # Final linear projection
        output = self.out_proj(attention_output)

        return output



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, embed_dim = x.size()

        # Linear transformations
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split into heads
        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)

        # Final linear projection
        output = self.out_proj(attention_output)
        return output

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerLayer, self).__init__()
        self.self_attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention sub-layer
        attn_output = self.self_attention(x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward sub-layer
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x



# class ECG_CNN(nn.Module):
#         def __init__(self, num_conv_layers=6, in_channels=2, first_layer_filters=128, num_classes=3, kernel_size=7):
#             super(ECG_CNN, self).__init__()

#             self.num_conv_layers = num_conv_layers
#             self.in_channels = in_channels
#             self.first_layer_filters = first_layer_filters

#             # Initialize the convolutional layers dynamically
#             conv_layers = []
#             current_filters = first_layer_filters # 128
#             for i in range(num_conv_layers): # 10

#                 if i % 2 != 0:
#                     current_filters = int(current_filters)*2 # reset back to last number of filters for odd numbered layers (repeated num filters)
#                     out_channels = int(current_filters)

#                     # add max pooling layer at every other conv block
#                     conv_layers.extend([
#                         nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
#                         nn.MaxPool1d(kernel_size=2,stride=2),
#                         nn.BatchNorm1d(out_channels),
#                         nn.ReLU()
#                     ])
#                 else: # add conv block without max pooling
#                     out_channels = int(current_filters)
#                     conv_layers.extend([
#                         nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
#                         nn.BatchNorm1d(out_channels),
#                         nn.ReLU()
#                     ])

#                 in_channels = out_channels  # Update in_channels for the next layer
#                 current_filters /= 2  # Halve the number of filters for the next layer

#             self.conv_blocks = nn.Sequential(*conv_layers)
#             self.global_max_pool = nn.AdaptiveMaxPool1d(1)

#             self._initialize_fc_layers()

#         def _initialize_fc_layers(self):
#             # Compute output size dynamically
#             example_input = torch.randn(16, 2, 2500)  # Example input tensor
#             with torch.set_grad_enabled(True):
#                 conv_output = self.conv_blocks(example_input)
#                 max_pool_output = self.global_max_pool(conv_output)
#                 conv_output_shape = max_pool_output.shape[1] # Number of channels = 8



#             self.fc_layers = nn.Sequential(
#                 nn.Linear(conv_output_shape, 16),
#                 nn.ReLU(),
#                 nn.Dropout(0.10),
#                 nn.Linear(16, 8),
#                 nn.ReLU(),
#                 nn.Linear(8, 2)
#             )
        
#         def forward(self, x):
#             # Convolutional layers
#             x = self.conv_blocks(x)

#             x = self.global_max_pool(x)

#             #x, _ = torch.max(x, dim=1, keepdim=True)
            
#             x = x.squeeze(-1) 
#             #x = x.squeeze(1) 
            

#             # Fully connected layers
#             #x = self.fc_layers(x)

#             #x = F.softmax(x, dim=-1)

#             return x

class AttentionFusion(nn.Module):
    def __init__(self, weight_type):
        super(AttentionFusion, self).__init__()

        # Echo CNN
        if weight_type == 'AP4':

            ap4_model = torch.load('./models/AP4_resnext101_best.pth')

            self.model_ft = nn.Sequential(*list(ap4_model.children())[:-1])

        else:
            self.model_ft = models.resnext101_32x8d(pretrained=True) 

        self.ecg_cnn = ECG_CNN(num_conv_layers=6, first_layer_filters=128, kernel_size=7)

        self.cross_attention = CrossAttention(embed_dim=32, num_heads=4)
    
        # downsample output features from Echo CNN
        self.fc_layers = nn.Sequential(
                # nn.Linear(2048, 512),
                # nn.ReLU(),
                # nn.Dropout(0.10),
                # nn.Linear(512, 128),
                # nn.ReLU(),
                # nn.Linear(128, 64),
                # nn.ReLU(),
                # nn.Linear(64, 32)

                nn.Linear(2048, 1000),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(1000, 32)
                #nn.Linear(1000, 8)
            )

        self.ecg_mlp = nn.Sequential(
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8)
                    # output of 8
            )

        # get 3 value output using fusion input
        self.final_mlp = nn.Sequential(
                    #nn.Linear(52, 16),
                    #nn.ReLU(),
                    #nn.Dropout(0.10),
                    nn.Linear(64, 32), # cross + self
                    nn.Linear(32,8),
                    #nn.Linear(16, 8), use for regular model
                    nn.ReLU(),
                    nn.Linear(8, 3)
                )
        self.self_attention_cxr = TransformerLayer(embed_dim = 32, num_heads = 4, ff_dim = 32)
        self.self_attention_ecg = TransformerLayer(embed_dim = 32, num_heads = 4, ff_dim = 32)

    def forward(self, echo_input, ecg_input):

        # extract Echo features from ResNext101
        #features = self.embeddings(echo_input)

        echo_features = self.model_ft(echo_input)
        echo_features = echo_features.squeeze() 

        # downsample Echo features to match ECG output
        echo_down_feats = self.fc_layers(echo_features)

        ecg_features = self.ecg_cnn(ecg_input) 
        #ecg_features = self.ecg_mlp(ecg_features) dont need for self attention

        echo_down_feats = echo_down_feats.view(echo_down_feats.size(0), -1, echo_down_feats.size(-1))
        ecg_features = ecg_features.view(ecg_features.size(0), -1, ecg_features.size(-1))

        feat_map1_attention = self.self_attention_cxr(echo_down_feats)
        feat_map2_attention = self.self_attention_ecg(ecg_features)

        feat_map1_ = feat_map1_attention.view(feat_map1_attention.size(0), feat_map1_attention.size(2))
        feat_map2_ = feat_map2_attention.view(feat_map2_attention.size(0), feat_map2_attention.size(2))


        feat_map1_ = feat_map1_ / feat_map1_.norm(dim=1, keepdim=True)
        feat_map2_ = feat_map2_ / feat_map2_.norm(dim=1, keepdim=True)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * feat_map1_ @ feat_map2_.t()
        logits_per_text = logits_per_image.t()

      
        n = logits_per_image.size(0)
        labels = torch.arange(n).to(logits_per_image.device)  # Ensure labels are on the same device as logits

        # Compute cross-entropy loss for image-to-text (axis=0 in the pseudocode)
        loss_i = F.cross_entropy(logits_per_image, labels)

        # Compute cross-entropy loss for text-to-image (axis=1 in the pseudocode)
        loss_t = F.cross_entropy(logits_per_text, labels)

        # Compute the final loss as the average of loss_i and loss_t
        loss = (loss_i + loss_t) / 2

        # get size of batch through feature vector
        batch_size = echo_down_feats.size()[0]
        feat_map = self.cross_attention(feat_map1_attention, feat_map2_attention)
        feat_map = feat_map.permute(0, 2, 1).reshape(batch_size, -1, 1, 1) 
        feat_map = feat_map.squeeze(-1).squeeze(-1)

        # concatenate Echo features with 26 time domain ecg features
        #concat_feats = torch.concat([ecg_features.float(), echo_down_feats.float()], dim=1)


        # feed concatenated features into final MLP thats output has 3 values (softmax in training base code)
        output = self.final_mlp(feat_map)


        return output, loss


class ECG_CNN(nn.Module):
        def __init__(self, num_conv_layers=6, in_channels=2, first_layer_filters=128, num_classes=3, kernel_size=7):
            super(ECG_CNN, self).__init__()

            self.num_conv_layers = num_conv_layers
            self.in_channels = in_channels
            self.first_layer_filters = first_layer_filters

            # Initialize the convolutional layers dynamically
            conv_layers = []
            current_filters = first_layer_filters # 128
            for i in range(num_conv_layers): # 10

                if i % 2 != 0:
                    current_filters = int(current_filters)*2 # reset back to last number of filters for odd numbered layers (repeated num filters)
                    out_channels = int(current_filters)

                    # add max pooling layer at every other conv block
                    conv_layers.extend([
                        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                        nn.MaxPool1d(kernel_size=2,stride=2),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU()
                    ])
                else: # add conv block without max pooling
                    out_channels = int(current_filters)
                    conv_layers.extend([
                        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
                        nn.BatchNorm1d(out_channels),
                        nn.ReLU()
                    ])

                in_channels = out_channels  # Update in_channels for the next layer
                current_filters /= 2  # Halve the number of filters for the next layer

            self.conv_blocks = nn.Sequential(*conv_layers)
            self.global_max_pool = nn.AdaptiveMaxPool1d(1)

            self._initialize_fc_layers()

        def _initialize_fc_layers(self):
            # Compute output size dynamically
            example_input = torch.randn(16, 2, 2500)  # Example input tensor
            with torch.set_grad_enabled(True):
                conv_output = self.conv_blocks(example_input)
                max_pool_output = self.global_max_pool(conv_output)
                conv_output_shape = max_pool_output.shape[1] # Number of channels = 8



            self.fc_layers = nn.Sequential(
                nn.Linear(conv_output_shape, 16),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 3)
            )
        
        def forward(self, x):
            # Convolutional layers
            x = self.conv_blocks(x)

            x = self.global_max_pool(x)

            #x, _ = torch.max(x, dim=1, keepdim=True)
            
            x = x.squeeze(-1) 
            #x = x.squeeze(1) 
            

            # Fully connected layers
            #x = self.fc_layers(x)

            #x = F.softmax(x, dim=-1)

            return x

class ECG_CNN_Echo_CNN_Fusion(nn.Module):
    def __init__(self, weight_type):
        super(ECG_CNN_Echo_CNN_Fusion, self).__init__()

        # Echo CNN
        if weight_type == 'AP4':

            #ap4_model = torch.load('./models/AP4_resnext101_best.pth')
            ap4_model = torch.load('./models/4class_resnext101_best.pth')

            self.model_ft = nn.Sequential(*list(ap4_model.children())[:-1])

        else:
            self.model_ft = models.resnext101_32x8d(pretrained=True) 

        self.ecg_cnn = ECG_CNN(num_conv_layers=6, first_layer_filters=128, kernel_size=7)

    
        # downsample output features from Echo CNN
        self.fc_layers = nn.Sequential(
                nn.Linear(2048, 1000),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(1000, 8)
            )

        self.ecg_mlp = nn.Sequential(
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8)
                    # output of 8
            )

        # get 3 value output using fusion input
        self.final_mlp = nn.Sequential(
                    nn.Linear(16, 8), #use for regular model
                    nn.ReLU(),
                    nn.Linear(8, 4)
                )


    def forward(self, echo_input, ecg_input):

        echo_features = self.model_ft(echo_input)
        echo_features = echo_features.squeeze() 

        # downsample Echo features to match ECG output
        echo_down_feats = self.fc_layers(echo_features)

        ecg_features = self.ecg_cnn(ecg_input) 
        ecg_features = self.ecg_mlp(ecg_features) 

        concat_feats = torch.concat([ecg_features.float(), echo_down_feats.float()], dim=1)


        # feed concatenated features into final MLP thats output has 3 values (softmax in training base code)
        output = self.final_mlp(concat_feats)


        return output



class ECG_ML_Echo_CNN_Fusion(nn.Module):
    def __init__(self, weight_type):
        super(ECG_ML_Echo_CNN_Fusion, self).__init__()

        if weight_type == 'AP4':

            ap4_model = torch.load('./models/AP4_resnext101_best.pth')

            self.model_ft = nn.Sequential(*list(ap4_model.children())[:-1])

        else:
            self.model_ft = models.resnext101_32x8d(pretrained=True) 

        # downsample output features from Echo CNN
        self.fc_layers = nn.Sequential(
                # nn.Linear(2048, 512), # set to 1000 if using imagenet, 2048 for AP4
                # nn.ReLU(),
                # nn.Dropout(0.10),
                # nn.Linear(512, 128),
                # nn.ReLU(),
                # nn.Linear(128, 32),
                # nn.ReLU(),
                # nn.Linear(32, 16),
                # nn.ReLU(),
                # nn.Linear(16, 8) # output of 8

                nn.Linear(2048, 1000), # set to 1000 if using imagenet, 2048 for AP4
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(1000, 8)
            )
        self.ecg_mlp = nn.Sequential(
                    nn.Linear(26, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8)
                    # output of 8
            )
        
        # get 3 value output using fusion input
        self.final_mlp = nn.Sequential(
                    #nn.Linear(52, 16),
                    #nn.ReLU(),
                    #nn.Dropout(0.10),
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 3)
                )

    

    def forward(self, echo_input, ecg_input):

        # extract Echo features from ResNext101
        # features = self.embeddings(echo_input)
        

        features = self.model_ft(echo_input) # 1000
        
        
        features = features.squeeze()

        # downsample Echo features to 8 to match 
        down_feats = self.fc_layers(features)

        ecg_input = ecg_input.float()

        ecg_features = self.ecg_mlp(ecg_input)

        concat_feats = torch.concat([ecg_features.float(), down_feats.float()], dim=1)

        # feed concatenated features into final MLP thats output has 3 values (softmax in training base code)
        output = self.final_mlp(concat_feats)


        return output

        




class Fusion_Modelv3(nn.Module):
    def __init__(self, model, model_ftrs, num_classes, demo_ftrs=None, icd_ftrs=None, vital_ftrs=None):
        """v3 where the demographic and vital information features are just concatinated to the others to reduce
        information reduction as well as a dense layer (torch.linear) to the maximum feature vector - although there
        will be the problem of information dilution, it's better than the previous information loss from going to a very
        small layer (I think)

        model is the actual model with the linear output being the same size as linear input (image features)
        model_ftrs is the length of the image feature vector
        demo_ftrs is the length of the demographic feature vector
        icd_ftrs is the length of the icd code feature vector
        vital_ftrs is the length of the vital information feature vector
        """
        super(Fusion_Modelv3, self).__init__()
        self.image_branch = model

        # make all that is none is zero so the vectors are consistent
        if demo_ftrs == None:
            demo_ftrs = 0
        if icd_ftrs == None:
            icd_ftrs = 0
        if vital_ftrs == None:
            vital_ftrs = 0
        # summing up the feature length for non-image features since they are going to be concatinated
        non_img_ftrs = sum((demo_ftrs, icd_ftrs, vital_ftrs))
        # find the maximum feature vector
        max_feat = max(model_ftrs, demo_ftrs, icd_ftrs, vital_ftrs)
        # non-image branch is just the concat features densely connected to the larger feature vector
        self.non_img_branch = nn.Linear(non_img_ftrs, max_feat)
        self.fused_output = nn.Linear(max_feat*2, num_classes)

    def forward(self, image, demo_feat=None, icd_feat=None, vital_feat=None):
        img_ft = self.image_branch(image)  # shape: 32, 1024, 4, 4
        # make all that is none as an empty tensor so that torch cat ignores empty ones
        if demo_feat != None and icd_feat != None and vital_feat != None:
            non_img_ft = self.non_img_branch(
                torch.cat([torch.squeeze(demo_feat), torch.squeeze(icd_feat), torch.squeeze(vital_feat)], 1))
        elif demo_feat != None:
            non_img_ft = self.non_img_branch(torch.squeeze(demo_feat))
        elif icd_feat != None:
            non_img_ft = self.non_img_branch(torch.squeeze(icd_feat))
        elif vital_feat != None:
            non_img_ft = self.non_img_branch(torch.squeeze(vital_feat))

        # concat the image and non-image features together
        fused_ft = torch.cat([img_ft, non_img_ft], 1)

        return self.fused_output(fused_ft)


class Fusion_Modelv2(nn.Module):
    # v2.1 where the demographic features are just concatinated
    def __init__(self, model, model_ftrs, num_classes, demo_ftrs=None, icd_ftrs=None):
        """model is the actual model with the linear output being the same size as linear input
        model_ftrs is the length of the model feature vector
        text_ftrs is the length of the input text features"""
        super(Fusion_Modelv2, self).__init__()
        self.image_branch = model

        # find the smallest features (while checking if demo or icd or both are inputs)
        if demo_ftrs != None and icd_ftrs != None:
            min_feat = min(model_ftrs, demo_ftrs, icd_ftrs)
            self.demo_branch = nn.Linear(demo_ftrs, min_feat)
            self.icd_branch = nn.Linear(icd_ftrs, min_feat)
        elif demo_ftrs != None and icd_ftrs == None:
            min_feat = min(model_ftrs, demo_ftrs)
            self.demo_branch = nn.Linear(demo_ftrs, min_feat)
        elif demo_ftrs == None and icd_ftrs != None:
            min_feat = min(model_ftrs, icd_ftrs)
            self.icd_branch = nn.Linear(icd_ftrs, min_feat)

        if demo_ftrs != None and icd_ftrs != None:
            self.fused_output = nn.Linear(min_feat*3, num_classes)
        else:
            self.fused_output = nn.Linear(min_feat*2, num_classes)

    def forward(self, image, demo_feat=None, icd_feat=None):
        img_ft = self.image_branch(image)  # shape: 32, 1024, 4, 4
        if demo_feat != None and icd_feat != None:
            demo_ft = self.demo_branch(demo_feat)
            icd_ft = self.icd_branch(icd_feat)
            fused_ft = torch.cat([img_ft, torch.squeeze(demo_ft), torch.squeeze(icd_ft)], 1)
        elif demo_feat != None and icd_feat == None:
            demo_ft = self.demo_branch(demo_feat)
            fused_ft = torch.cat([img_ft, torch.squeeze(demo_ft)], 1)
        elif demo_feat == None and icd_feat != None:
            icd_ft = self.icd_branch(icd_feat)
            fused_ft = torch.cat([img_ft, torch.squeeze(icd_ft)], 1)

        return self.fused_output(fused_ft)


class Fusion_Modelv1(nn.Module):
    def __init__(self, model, model_ftrs, text_ftrs, num_classes):
        """model is the actual model with the linear output being the same size as linear input
        model_ftrs is the length of the model feature vector
        text_ftrs is the length of the input text features"""
        super(Fusion_Modelv1, self).__init__()
        # find the smaller features
        min_feat = min(model_ftrs, text_ftrs)
        self.image_branch = nn.Sequential(
            model, nn.Linear(model_ftrs, min_feat)
        )
        # creates a linear layer to make it the same size as the model output
        self.text_branch = nn.Linear(text_ftrs, min_feat)
        # average pool
        self.fused_output = nn.Linear(min_feat*2, num_classes)

    def forward(self, image, text_feat):
        img_ft = self.image_branch(image)  # shape: 32, 1024, 4, 4
        txt_ft = self.text_branch(text_feat)
        fused_ft = torch.cat([img_ft, torch.squeeze(txt_ft)], 1)
        return self.fused_output(fused_ft)


def initialize_model(params):
    """To add another model into this function:
    1) install the model; 2) create a model_name; 3) copy the format of the other models and create the
    model_ft, set_parameter, etc. 3.5) the num_ftrs might need to load different names (i.e. model_ft.fc or
    model_ft._fc - different models will have different final layers but will generally be 'fc' for pytorch pretrained
    models; '_fc' for efficientNet and 'last_layer' for pretrainedmodels)
    4) add the final layer as just a linear layer*
    4.5)* final layer needs to be a linear layer because pytorch loss functions include softmax and sigmoid activations
    within them
    5) make sure to read the originial architecture and set the input size

    Currently the models I have added are:
    EfficientNet-B7, Resnet18, Resnet50, Resnet101, Resnet152, Alexnet, VGG11_bn, VGG19_bn, Squeezenet1.1, Densenet121,
    Densenet169, Densenet161, MobileNetV2, and Inception v3

    see link for more models and details on each: https://pytorch.org/docs/stable/torchvision/models.html
    """

    print('initializing {} model'.format(params['model_name']))
    model_name = params['model_name']
    num_classes = params['num_classes']
    feature_extract = params['feature_extract']
    use_pretrained = params['use_pretrained']
    fusion = params['fusion_model']  # fusion1 or fusion2 for different fusion models
    if fusion != 'fusion1' and fusion != 'fusion2' and fusion != 'fusion3':
        demo_ftrs = 0
        icd_ftrs = 0
        vit_ftrs = 0
    else:
        demo_ftrs = params['demo_ft_len']
        icd_ftrs = params['icd_ft_len']
        vit_ftrs = params['vit_ft_len']
    if demo_ftrs != None and icd_ftrs != None and vit_ftrs != None:
        text_ftrs = demo_ftrs + icd_ftrs + vit_ftrs
    elif demo_ftrs == None and icd_ftrs != None and vit_ftrs == None:
        text_ftrs = icd_ftrs
    elif demo_ftrs != None and icd_ftrs == None and vit_ftrs == None:
        text_ftrs = demo_ftrs
    elif demo_ftrs == None and icd_ftrs == None and vit_ftrs != None:
        text_ftrs = vit_ftrs

    # Initialize these variables which will be set in this if statement. Each of these variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnext101":
        """resnext101_32x8d
        """
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features


        if fusion == 'fusion1':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "se-resnext101":
        """ SE-ResNeXt101
        """
        if use_pretrained:
            model_ft = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        else:
            model_ft = se_resnext101_32x4d(num_classes=1000, pretrained=None)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        if fusion == 'fusion1':
            model_ft.last_linear = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.last_linear = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.last_linear = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "inceptionresnetv2":
        """ Inception ResNet v2
        """
        if use_pretrained:
            model_ft = inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        else:
            model_ft = inceptionresnetv2(num_classes=1000, pretrained=None)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        if fusion == 'fusion1':
            model_ft.last_linear = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.last_linear = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.last_linear = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    elif model_name == "effnet-b7":
        """ EfficientNet-B7
        """
        if use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b7')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b7')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft._fc.in_features
        if fusion == 'fusion1':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft._fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft._fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "effnet-b5":
        """ EfficientNet-B5
        """
        if use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b5')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b5')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft._fc.in_features
        if fusion == 'fusion1':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft._fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft._fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "effnet-b3":
        """ EfficientNet-B3
        """
        if use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b3')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b3')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft._fc.in_features
        if fusion == 'fusion1':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft._fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft._fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "effnet-b2":
        """ EfficientNet-B2
        """
        if use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b2')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b2')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft._fc.in_features
        if fusion == 'fusion1':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft._fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft._fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "effnet-b1":
        """ EfficientNet-B1
        """
        if use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b1')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b1')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft._fc.in_features
        if fusion == 'fusion1':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft._fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft._fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "effnet-b0":
        """ EfficientNet-B0
        """
        if use_pretrained:
            model_ft = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            model_ft = EfficientNet.from_name('efficientnet-b0')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft._fc.in_features
        if fusion == 'fusion1':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft._fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft._fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft._fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == 'resnet9':
        """ Modified Resnet18 with only one block per layer so that it's 'half'
        - this is a good way to make custom models from pytorch classes
        """
        model_ft = resnet._resnet('resnet18_half', resnet.BasicBlock, [1, 1, 1, 1], False, False)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        if fusion == 'fusion1':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        if fusion == 'fusion1':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        if fusion == 'fusion1':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        if fusion == 'fusion1':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        if fusion == 'fusion1':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.fc = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.fc = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        if fusion == 'fusion1':
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.classifier[6] = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg11bn":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        if fusion == 'fusion1':
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.classifier[6] = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg19bn":
        """ VGG19_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        if fusion == 'fusion1':
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.classifier[6] = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet1.1":
        """ Squeezenet1.1
        """
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet121":
        """ Densenet121
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        if fusion == 'fusion1':
            model_ft.classifier = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.classifier = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.classifier = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "densenet169":
        """ Densenet169
        """
        model_ft = models.densenet169(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        if fusion == 'fusion1':
            model_ft.classifier = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.classifier = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.classifier = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "densenet161":
        """ Densenet161
        """
        model_ft = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        if fusion == 'fusion1':
            model_ft.classifier = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.classifier = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.classifier = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "mobilenet":
        """ MobileNetV2
        """
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        if fusion == 'fusion1':
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv1(model_ft, num_ftrs, text_ftrs, num_classes)
        elif fusion == 'fusion2':
            if demo_ftrs != None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, demo_ftrs, icd_ftrs)
            elif demo_ftrs == None and icd_ftrs != None:
                min_ftrs = min(num_ftrs, icd_ftrs)
            elif demo_ftrs != None and icd_ftrs == None:
                min_ftrs = min(num_ftrs, demo_ftrs)
            model_ft.classifier[1] = nn.Linear(num_ftrs, min_ftrs)
            model_ft = Fusion_Modelv2(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs)
        elif fusion == 'fusion3':
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_ftrs)
            model_ft = Fusion_Modelv3(model_ft, num_ftrs, num_classes, demo_ftrs, icd_ftrs, vit_ftrs)
        else:
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size
