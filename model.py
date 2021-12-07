
import torch
from metric_learning import *
from torch.utils.model_zoo import load_url as load_state_dict_from_url

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class AttentionPool(nn.Module):
    def __init__(self, in_units, hidden = 2048, reduction = 'mean'):
        super(AttentionPool, self).__init__()

        self.reduction = reduction
        self.att = nn.Sequential(nn.Linear(in_units, hidden),
                                  nn.ReLU(),
                                  nn.Linear(hidden, 1))


    def forward(self, x):

        # x has shape (bs, n_patches, feature_dim)
        att_weights = torch.softmax(self.att(x),dim=1)
        
        if self.reduction == 'sum':
            x_pooled = (x * att_weights).sum(1)
        elif self.reduction == 'mean':
            x_pooled = (x * att_weights).mean(1)
        elif self.reduction == 'max':
            x_pooled = (x * att_weights).max(1)[0]

        return x_pooled
    
class PatchAttentionModel(nn.Module):
    def __init__(self, n_classes, n_patches, pretrained, metric_head = 'arc_margin', kernel = 2):
        
        super(PatchAttentionModel, self).__init__()
        
        self.n_classes = n_classes
        self.n_patches = n_patches
        self.pretrained = pretrained
        self.metric_head = metric_head      
        self.kernel = kernel
        
        self.att_pool = AttentionPool(2048)
        self.backbone_holistic = models.vgg16(pretrained=self.pretrained).features
        self.backbone_holistic.avgpool = nn.AdaptiveAvgPool2d(self.kernel)
        
        self.backbone_patch = models.vgg16(pretrained=self.pretrained).features
        self.backbone_patch.avgpool = nn.AdaptiveAvgPool2d(self.kernel)
        
        self.fc1 = nn.Linear(2*512*self.kernel*self.kernel, 2*1024)
        self.bn1 = nn.BatchNorm1d(num_features=2*1024)
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(0.3) # 0.3
        
        self.fc2 = nn.Linear(2*1024, 2*512)
        self.bn2 = nn.BatchNorm1d(num_features=2*512) 
        
        if self.metric_head == 'add_margin':
           self.metric_fc = AddMarginProduct(1024, self.n_classes, s=30, m=0.35)
        elif self.metric_head == 'arc_margin':
           self.metric_fc = ArcMarginProduct(1024, self.n_classes, s=64, m=0.5, easy_margin=False) # s = 30 # 64
        elif self.metric_head == 'sphere':
           self.metric_fc = SphereProduct(1024, self.n_classes, m=4)
        elif self.metric_head == 'ada_cos':
           self.metric_fc = AdaCos(1024, self.n_classes, m=0.5)

    def forward_features_holistic(self, x):
        features = self.backbone_holistic(x)
        features = features.view(features.size(0), 512*self.kernel*self.kernel)
        return features
    
    def forward_features_patches(self, patches):

        patches_features = ()
        feature_matrix = ()
        
        for patch in patches: 

            patch_features = self.backbone_patch(patch)
            patch_features = patch_features.view(patch_features.size(0), 512*self.kernel*self.kernel)
             
            feature_matrix += (patch_features, )
            patches_features += (torch.max(patch_features, 0)[0].unsqueeze(0),)            
    
        feature_matrix = torch.stack(feature_matrix, 0)
        local_features = self.att_pool(feature_matrix)
                   
        return local_features
    
    def forward_combined_features(self, holistic_features, patches_features):
        combined_features = torch.cat((holistic_features, patches_features), 1)
        
        combined_features = self.fc1(combined_features)
        combined_features = self.bn1(combined_features)
        combined_features = self.relu(combined_features)
        combined_features = self.dropout(combined_features)
        
        combined_features = self.fc2(combined_features)
        combined_features = self.bn2(combined_features)
        combined_features = F.normalize(combined_features)
        
        return combined_features
    
    def forward_final_features(self, imgs, patches):
        holistic_features = self.forward_features_holistic(imgs)
        local_features = self.forward_features_patches(patches)
        combined_features = self.forward_combined_features(holistic_features, local_features)
        return combined_features
    
    def forward(self, imgs, patches, labels):        
        
        combined_features = self.forward_final_features(imgs, patches)
        logits = self.metric_fc(combined_features, labels)
        
        return logits, combined_features

