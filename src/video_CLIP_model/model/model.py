import torch
import torch.nn as nn
import numpy as np
from transformers import (
    BertModel,
    BertConfig,
    BertTokenizer,
    TimesformerConfig,
    TimesformerModel
)
from torch.nn import CrossEntropyLoss

class CLIPOperator(torch.nn):
    def __init__(self, device = "auto"):
        super(CLIPOperator, self).__init__()
    
    def forward(self, text_features: torch.Tensor,video_features:torch.Tensor):
        video_features.transpose(1,0)
        similarities_score = torch.matmul(text_features,video_features)
        label = torch.from_numpy(np.arange(0,len(text_features),1))
        loss_axis_1 = CrossEntropyLoss(similarities_score,label)
        loss_axis_2 = CrossEntropyLoss(similarities_score.transpose(1,0),label)
        loss = (loss_axis_1 + loss_axis_2)/2
        return {'loss': loss,
                'output': similarities_score}

class VideoCLIPModel(torch.nn):
    def __init__(self, 
                 bert_cfg:BertConfig, 
                 timesformer_cfg:TimesformerConfig, 
                 device = "auto"):
        super(VideoCLIPModel, self).__init__()
        self.device = device
        self.text_model = BertModel(bert_cfg)
        self.video_model = TimesformerModel(timesformer_cfg)
        self.operator = CLIPOperator(device = self.device)
    def forward(self, pixels_values, text_index):
        text_features = self.text_model(text_index)[0][:,0]
        video_features = self.video_model(pixels_values)[0][:,0]
        return self.operator(text_features, video_features)

