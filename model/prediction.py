import torch
import torch.nn as nn
import torch.nn.functional as F



class TextAudioClassifierForCrossModalAttn(nn.Module):
    def __init__(self, audio_model = None, speaker_model = None, text_model = None, 
                 audio_dim=512, text_dim=None, hidden_dim=256, num_classes=4, dropout_prob=0.5, 
                speaker_dim=10, cross_modal_atten = False, modal = 'audio', multimodal_pooling=None):

        super(TextAudioClassifierForCrossModalAttn, self).__init__()
        
        self.audio_model = audio_model
        self.speaker_model = speaker_model
        self.text_model = text_model
        
        self.audio_dim, self.text_dim, self.hidden_dim, self.speaker_dim = audio_dim, text_dim, hidden_dim, speaker_dim
        self.num_classes = num_classes
        self.modal = modal 
        self.cross_modal_atten = cross_modal_atten

        self.dropout_prob = dropout_prob
        self.multimodal_pooling = multimodal_pooling
        
        if self.audio_model is not None: 
            if self.text_model is not None:
                if cross_modal_atten: 
                    self.input_dim = text_dim 
                    if modal in ['multimodal_concat']:
                        self.input_dim = text_dim + audio_dim        
                else: self.input_dim = text_dim + audio_dim            
            # [Audio Unimodal] 
            else: self.input_dim = audio_dim 

        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)    
        self.fuse_proj = nn.Linear(text_dim, hidden_dim)

        # final classifier 
        self.pred_linear = nn.Sequential(
            nn.Linear(self.text_dim, self.num_classes)
        )
        self.initialize_weights()
        
    def forward(self, audio_input, text_input = None, length=None):

        ## Text encoder encoder 
        audio_embeds, a_mask = self.audio_model(audio_input, length, True) 
        text_embeds, t_mask, text_seq, input_ids = self.text_model(text_input)            

        ## multimodal encoder 
        fuse_cls     = self.text_model(embeddings=text_seq, 
                                    s_attention_mask = t_mask,
                                    acoustic_encode = audio_embeds,
                                    a_attention_mask = a_mask,      
                                    return_dict = True,
                                    mode = 'fusion', 
                                    input_ids = input_ids,
                                    multimodal_pooling = self.multimodal_pooling
                                    )                     
        output = self.pred_linear(fuse_cls)
        
        ## text_feat & audio_feature for SSL loss  
        audio_feature  = F.normalize(self.audio_proj(audio_embeds[:,0,:]),dim=-1)  
        text_feat      = F.normalize(self.text_proj(text_embeds),dim=-1)   
        
        fuse_feat = F.normalize(self.fuse_proj(fuse_cls))
        return output, audio_feature, text_feat, fuse_feat
    
    def initialize_weights(self):
        for m in self.pred_linear:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



class TextAudioClassifier(nn.Module):
    def __init__(self, audio_model = None, speaker_model = None, text_model = None, 
                 audio_dim=512, text_dim=None, hidden_dim=256, num_classes=4, dropout_prob=0.5, 
                 speaker_dim=10, cross_modal_atten = False, modal = 'audio'):

        super(TextAudioClassifier, self).__init__()
        
        self.audio_model = audio_model
        self.speaker_model = speaker_model
        self.text_model = text_model
        
        self.audio_dim, self.text_dim, self.hidden_dim, self.speaker_dim = audio_dim, text_dim, hidden_dim, speaker_dim
        self.num_classes = num_classes
        self.modal = modal 
        self.cross_modal_atten = cross_modal_atten

        self.dropout_prob = dropout_prob
        
        if self.audio_model is not None: 
            # [Audio + Text Multimodal]
            if self.text_model is not None:
                if cross_modal_atten: 
                    self.input_dim = text_dim 
                    if modal in ['multimodal_concat']:
                        self.input_dim = text_dim + audio_dim        
                else: 
                    self.input_dim = text_dim + audio_dim
            
            # [Audio Unimodal] 
            else: self.input_dim = audio_dim 
        
        # [Text Unimodal] 
        else: self.input_dim  = text_dim

        
        self.pred_linear = nn.Sequential(
            nn.Linear(self.input_dim, self.num_classes)
        )
        self.initialize_weights()
        
    def forward(self, audio_input, text_input = None, length=None):
        
        # Unimodal [AUDIO]
        if self.modal =='audio': 
            feature, a_mask = self.audio_model(audio_input, length, True) 
            feature = feature[:,0,:]
        # Unimodal [TEXT]
        elif self.modal=='text': 
            feature, _ , _ , _ = self.text_model(embeddings=text_input)

        output = self.pred_linear(feature)

        return output, None, None, None
    
    def initialize_weights(self):
        for m in self.pred_linear:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
