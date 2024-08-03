import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from interpretable_multihead_attention_pooling import  InterpretableMultiHeadAttentionPool

class IPS_Net(nn.Module):
    def __init__(self, ):
        super(IPS_Net, self).__init__()

        self.model = smp.UnetPlusPlus(encoder_name='efficientnet-b4',in_channels=3)
        self.r_encoder = smp.UnetPlusPlus(encoder_name='efficientnet-b4', in_channels=3).encoder
        self.i_encoder = smp.UnetPlusPlus(encoder_name='efficientnet-b4', in_channels=3).encoder
        self.l_encoder = smp.UnetPlusPlus(encoder_name='efficientnet-b4', in_channels=3).encoder
        self.cl_encoder = smp.UnetPlusPlus(encoder_name='efficientnet-b4', in_channels=3).encoder
        
        self.dec = self.model.decoder
        self.dec_ref = smp.UnetPlusPlus(encoder_name='efficientnet-b4', in_channels=3).decoder
        self.dec_illum = smp.UnetPlusPlus(encoder_name='efficientnet-b4', in_channels=3).decoder
        self.dec_clahe = smp.UnetPlusPlus(encoder_name='efficientnet-b4', in_channels=3).decoder 
        self.dec_inp = smp.UnetPlusPlus(encoder_name='efficientnet-b4', in_channels=3).decoder 
   
        self.segm = self.model.segmentation_head 
        self.segm_ref = smp.UnetPlusPlus(encoder_name='efficientnet-b4', in_channels=3).segmentation_head
        self.segm_illum = smp.UnetPlusPlus(encoder_name='efficientnet-b4', in_channels=3).segmentation_head
        self.segm_clahe = smp.UnetPlusPlus(encoder_name='efficientnet-b4', in_channels=3).segmentation_head
        self.segm_inp = smp.UnetPlusPlus(encoder_name='efficientnet-b4', in_channels=3).segmentation_head

        self.layer = InterpretableMultiHeadAttentionPool(dims="bfbbs",n_head=5, d_model = 3)
        self.layers = nn.ModuleList([])

        shapes = [3,48,32,56,160,448]
        for i in range(6):
            self.layers.append(InterpretableMultiHeadAttentionPool(dims="bfbbs",n_head=5, d_model = shapes[i]))

   
    def forward(self,I_inp, R_inp, L_inp, CL_inp,mode="Train"):
        r_o = self.r_encoder(R_inp)
        i_o = self.i_encoder(I_inp)
        l_o = self.l_encoder(L_inp)
        cl_o = self.cl_encoder(CL_inp)
            
        att_outputs = []
        att_weights = []
        for i in range(6):
            input_i_stage = torch.cat((r_o[i].unsqueeze(4), i_o[i].unsqueeze(4), l_o[i].unsqueeze(4), cl_o[i].unsqueeze(4)),dim=4)
            att_outputs.append(self.layers[i](input_i_stage)[0][:,:,:,:,0])
            att_weights.append(self.layers[i](input_i_stage)[1])

        decoder_out = self.dec(*att_outputs)
        decoder_out_ref = self.dec_ref(*r_o)
        decoder_out_illum = self.dec_illum(*l_o)
        decoder_out_inp = self.dec_inp(*i_o)
        decoder_out_clahe = self.dec_clahe(*cl_o)
        
        masks = self.segm(decoder_out)
        masks_ref = self.segm_ref(decoder_out_ref)
        masks_illum = self.segm_illum(decoder_out_illum)
        masks_clahe = self.segm_clahe(decoder_out_clahe)
        masks_inp = self.segm_inp(decoder_out_inp)
        
        if mode == "Train":
            return masks, masks_ref, masks_illum, masks_clahe, masks_inp, att_weights, att_outputs
        else:
            return masks, att_weights, att_outputs

        
    
