import torch
# from csdn





def get_parameter_number(model: torch.nn.Module) -> dict:
    """
    stat the total param num and the num of trainable
    model: the model to be evaluated.
    ret: the dict of "Total" and "Trainable"
    """
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


# from pytorch_lightning import LightningModule
# def get_detail_pl_pnumber(model: LightningModule, model_name='model'):
#     ret = []
#     for submodel in model._modules['model'].children():
#         name = submodel._get_name()
#         total_num = sum(p.numel() for p in submodel.parameters())
#         trainable_num = sum(p.numel() for p in submodel.parameters() if p.requires_grad)
#         ret.append((name, {'Total': total_num, 'Trainable': trainable_num}))
#     return ret
