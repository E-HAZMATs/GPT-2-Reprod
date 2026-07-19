import torch

'''
TODO?: checkpoint last model and best model?
'''
def save_ckpt(model: torch.nn.Module, optim, epoch, iter, total_iter, t_loss, e_loss):
    path = 'data/ckpt.pt'
    torch.save({
        'epoch': epoch,
        'iter': iter,
        'total_iter': total_iter,
        'eval_loss': e_loss, # Averaged eval loss.
        'train_loss': t_loss, # XXX: Included to catch overfitting. Is having last tr_loss good enough? or should average `eval_every` losses? 
        'model_sd': model.state_dict(),
        'optim_sd': optim.state_dict(),
    }, path)    