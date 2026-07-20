import torch
'''
TODO?: checkpoint last model and best model?
'''
def save_ckpt(model: torch.nn.Module, optim, epoch, iter, total_iter, t_loss, e_loss, name='ckpt.pt'):
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
    print('checkpoint saved.')


def load_ckpt(model, optim):
    path = 'data/ckpt.pt'

    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint['model_sd'])
    optim.load_state_dict(checkpoint['optim_sd'])
    return dict(
        epoch = checkpoint['epoch'],
        iter = checkpoint['iter'],
        total_iter = checkpoint['total_iter']
        )


# load_ckpt