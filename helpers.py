import torch
import math
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

# Applies learning rate warmup and decay.
def get_lr(iter, max_steps, warmup_steps, max_lr, min_lr):
    # warmup. start near 0 keep increasing.
    if iter < warmup_steps:
        return max_lr * (iter+1) / warmup_steps
    # No more decay
    elif iter > max_steps:
        return min_lr
    # given formula. works I guess.
    decay_ratio = (iter - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)