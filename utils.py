
def set_torch_random_seed(seed=0):
    import torch
    torch.manual_seed(seed)
    # set CUDA seeds
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU