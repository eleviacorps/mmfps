import torch
cuda = torch.cuda.is_available()
print(f"CUDA: {cuda}")
if cuda:
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
