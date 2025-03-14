import torch 

print(f"PyTorch版本: {torch.__version__}") 
print(f"CUDA可用: {torch.cuda.is_available()}") 
print("测试张量运算:", torch.rand(3,3) @ torch.rand(3,3))
