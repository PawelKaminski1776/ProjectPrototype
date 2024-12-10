import torch

if __name__ == '__main__':

     #Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available!")
        print("Number of GPUs:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Using CPU.")
