import torch


def check_torch_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(device)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available. Using device: MPS")
    else:
        device = torch.device("cpu")
        print("CUDA and MPS are not available. Using device: CPU")

    return device


def main():
    device = check_torch_device()
    # Example tensor operation to verify device functionality
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    print(f"Tensor on {device}: {x}")


if __name__ == "__main__":
    main()
