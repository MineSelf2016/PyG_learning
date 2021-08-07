import torch

def deep_learning():

    model = torch.nn.Module("a lot of architecture")

    try:
        model.fit("a lot of hyperparameters", "real dataset")
    except Exception as e:
        print(e, "Fine-tune!")
        deep_learning()

    print("You achieved STOA of this!")