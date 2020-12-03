from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as dsets


def get_data():
    dataset = dsets.MNIST(
        root='MNIST',
        train=True,
        transform=T.ToTensor(),
        download=True
    )
    return dataset


def get_dataloader():
    loader = DataLoader(
        dataset=get_data(),
        batch_size=32,
        shuffle=True
    )
    return loader


if __name__ == '__main__':
    data = get_data()
    loader = get_dataloader()
    for x, y in loader:
        print(x.shape)
        break