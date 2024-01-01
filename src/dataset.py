from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def training_dataset():
        pass

    @abstractmethod
    def validation_dataset():
        pass

    @abstractmethod
    def test_dataset():
        pass


class Mazurkas(Dataset):
    pass


class RondoDB(Dataset):
    pass


class Covers80(Dataset):
    pass
