from dataset import Mazurkas, RondoDB, Covers80
from models import (
    QuantizedFullModel,
    QuantizedSmallModel,
    PrunedFullModel,
    PrunedSmallModel,
    QuantizedPrunedFullModel,
    QuantizedPrunedSmallModel,
)

if __name__ == "__main__":
    datasets = [Mazurkas(), RondoDB(), Covers80()]
    for dataset in datasets:
        # quantized full model
        model = QuantizedFullModel(dataset.num_classes)
        model.train(dataset)
        model.convert(dataset)

        # quantized small model
        model = QuantizedSmallModel(dataset.num_classes)
        model.train(dataset)
        model.convert(dataset)

        # pruned full model
        model = PrunedFullModel(dataset.num_classes)
        model.train(dataset)
        model.convert(dataset)

        # pruned small model
        model = PrunedSmallModel(dataset.num_classes)
        model.train(dataset)
        model.convert(dataset)

        # quantized pruned full model
        model = QuantizedPrunedFullModel(dataset.num_classes)
        model.train(dataset)
        model.convert(dataset)

        # quantized pruned small model
        model = QuantizedPrunedSmallModel(dataset.num_classes)
        model.train(dataset)
        model.convert(dataset)
