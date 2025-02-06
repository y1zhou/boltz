"""Data preparation module for BoltzCraft."""

from torch.utils.data import DataLoader

from boltz.data.module.inference import (
    BoltzInferenceDataModule,
    PredictionDataset,
    collate,
)


class BoltzCraftDataset(PredictionDataset):
    """Base iterable dataset for using Boltz as a design model."""

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Get a sample from the dataset
        features = super().__getitem__(idx)

        return features


class BoltzCraftDataModule(BoltzInferenceDataModule):
    """DataModule for BoltzCraft inference."""

    def predict_dataloader(self) -> DataLoader:
        """Get the training dataloader.

        Returns
        -------
        DataLoader
            The training dataloader.

        """
        dataset = BoltzCraftDataset(
            manifest=self.manifest,
            target_dir=self.target_dir,
            msa_dir=self.msa_dir,
        )
        return DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate,
        )
