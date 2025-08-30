"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-04-04 13:38:43
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-04-04 15:16:55
"""

import os
import numpy as np
import torch

from torch import Tensor
from torchpack.datasets.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_url
from typing import Any, Callable, Dict, List, Optional, Tuple


__all__ = ["Vowel"]


class VowelRecognition(VisionDataset):
    """Vowel Recognition dataset.

    Attributes:
        classes (list): List of classes in the dataset.
        class_to_idx (dict): Mapping from class names to class indices.
        idx_to_class (dict): Mapping from class indices to class names.
        samples (list): List of (sample path, class index) tuples.
        n_features (int): Number of features to consider from the dataset.
    """

    url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases"
        "/undocumented/connectionist-bench/vowel/vowel-context.data"
    )
    filename = "vowel-context.data"
    folder = "vowel"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        n_features: int = 10,
        train_ratio: float = 0.7,
        download: bool = False,
    ) -> None:
        """Initialize the Vowel Recognition dataset.
        
        Args:
            root (str): Root directory of the dataset.
            train (bool, optional): Determines whether to load the training set or the test set. 
                Defaults to True (training set).
            transform (callable, optional): A function/transform that takes in the raw data and returns a transformed version.
                Defaults to None.
            target_transform (callable, optional): A function/transform that takes in the target and returns a transformed version.
                Defaults to None.
            n_features (int, optional): Number of features to consider from the dataset.
                Defaults to 10.
            train_ratio (float, optional): Ratio of training samples to use if the dataset is split into training and test sets.
                Defaults to 0.7.
            download (bool, optional): If True, downloads the dataset from the internet and places it in the root directory.
                Defaults to False.

        Raises:
            RuntimeError: If the dataset is not found or corrupted and download is not enabled.

        Examples:
            >>> dataset = VowelRecognition(root='data', train=True, transform=None, download=True)
        """

        root = os.path.join(os.path.expanduser(root), self.folder)
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])
        super(VowelRecognition, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.train = train  # training set or test set
        self.train_ratio = train_ratio

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        self.n_features = n_features
        assert 1 <= n_features <= 10, print(
            f"Only support maximum 10 features, but got{n_features}"
        )
        self.data: Any = []
        self.targets = []

        self.process_raw_data()
        self.data, self.targets = self.load(train=train)

    def process_raw_data(self) -> None:
        """Process the raw data of the dataset.

        This method is called during initialization to load and process the raw data into a suitable format for the dataset.

        Returns:
            None.

        Examples:
            >>> dataset = VowelRecognition(root='data', train=True, transform=None, download=True)
            >>> dataset.process_raw_data()
        """

        processed_dir = os.path.join(self.root, "processed")
        processed_training_file = os.path.join(processed_dir, "training.pt")
        processed_test_file = os.path.join(processed_dir, "test.pt")
        if os.path.exists(processed_training_file) and os.path.exists(
            processed_test_file
        ):
            with open(os.path.join(self.root, "processed/training.pt"), "rb") as f:
                data, targets = torch.load(f)
                if data.shape[-1] == self.n_features:
                    print("Data already processed")
                    return
        data, targets = self._load_dataset()
        data_train, targets_train, data_test, targets_test = self._split_dataset(
            data, targets
        )
        data_train, data_test = self._preprocess_dataset(data_train, data_test)
        self._save_dataset(
            data_train, targets_train, data_test, targets_test, processed_dir
        )

    def _load_dataset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load the dataset from the raw data file.

        Returns:
            data_train, targets_train, data_test, targets_test
            tuple: A tuple containing the data tensor and the target tensor.

        Examples:
            >>> data, targets = dataset._load_dataset()
            >>> data.shape
            torch.Size([528, 10])
            >>> targets.shape
            torch.Size([528])
        """
        
        data = []
        targets = []
        with open(os.path.join(self.root, "raw", self.filename), "r") as f:
            for line in f:
                line = line.strip().split()[3:]
                label = int(line[-1])
                targets.append(label)
                example = [float(i) for i in line[:-1]]
                data.append(example)

            data = torch.Tensor(data)
            targets = torch.LongTensor(targets)
        return data, targets

    def _split_dataset(self, data: Tensor, targets: Tensor) -> Tuple[Tensor, ...]:
        """Split the dataset into training and test sets.

        Args:
            data (torch.Tensor): The input data tensor.
            targets (torch.Tensor): The target tensor.

        Returns:
            tuple: A tuple containing the training data, training targets, test data, and test targets.

        Examples:
            >>> dataset = VowelRecognition(root='data', train=True, transform=None, download=True)
            >>> data, targets = dataset._load_dataset()
            >>> data_train, targets_train, data_test, targets_test = dataset._split_dataset(data, targets)
        """

        from sklearn.model_selection import train_test_split

        data_train, data_test, targets_train, targets_test = train_test_split(
            data, targets, train_size=self.train_ratio, random_state=42
        )
        print(
            f"training: {data_train.shape[0]} examples, "
            f"test: {data_test.shape[0]} examples"
        )
        return data_train, targets_train, data_test, targets_test

    def _preprocess_dataset(
        self, data_train: Tensor, data_test: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Preprocess the dataset by applying PCA and scaling transformations.

        Args:
            data_train (torch.Tensor): The training data tensor.
            data_test (torch.Tensor): The test data tensor.

        Returns:
            tuple: A tuple containing the preprocessed training data and test data.

        Examples:
            >>> dataset = VowelRecognition(root='data', train=True, transform=None, download=True)
            >>> data, targets = dataset._load_dataset()
            >>> data_train, targets_train, data_test, targets_test = dataset._split_dataset(data, targets)
            >>> preprocessed_train_data, preprocessed_test_data = dataset._preprocess_dataset(data_train, data_test)
        """

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import MinMaxScaler, RobustScaler

        pca = PCA(n_components=self.n_features)
        data_train_reduced = pca.fit_transform(data_train)
        data_test_reduced = pca.transform(data_test)

        rs = RobustScaler(quantile_range=(10, 90)).fit(
            np.concatenate([data_train_reduced, data_test_reduced], 0)
        )
        data_train_reduced = rs.transform(data_train_reduced)
        data_test_reduced = rs.transform(data_test_reduced)
        mms = MinMaxScaler()
        mms.fit(np.concatenate([data_train_reduced, data_test_reduced], 0))
        data_train_reduced = mms.transform(data_train_reduced)
        data_test_reduced = mms.transform(data_test_reduced)

        return (
            torch.from_numpy(data_train_reduced).float(),
            torch.from_numpy(data_test_reduced).float(),
        )

    @staticmethod
    def _save_dataset(
        data_train: Tensor,
        targets_train: Tensor,
        data_test: Tensor,
        targets_test: Tensor,
        processed_dir: str,
    ) -> None:
        """Save the preprocessed dataset to disk.

        Args:
            data_train (torch.Tensor): The preprocessed training data tensor.
            targets_train (torch.Tensor): The training targets tensor.
            data_test (torch.Tensor): The preprocessed test data tensor.
            targets_test (torch.Tensor): The test targets tensor.
            processed_dir (str): The directory to save the processed dataset.

        Returns:
            None.

        Examples:
            >>> dataset = VowelRecognition(root='data', train=True, transform=None, download=True)
            >>> data, targets = dataset._load_dataset()
            >>> data_train, targets_train, data_test, targets_test = dataset._split_dataset(data, targets)
            >>> preprocessed_train_data, preprocessed_test_data = dataset._preprocess_dataset(data_train, data_test)
            >>> dataset._save_dataset(preprocessed_train_data, targets_train, preprocessed_test_data, targets_test, 'processed')
            Processed dataset saved
        """

        os.makedirs(processed_dir, exist_ok=True)
        processed_training_file = os.path.join(processed_dir, "training.pt")
        processed_test_file = os.path.join(processed_dir, "test.pt")
        with open(processed_training_file, "wb") as f:
            torch.save((data_train, targets_train), f)

        with open(processed_test_file, "wb") as f:
            torch.save((data_test, targets_test), f)
        print(f"Processed dataset saved")

    def load(self, train: bool = True):
        """Load the dataset.

        This method loads the dataset from the processed data and returns the data and target labels.

        Args:
            train (bool, optional): Determines whether to load the training set or the test set.
                Defaults to True (training set).

        Returns:
            data: Loaded data.
            targets: Target labels.

        Examples:
            >>> dataset = VowelRecognition(root='data', train=True, transform=None, download=True)
            >>> data, targets = dataset.load(train=True)
        """

        filename = "training.pt" if train else "test.pt"
        with open(os.path.join(self.root, "processed", filename), "rb") as f:
            data, targets = torch.load(f)
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if isinstance(targets, np.ndarray):
                targets = torch.from_numpy(targets)
        return data, targets

    def download(self) -> None:
        """Download the dataset.

        This method downloads the dataset from the internet and places it in the root directory.

        Returns:
            None

        Examples:
            >>> dataset = VowelRecognition(root='data', train=True, transform=None, download=True)
            >>> dataset.download()
        """

        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_url(
            self.url, root=os.path.join(self.root, "raw"), filename=self.filename
        )

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset.

        This method checks if the dataset exists and is not corrupted.

        Returns:
            bool: True if the dataset is found and intact, False otherwise.

        Examples:
            >>> dataset = VowelRecognition(root='data', train=True, transform=None, download=True)
            >>> dataset._check_integrity()
            True
        """
        
        return os.path.exists(os.path.join(self.root, "raw", self.filename))

    def __len__(self):
        """Get the number of items in the dataset.

        Returns:
            int: Number of items in the dataset.

        Examples:
            >>> dataset = VowelRecognition(root='data', train=True, transform=None, download=True)
            >>> len(dataset)
            143
        """

        return self.targets.size(0)

    def __getitem__(self, item):
        """Get a specific item from the dataset.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: A tuple containing the transformed item data and its corresponding target class index.

        Examples:
            >>> dataset = VowelRecognition(root='data', train=True, transform=None, download=True)
            >>> img, target = dataset[0]
        """

        return self.data[item], self.targets[item]

    def extra_repr(self) -> str:
        """Return a string representation of the dataset's split (train or test).

        Returns:
            str: A string indicating the dataset's split.

        Examples:
            >>> dataset = VowelRecognition(root='data', train=True, transform=None, download=True)
            >>> dataset.extra_repr()
            'Split: Train'
        """

        return "Split: {}".format("Train" if self.train is True else "Test")


class VowelRecognitionDataset:
    """Vowel Recognition dataset.
    
    Attributes:
        root (str): Root directory of the dataset.
        split (str): Split name ('train', 'valid', or 'test').
        test_ratio (float): Ratio of data to use for testing.
        train_valid_split_ratio (List[float]): Ratio of data to use for training and validation split.
        data (Dataset): Loaded dataset.
        resize (int): Size to resize the input data.
        binarize (bool): Whether to binarize the input data.
        binarize_threshold (float): Threshold value for binarization.
        digits_of_interest (List[int]): List of digits of interest.
        n_instance (int): Number of instances in the dataset.
    """

    def __init__(
        self,
        root: str,
        split: str,
        test_ratio: float,
        train_valid_split_ratio: List[float],
        resize: int,
        binarize: bool,
        binarize_threshold: float,
        digits_of_interest: List[int],
    ):
        """Initialize Vowel Recognition dataset.
        
        Args:
            root (str): Root directory of the dataset.
            split (str): Split name ('train', 'valid', or 'test').
            test_ratio (float): Ratio of data to use for testing.
            train_valid_split_ratio (List[float]): Ratio of data to use for training and validation split.
            resize (int): Size to resize the input data.
            binarize (bool): Whether to binarize the input data.
            binarize_threshold (float): Threshold value for binarization.
            digits_of_interest (List[int]): List of digits of interest.
            
        Returns:
            None.
        
        Raises:
            AssertionError: If `test_ratio` is not within the range (0, 1).

        Examples:
            >>> dataset = VowelRecognitionDataset(
            >>>     root='data',
            >>>     split='train',
            >>>     test_ratio=0.2,
            >>>     train_valid_split_ratio=[0.8, 0.2],
            >>>     resize=32,
            >>>     binarize=True,
            >>>     binarize_threshold=0.5,
            >>>     digits_of_interest=[0, 1, 2],
            >>> )
        """
        
        self.root = root
        self.split = split
        self.test_ratio = test_ratio
        assert 0 < test_ratio < 1, print(
            f"Only support test_ratio from (0, 1), but got {test_ratio}"
        )
        self.train_valid_split_ratio = train_valid_split_ratio
        self.data = None
        self.resize = resize
        self.binarize = binarize
        self.binarize_threshold = binarize_threshold
        self.digits_of_interest = digits_of_interest

        self.load()
        self.n_instance = len(self.data)

    def load(self):
        """Load the dataset based on the split and other parameters.
        
        Returns:
            None.
        """

        tran = [transforms.ToTensor()]
        transform = transforms.Compose(tran)

        if self.split == "train" or self.split == "valid":
            train_valid = VowelRecognition(
                self.root,
                train=True,
                download=True,
                transform=transform,
                n_features=self.resize,
                train_ratio=1 - self.test_ratio,
            )
            idx, _ = torch.stack(
                [train_valid.targets == number for number in self.digits_of_interest]
            ).max(dim=0)
            train_valid.targets = train_valid.targets[idx]
            train_valid.data = train_valid.data[idx]

            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            split = [train_len, len(train_valid) - train_len]
            train_subset, valid_subset = torch.utils.data.random_split(
                train_valid, split, generator=torch.Generator().manual_seed(1)
            )

            if self.split == "train":
                self.data = train_subset
            else:
                self.data = valid_subset

        else:
            test = VowelRecognition(
                self.root,
                train=False,
                download=True,
                transform=transform,
                n_features=self.resize,
                train_ratio=1 - self.test_ratio,
            )
            idx, _ = torch.stack(
                [test.targets == number for number in self.digits_of_interest]
            ).max(dim=0)
            test.targets = test.targets[idx]
            test.data = test.data[idx]

            self.data = test

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Get a specific instance from the dataset.

        Args:
            index (int): Index of the instance to retrieve.

        Returns:
            Dict[str, Tensor]: A dictionary containing the input data and label.

        Examples:
            >>> instance = dataset[0]
        """

        data = self.data[index][0]
        if self.binarize:
            data = 1.0 * (data > self.binarize_threshold) + -1.0 * (
                data <= self.binarize_threshold
            )

        digit = self.digits_of_interest.index(self.data[index][1])
        instance = {"vowel": data, "digit": digit}
        return instance

    def __len__(self) -> int:
        """Get the number of instances in the dataset.

        Returns:
            int: Number of instances in the dataset.

        Examples:
            >>> len(dataset)
            10000
        """

        return len(self.data)

    def __call__(self, index: int) -> Dict[str, Tensor]:
        """Call the dataset to retrieve a specific instance.

        Args:
            index (int): Index of the instance to retrieve.

        Returns:
            Dict[str, Tensor]: A dictionary containing the input data and label.

        Examples:
            >>> instance = dataset(0)
        """
        
        return self.__getitem__(index)


class Vowel(Dataset):
    """Vowel dataset.
    
    Attributes:
        root (str): Root directory of the dataset.
        splits (Dict[str, VowelRecognitionDataset]): Dictionary of dataset splits.
    """

    def __init__(
        self,
        root: str,
        test_ratio: float,
        train_valid_split_ratio: List[float],
        resize=28,
        binarize=False,
        binarize_threshold=0.1307,
        digits_of_interest=tuple(range(10)),
    ):
        """Initialize Vowel dataset.

        Args:
            root (str): Root directory of the dataset.
            test_ratio (float): Ratio of test examples.
            train_valid_split_ratio (List[float]): Ratios of train and validation examples.
            resize (int, optional): Size to resize the images.
                Defaults to 28.
            binarize (bool, optional): Whether to binarize the images.
                Defaults to False.
            binarize_threshold (float, optional): Threshold for binarization.
                Defaults to 0.1307.
            digits_of_interest (Tuple[int], optional): Tuple of digits to include.
                Defaults to tuple(range(10)).
            
        Returns:
            None.
        
        Examples:
            >>> dataset = Vowel(root='data', test_ratio=0.2, train_valid_split_ratio=[0.8, 0.2])
        """

        self.root = root

        super().__init__(
            {
                split: VowelRecognitionDataset(
                    root=root,
                    split=split,
                    test_ratio=test_ratio,
                    train_valid_split_ratio=train_valid_split_ratio,
                    resize=resize,
                    binarize=binarize,
                    binarize_threshold=binarize_threshold,
                    digits_of_interest=digits_of_interest,
                )
                for split in ["train", "valid", "test"]
            }
        )


def test_vowel():
    """Test the Vowel dataset.
    
    Returns:
        None.

    Examples:
        >>> test_vowel()
    """

    import pdb

    pdb.set_trace()
    vowel = VowelRecognition(root=".", download=True, n_features=6)
    print(vowel.data.size(), vowel.targets.size())
    vowel = VowelRecognitionDataset(
        root=".",
        split="train",
        test_ratio=0.3,
        train_valid_split_ratio=[0.9, 0.1],
        resize=8,
        binarize=0,
        binarize_threshold=0,
        digits_of_interest=tuple(range(10)),
    )
    # digits_of_interest=(3, 6))
    print(vowel(20))


if __name__ == "__main__":
    test_vowel()
