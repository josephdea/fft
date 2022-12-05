import numpy as np
from .autograd import Tensor
import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
from needle import backend_ndarray as nd


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if not flip_img:
            return img
        return img[:, ::-1, :]
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        ### BEGIN YOUR SOLUTION
        padded = np.pad(img, ((self.padding, self.padding),
                              (self.padding, self.padding),
                              (0, 0)))
        shift_x += self.padding
        shift_y += self.padding
        return padded[shift_x:shift_x + img.shape[0], shift_y:shift_y + img.shape[1], :]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.ordering = None
        self._i = 0

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(indices)
        self.ordering = np.array_split(indices,
                                       range(self.batch_size, len(self.dataset), self.batch_size))
        self._i = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self._i == len(self.ordering):
            raise StopIteration
        batch = tuple(map(Tensor, self.dataset[self.ordering[self._i]]))
        self._i += 1
        return batch
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        with gzip.open(image_filename, 'rb') as image_file:
            with gzip.open(label_filename, 'rb') as label_file:
                # magic numbers
                image_file.read(4)
                label_file.read(4)

                # number of items
                image_count = int.from_bytes(image_file.read(4), 'big')
                label_count = int.from_bytes(label_file.read(4), 'big')
                self._length = label_count

                # actual images and labels
                image_rows = int.from_bytes(image_file.read(4), 'big')
                image_cols = int.from_bytes(image_file.read(4), 'big')
                image_bytes = image_rows * image_cols

                X = np.empty((image_count, image_rows, image_cols, 1), dtype=np.float32)
                y = np.empty((label_count,), dtype=np.uint8)

                for i in range(image_count):
                    image = struct.unpack('c' * image_bytes,
                                          image_file.read(image_bytes))
                    image = map(lambda x: int.from_bytes(x, 'big'), image)
                    label = int.from_bytes(label_file.read(1), 'big')
                    X[i, :, :, 0] = np.fromiter(image, dtype=np.float32).reshape(image_rows, image_cols) / 255
                    y[i] = label

        self._X = X
        self._y = y
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        return self.apply_transforms(self._X[index]), self._y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self._length
        ### END YOUR SOLUTION


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        if train:
            filenames = [os.path.join(base_folder, 'data_batch_{}'.format(i + 1)) for i in range(5)]
        else:
            filenames = [os.path.join(base_folder, 'test_batch')]
        
        X = []
        y = []
        
        for filename in filenames:
            with open(filename, 'rb') as f:
                d = pickle.load(f, encoding='bytes')
                X.extend(d[b'data'])
                y.extend(d[b'labels'])
        
        self._X = np.array(X, dtype='float32')
        self._y = np.array(y, dtype='float32')
        
        self._X /= 255
        self._X = self._X.reshape(-1, 3, 32, 32)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        return self.apply_transforms(self._X[index]), self._y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self._y)
        ### END YOUR SOLUTION


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])


class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        ### BEGIN YOUR SOLUTION
        if word in self.word2idx:
            return self.word2idx[word]

        index = len(self.idx2word)  # this works because 0 indexing
        self.idx2word.append(word)
        self.word2idx[word] = index
        return index
        ### END YOUR SOLUTION

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        ### BEGIN YOUR SOLUTION
        assert len(self.word2idx) == len(self.idx2word)
        return len(self.idx2word)
        ### END YOUR SOLUTION



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        ### BEGIN YOUR SOLUTION
        ids = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                if i == max_lines:
                    break
                for word in line.strip().split():
                    if word:
                        ids.append(self.dictionary.add_word(word))
                ids.append(self.dictionary.add_word('<eos>'))
        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    ### BEGIN YOUR SOLUTION
    result = np.fromiter(data, dtype, count=len(data) - (len(data) % batch_size))
    result = result.reshape((batch_size, -1))
    return result.T
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    ### BEGIN YOUR SOLUTION
    batch_size = min(bptt, len(batches) - 1 - i)
    data = batches[i:i + batch_size]
    target = batches[i + 1:i + 1 + batch_size].flatten()
    return Tensor(data, device=device, dtype=dtype), Tensor(target, device=device, dtype=dtype)
    ### END YOUR SOLUTION
