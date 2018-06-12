import math

import numpy as np
from torch.utils.data.sampler import BatchSampler


class BalancedBatchSampler(BatchSampler):

    def __init__(self, image_folder_dataset, n_classes, n_samples):
        self.labels = [x[1] for x in image_folder_dataset.imgs]

        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = [x[0] for x in image_folder_dataset.imgs]
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        #while self.count + self.batch_size < len(self.dataset):
        while True:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return int(math.ceil(len(self.dataset) / float(self.batch_size)))
