import numpy as np
from torch.utils.data.sampler import Sampler

from config import get_config


class OrderedSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.labels_unique = np.unique(labels)
        self.batch_size = batch_size

        self.label_indexes = {}
        for i, label in enumerate(labels):
            self.label_indexes.setdefault(label, []).append(i)

    def __iter__(self):
        config = get_config()
        labels = np.random.choice(self.labels_unique, config.label_count, replace=False)
        image_count = 256 / config.label_count

        for i in range(self.__len__()):
            inds = np.array([], dtype=np.int)
            for label in labels:
                subsample = np.random.choice(self.label_indexes[label], image_count, replace=False)
                inds = np.append(inds, subsample)

            yield list(inds)

    def __len__(self):
        return len(self.labels) // 256


def select_triplets(embeddings, images_per_class, label_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    for i in xrange(label_per_batch):
        for j in xrange(1, images_per_class):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, j + images_per_class):
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx + j + images_per_class] = np.NaN
                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((embeddings[a_idx], embeddings[p_idx], embeddings[n_idx]))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets
