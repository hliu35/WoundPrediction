import numpy as np
import torch


def synthesize_softmax_labels(n_classes=4, batch_size=4, count=1000):
    batch = np.zeros((batch_size, n_classes))

    for i in range(batch_size):
        mu = np.random.randint(low=-0.49, high=n_classes+0.49)
        sigma = np.random.uniform(0.1, 0.5)
        
        rand_samples = np.random.normal(loc=mu, scale=sigma, size=count)
        rand_samples = np.round(rand_samples)

        for j in range(n_classes):
            batch[i, j] = np.count_nonzero(rand_samples == j)
        
        batch[i, :] /= np.sum(batch[i, :])
        if 1.0 in batch[i, :]: np.random.shuffle(batch[i, :])
        
    return torch.FloatTensor(batch)


def synthesize_onehot_labels(n_classes=4, batch_size=4, fixed=True):
    batch = np.zeros((batch_size, n_classes))

    for i in range(batch_size):
        if not fixed: j = np.random.randint(low=0, high=n_classes)
        else: j = i % n_classes
        batch[i, j] = 1.0
    
    return torch.FloatTensor(batch)



if __name__ == "__main__":
    a = synthesize_softmax_labels()
    print(a)