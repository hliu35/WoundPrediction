import numpy as np
import torch


def synthesize_softmax_labels(n_classes, batch_size, count=1000):
    batch = np.zeros((batch_size, n_classes))
    mu_low = 0
    mu_high = n_classes-1
    std_low = 0.1
    std_high = 0.2

    for i in range(batch_size):
        #mu = np.random.randint(low=-0.48, high=n_classes+0.49)
        mu = np.random.uniform(mu_low, mu_high)
        sigma = np.random.uniform(std_low, std_high)
        
        rand_samples = np.random.normal(loc=mu, scale=sigma, size=count)
        bins = [x-0.5 for x in range(n_classes+1)]
        rand_ints = np.digitize(rand_samples, bins=bins) - 1

        for j in range(n_classes):
            batch[i, j] = np.count_nonzero(rand_ints == j)
        
        batch[i, :] /= np.sum(batch[i, :])
        if 1.0 in batch[i, :]: np.random.shuffle(batch[i, :])
    
    res = torch.FloatTensor(batch)
    return res


def synthesize_onehot_labels(n_classes, batch_size, sorted=True):
    batch = np.zeros((batch_size, n_classes))

    for i in range(batch_size):
        if not sorted: j = np.random.randint(low=0, high=n_classes)
        else: j = i % n_classes
        batch[i, j] = 1.0
    
    return torch.FloatTensor(batch)



if __name__ == "__main__":
    a = synthesize_softmax_labels()
    print(a)