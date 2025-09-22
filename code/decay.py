def linear_decay_weights(K):
    weights = [(K - k + 1) for k in range(1, K + 1)]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    return normalized_weights


def exponential_decay_weights(K, alpha=0.5):
    weights = [alpha ** (K - k) for k in range(1, K + 1)]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    return normalized_weights


def inverse_decay_weights(K):
    weights = [1 / k for k in range(1, K + 1)]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    return normalized_weights


# K = 4
# weights = linear_decay_weights(K)
# print("逆衰减权重:", weights)
