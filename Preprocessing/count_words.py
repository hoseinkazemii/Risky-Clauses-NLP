from collections import Counter


def count_words(X_tokenized, **params):
    print('counting each words number in the whole Dataset...')

    count = Counter()

    for clause in X_tokenized:
        for word in clause:
            count[word] += 1

    return count