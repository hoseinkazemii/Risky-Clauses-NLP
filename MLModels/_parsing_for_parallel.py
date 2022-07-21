
def parsing_for_parallel(X, w2indx):

    holder = []
    for sentence in X:
        indexed_sentence = []

        for token in sentence:
            try:
                indexed_sentence.append(w2indx[token])
            except:
                indexed_sentence.append(0)

        holder.append(indexed_sentence)

    return holder