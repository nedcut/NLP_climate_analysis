import numpy as np

def load_glove_embeddings(glove_path, word_index, embedding_dim=100):
    """
    Loads GloVe embeddings into an embedding matrix aligned with the tokenizer word index.
    """
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vector

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        vec = embeddings_index.get(word)
        if vec is not None:
            embedding_matrix[i] = vec
    return embedding_matrix
