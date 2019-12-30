import utils
from skipgram import word2vec

def main():
    corpus = [['the','quick','brown','fox','jumped','over','the','lazy','dog']]
    obj = word2vec(corpus,window_size=3)
    emb = obj.get_embeddings()
    print(emb)

if __name__ == '__main__':
    main()