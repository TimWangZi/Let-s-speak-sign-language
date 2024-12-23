from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

MODEL_LOAD_PATH = './model/embedding1.model'

model = KeyedVectors.load_word2vec_format(MODEL_LOAD_PATH, binary=True)
print(model.most_similar('大雄' ,topn=10))
