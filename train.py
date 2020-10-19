from tqdm import tqdm
from models.NaiveBayes import NaiveBayes
from models.PGMNaiveBayes import PGMNaiveBayes
from data.DataSet import DataSet
from functools import partial
from time import time

print('Collecting data (can take minutes)...')
with tqdm() as bar:
    def update_bar(data, index, total):
        if total is not None: bar.total = total
        bar.update()
    data = DataSet.FromJSON('./dataset/data.json', './dataset/keywords.json', on_generate=update_bar)

direct_data = data[ lambda x: not not x.score ]

def train(Model, name, path, ds):
    print('Training Bayes Network (can take several minutes): {} - {}'.format(Model.__name__, name))
    model = Model()
    start = time()
    model.fit(ds)
    stop = time()
    print('Trained in {:.3f}s'.format(stop - start))
    print('Saving Bayes Network: {} - {} @ {}'.format(Model.__name__, name, path))
    model.save(path)
    return model

# train(NaiveBayes, 'Custom Direct', './dataset/models/custom.direct.json', direct_data)
# train(NaiveBayes, 'Custom Neutral', './dataset/models/custom.neutral.json', data)

train(PGMNaiveBayes, 'PGM Direct', './dataset/models/pgm.direct.json', direct_data)
# train(PGMNaiveBayes, 'PGM Neutral', './dataset/models/pgm.neutral.json', data)