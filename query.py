from time import time
from models.NaiveBayes import NaiveBayes
from models.PGMNaiveBayes import PGMNaiveBayes

def query_time(model, neutral, iterations=1):
    categories = ('positive', 'negative', 'neutral') if neutral else ('positive', 'negative')
    start = time()
    quantity = 0
    for _ in range(iterations):
        for category in categories:
            quantity += 1
            model.words(category)
    return quantity / (time() - start)

def query(model, name, neutral):
    positive = model.words('positive')['positive']
    negative = model.words('negative')['negative']
    neutral = model.words('neutral')['neutral'] if neutral else None

    most_positive = positive.sort_values(ascending=False)[0:10] * 100
    most_negative = negative.sort_values(ascending=False)[0:10] * 100
    most_neutral = (neutral.sort_values(ascending=False)[0:10] * 100) if neutral is not None else None
    diff_p = ((positive - negative) / (positive + negative)).sort_values(ascending=False)[0:10] * 100
    diff_n = ((negative - positive) / (positive + negative)).sort_values(ascending=False)[0:10] * 100

    perf1 = query_time(model, neutral is not None)
    perf2 = 1000/perf1 if perf1 != 0 else 0

    print(
    '''*** {name} ***

        References:
            P = probability
            C = category (positive / negative{net_slash})
            w = word
        
        positive probability => P(C=positive): {pos:.2f}%
        negative probability => P(C=negative): {neg:.2f}%{net}

        query performance:
            {n_query_1:.3f} query/s
            {n_query_2:.3f} ms/query

        10 most used words => P(w|C):

        positive:
                {m_pos}
        
        negative:
                {m_neg}
        {m_neutral}
        difference => [ P(w|C=positive) - P(w|C=negative) ] / P(w):

                {m_pos_neg}

                {m_neg_pos}

    '''.format_map({
        'name': name,
        'net_slash': ' / neutral' if neutral is not None else '',
        'pos': model.category_probability('positive') * 100,
        'neg': model.category_probability('negative') * 100,
        'net': '\n\tneutral probability => P(C=neutral): {:.2f}%'.format(model.category_probability('neutral') * 100) if neutral is not None else '',
        'n_query_1': perf1,
        'n_query_2': perf2,
        'm_pos': str.join('\n\t\t', [ '"{}":\t{:.2f}%'.format(k, v) for k, v in most_positive.items() ]),
        'm_neg': str.join('\n\t\t', [ '"{}":\t{:.2f}%'.format(k, v) for k, v in most_negative.items() ]),
        'm_pos_neg': str.join('\n\t\t', [ '"{}\t+{:.2f}%'.format(k + ('":' if len(k) > 10 else '":  \t'), v) for k, v in diff_p.items() ]),
        'm_neg_pos': str.join('\n\t\t', [ '"{}\t-{:.2f}%'.format(k + ('":' if len(k) > 11 else '": \t'), v) for k, v in diff_n.items() ]),
        'm_neutral': ('\n\tneutral:\n\t\t' + str.join('\n\t\t', [ '"{}":\t{:.2f}%'.format(k, v) for k, v in most_neutral.items() ]) + '\n') if neutral is not None else '',
    }))


print('=== NAIVE BAYES CUSTOM IMPLEMENTATION ===')

direct_model = NaiveBayes('./dataset/models/custom.direct.json')
query(direct_model, 'DIRECT MODEL', False)

netural_model = NaiveBayes('./dataset/models/custom.neutral.json')
query(netural_model, 'NEUTRAL MODEL', True)

pgm_direct_model = PGMNaiveBayes('./dataset/models/pgm.direct.json')
query(pgm_direct_model, 'PGM DIRECT MODEL', False)
