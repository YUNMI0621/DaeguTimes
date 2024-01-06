
# 요약분석

from test import MySummarizer

def main(text):

    # Loading the trained weights
    weights_path = 'make_model/summary.ckpt'
    model_name = "digit82/kobart-summarization"

    # Create the summarizer
    summarizer = MySummarizer(model_name, weights_path)

    # Generate and print summary
    summary = summarizer.generate_summary(text)
    return summary


# 감성분석

def pn_search(text):
    from konlpy.tag import Okt
    from sklearn.feature_extraction.text import CountVectorizer
    import pickle

    okt = Okt()
    nouns_ = okt.nouns(text)

    model = pickle.load(open("./make_model/pn_model.pkl", mode="rb"))
    cv = pickle.load(open("./make_model/model_cv.pb", mode="rb"))
    cv2 = CountVectorizer(vocabulary=cv.vocabulary_)
    dtm = cv2.fit_transform([' '.join(nouns_)])

    result = model.predict(dtm)

    sentiment = '긍정' if result == 1 else ('부정' if result == -1 else '중립')
    return f'예측 감성: {sentiment}'



# 토픽모델링

def topic_search(text):
    
    import pandas as pd
    import pickle
    from konlpy.tag import Okt

    okt = Okt()
    text = okt.nouns(text)

    model = pickle.load(open("./make_model/topic_model.pk", mode="rb"))

    predicted_category = model.predict([text])
    
    # Print the predicted category
    ss = predicted_category[0]
    # ret = set(ret)

    if ss == 1:
        ret = '부동산'
    elif ss == 2:
        ret = '정치이슈'
    elif ss == 3:
        ret = '정책'
    elif ss == 4:
        ret = '사건 사고 범죄'
    elif ss == 5:
        ret = '건강 및 안전'
    elif ss == 6:
        ret = '날씨 및 기상'
    elif ss == 7:
        ret = '기타'
    
    return f'Predicted Category : {ret}'


# 신뢰도분석

from keras import preprocessing
from keras.utils import pad_sequences
from keras.models import load_model

def reliability_input(title, content):
    import numpy as np
    vocab_size = 1000            # 상위 1000개의 고유한 단어로 제한한다는 의미
    max_sequence_length = 100    # 시퀀스에 허용되는 최대 토큰 수(단어,문자  단위)
    
    # num_words에 지정된 만큼만 숫자로 반환 - 부족하면 0으로 채우고, 넘치면 잘림
    tokenizer = preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<oov>")    # oov_tok을 사용하여 사전에 없는 단어집합을 만듬
    tokenizer.fit_on_texts([title])
    title_sequences = tokenizer.texts_to_sequences([title])
    title_padded = pad_sequences(title_sequences, maxlen=max_sequence_length)
    tokenizer.fit_on_texts([content])
    content_sequences = tokenizer.texts_to_sequences([content])
    content_padded = pad_sequences(content_sequences, maxlen=max_sequence_length)

    # Load the saved model
    loaded_model = load_model("make_model/reliability.h5")
    # Make predictions
    predictions = loaded_model.predict([title_padded, content_padded])
    predictions = predictions + 15
    predictions = np.clip(predictions, a_min=0,a_max=100)
    predictions = int(predictions.round())

    return predictions

