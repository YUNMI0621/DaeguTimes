import db_
from flask import Flask, render_template, request, url_for, redirect
import mariadb
import os
import cgi, sys, codecs, os
from pydoc import html
import joblib
from konlpy.tag import Okt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import model, test

app = Flask(__name__)
template_dir = os.path.join(os.path.dirname(__file__), 'templates')

# ====================== MariaDB 연결 정보 설정 =========================

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'port': 3307,
    'database': 'daegutimes'
}

try:
    conn = mariadb.connect(**db_config)

except mariadb.Error as e:
    print(f"Error connecting to MariaDB: {e}")
    sys.exit(1)

# ======================================================================

from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import pandas as pd
from konlpy.tag import Okt
import re


# 기간 검색해서 키워드 뽑기
day, keyword, content, summarize, title, index_, nouns, senti_score, senti_result, topic = db_.get_keyword(conn)
period = pd.DataFrame({'day':day, 'keyword':keyword, "content":content, "title":title, "index":index_, "nouns":nouns, "senti_score":senti_score, "senti_result": senti_result, "topic":topic})
period.drop_duplicates(subset='content', keep='first', inplace=True)

def dayKeyword(start_, end_, df):
    stop_ = ['대구','부산','전주','서울','이날','전국','경북','의원','국민','정부','조사','경기','과장','검사','관리','광주','부장','인천','지방','지역','팀장','제주','본부','본부장',
             '수준','운영','평가','대전','강원','포인트','경남','강릉','시장','예정','울산','1일','권역','농도','공기','기온','지방','국장','단장','경북지방','기업부','중부지방','구름']

    # Create a pattern for words containing any stop word
    pattern_ = re.compile(r'\b(?:' + '|'.join(stop_) + r')\b')
    # 
    ndf = df[(df.day >= start_) & (df.day <= end_)]
    ndf = ndf.sample(n=130, random_state=20, replace = False)   # 130은 일일 기사 평균값
    ndf.drop_duplicates(subset='title', keep='first', inplace= True)

    ndf['keyword'] = ndf['keyword'].apply(lambda x: pattern_.sub('',x))
    
    tfv = TfidfVectorizer(min_df=5, ngram_range=(2, 4), max_features=5)
    dtm_ = tfv.fit_transform(ndf['keyword'])
    name_ = tfv.get_feature_names_out()
    
    key = []
    for i in name_:
        if len(set(i.split())) != 1:
           key.append(i)
        else:
            key.append(list(set(i.split()))[0]) 
            
    return key

# 신뢰도 db
rtitle, rcontent = db_.get_keyword_reliability(conn)
reliability = pd.DataFrame({'title' : rtitle, 'content' : rcontent})

# 관련 기사 헤드라인 추출 함수

def searchNews(start_, end_, keyword, df, N=10) :
    ndf = df[(df.day >= start_) & (df.day <= end_) ]
    articles = ndf['content'].tolist()
    
    # Convert the articles to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(articles)
    
    # Vectorize the user input
    user_input_vector = vectorizer.transform([keyword])

    # Calculate cosine similarity between the user input and each article
    cosine_similarities = cosine_similarity(user_input_vector, tfidf_matrix)

    # Get indices of the top N most similar articles
    most_similar_indices = cosine_similarities.argsort()[0][-N:][::-1]

    # Print the top N most similar articles
    print(f"Top {N} most similar articles:")

    lst = []
    for index in most_similar_indices:
        similarity_score = cosine_similarities[0][index]
        # print(f"Index: {index}, Similarity: {similarity_score:.4f}")
        # print(ndf.title.iloc[index])
        # print("-----")
        lst.append(ndf.title.iloc[index])

    return lst

# ================ 타이틀 인덱스 df 함수 ==================
def title_index(title, df):
    if len(df[df['title'] == title]) != 0:
        result_ =  df[df['title'] == title]
    else:
        result_ = df.iloc[0]
    return result_

# ======================= 원문 가져오기 ========================
def original_content():
    day, keyword, content, summarize, title, index_, nouns, senti_score, senti_result, topic = db_.get_keyword(conn)
    period = pd.DataFrame({'day':day, 'keyword':keyword, "content":content, "title":title, "index":index_, "nouns":nouns, "senti_score":senti_score, "senti_result": senti_result, "topic":topic})


# ====================== 기능 구현 ==========================
# (1) 요약 분석 모델 호출
# topic_pklfile = os.path.dirname(__file__) + '/make_model/topic.pkl'
# topic_model = joblib.load(topic_pklfile)
# print(topic_model)

# (2) 감성 분류 모델 호출
# pn_pklfile = os.path.dirname(__file__) + "/make_model/pn_model.pkl"
# pn_model = joblib.load(pn_pklfile)

# (3) 토픽 모델링 모델 호출
# topic_pklfile = os.path.dirname(__file__) + '/make_model/topic.pkl'
# topic_model = joblib.load(topic_pklfile)
# print(topic_model)

# (3) 신뢰도 분석 모델 호출


# =================================================================================================

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/main.html')
def main_page():
    return render_template('main.html')

@app.route('/sub.html')
def sub_page():
    return render_template('sub.html')

@app.route('/redirect', methods=['POST'])
def redirect_to_keyword():
    selected_option = request.form.get('lang')
    start_date_value = request.form.get('start_date')
    end_date_value = request.form.get('end_date')

    if selected_option == 'KeyWord_1':
        return redirect(url_for('KeyWord1', start_date=start_date_value, end_date=end_date_value))
    elif selected_option == 'KeyWord_2':
        return redirect(url_for('KeyWord2', start_date=start_date_value, end_date=end_date_value))
    elif selected_option == 'KeyWord_3':
        return redirect(url_for('KeyWord3', start_date=start_date_value, end_date=end_date_value))
    elif selected_option == 'KeyWord_4':
        return redirect(url_for('KeyWord4', start_date=start_date_value, end_date=end_date_value))
    elif selected_option == 'KeyWord_5':
        return redirect(url_for('KeyWord5', start_date=start_date_value, end_date=end_date_value))
    else:
        # Handle other cases or provide a default redirection
        return redirect(url_for('index'))


@app.route('/KeyWord1.html', methods=['GET'])
def KeyWord1():

    _start_date = request.args.get("start_date",None)
    _end_date = request.args.get("end_date",None)
    _value = request.args.get("value",None)

    result = searchNews(_start_date, _end_date, _value, period)

    print(result)
    
    # print(_start_date , _end_date)

    return render_template('KeyWord1.html',_start_date=_start_date,_end_date=_end_date,result=result,value=_value)

# ========================== 요약 분석 ==========================

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



@app.route("/summary.html")
def iframe1():
    encrypted_var = request.args.get("title")

    from urllib import parse

    decrypted_var = parse.unquote(encrypted_var)


    result = title_index(decrypted_var,period)
    result.drop_duplicates(subset='title', keep='first', inplace = True)

    print(result)
    
    real_result = result['content'].tolist()
    original_content = result["content"].tolist()[0]
    
    return render_template('summary.html', title=decrypted_var, result= main(real_result), original=original_content)

# ========================== 감성 분석 ==========================
@app.route("/sentimental.html")
def iframe2():

    encrypted_var = request.args.get("title")

    from urllib import parse

    decrypted_var = parse.unquote(encrypted_var)


    result = title_index(decrypted_var,period)
    result.drop_duplicates(subset='title', keep='first', inplace = True)

    print(result)

    real_result = result["senti_result"]
    original_content = result["content"].tolist()[0]
    
    image=""
    if real_result.values == 1 :
        ret = '긍정'
        image="positive"
    elif real_result.values == -1 :
        ret = '부정'
        image="negative"
    else : 
        ret = '중립'
        image="neutral"

    return render_template('sentimental.html',title=decrypted_var, result= ret, original=original_content, image=image)

# ========================== 토픽 분석 ==========================
@app.route("/topic_modeling.html")
def iframe3():
    encrypted_var = request.args.get("title")

    from urllib import parse

    decrypted_var = parse.unquote(encrypted_var)


    result = title_index(decrypted_var,period)
    result.drop_duplicates(subset='title', keep='first', inplace = True)
    print(result)
    
    image=""
    real_result = result["topic"]
    original_content = result["content"].tolist()[0]
    if real_result.values == 1:
        ret = '부동산'        # 노랑
        image = "house"
    elif real_result.values == 2:
        ret = '정치이슈'      # 보라 
        image="politics"
    elif real_result.values == 3:
        ret = '정책'          # 주황
        image = "law"
    elif real_result.values == 4:
        ret = '사건 사고 범죄' # 빨강
        image = "issue"
    elif real_result.values == 5:
        ret = '건강 및 안전'   # 초록
        image = "health"
    elif real_result.values == 6:
        ret = '날씨 및 기상'   # 파랑
        image = "weather"
    elif real_result.values == 7:
        ret = '기타'           # 블랙
        image = "etc"


    return render_template('topic_modeling.html',title=decrypted_var, result= ret, original=original_content, image=image)


# ========================== 신뢰도 분석 =================================
# 토큰화, 벡터화
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


@app.route("/reliability.html")
def iframe4():
    encrypted_var = request.args.get("title")

    from urllib import parse

    decrypted_var = parse.unquote(encrypted_var)


    result = title_index(decrypted_var, reliability)
    print(result)

    try:
        title_ = result['title'].tolist()
        content_ = result['content'].tolist()
        
    except:
        title_ = result['title']
        content_ = result['content']

    return render_template('reliability.html',title=decrypted_var, result= reliability_input(title_, content_), original=content_)


@app.route("/news1_result.html")
def result1():

    var = request.args.get("title",None)
    
    from urllib import parse

    var = parse.quote(var)

    return render_template('news1_result.html', result_data="가나다",var = var)


@app.route("/request_date", methods=["POST"])
def process_1():
    start_date = request.form.get("start_date",None)
    end_date = request.form.get("end_date",None)

    start_date=start_date.replace("-", "")
    end_date=end_date.replace("-", "")

    result = dayKeyword(start_date,end_date,period)
    print(start_date, "요청에 대해서 " , end_date)
    
    print(result)

    import json

    return json.dumps(result)


# =======================================================
# 최종

# @app.route("/new_predict.html", methods=['GET'])
# def newData():
#     _value = request.args.get("value", None)

#     return render_template('new_predict.html')

# @app.route("/summary_2.html")
# def iframe1222():
#     title = request.args.get("title",None)
#     wonmun = request.args.get("data",None)

#     print(title,"제목 받았고, " + wonmun +"데이터를 받아음")

#     return render_template("summary.html", title="테스트", result= wonmun, original=wonmun)


if __name__ == '__main__':
    app.run(port=5005, debug=True)


