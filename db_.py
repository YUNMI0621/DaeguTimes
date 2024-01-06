def get_keyword(conn):
    cursor = conn.cursor()

    tb_keyword = '기간키워드'
    tb_summary = '학습데이터_요약'
    tb_senti = '학습데이터_감성'
    tb_topic = '학습데이터_토픽'

    query1 = f"SELECT * FROM {tb_keyword}"
    query2 = f"SELECT * FROM {tb_summary}"
    query3 = f"SELECT * FROM {tb_senti}"
    query4 = f"SELECT * FROM {tb_topic}"

    cursor.execute(query1)
    result1 = cursor.fetchall()
    
    cursor.execute(query2)
    result2 = cursor.fetchall()

    cursor.execute(query3)
    result3 = cursor.fetchall()

    cursor.execute(query4)
    result4 = cursor.fetchall()

    # 기간키워드
    day = []
    keyword = []
    for i in result1:
        day.append(i[0])
        keyword.append(i[1])

    # 요약분석
    content = []
    summarize = []
    title = []
    index_ = []
    for i in result2:
        content.append(i[1])
        summarize.append(i[2])
        title.append(i[3])
        index_.append(i[0])

    # 감성분석
    nouns = []
    senti_score = []
    senti_result = []
    for i in result3:
        nouns.append(i[0])
        senti_score.append(i[1])
        senti_result.append(i[2])

    # 토픽모델링
    topic = []
    for i in result4:
        topic.append(i[1])

    return day, keyword, content, summarize, title, index_, nouns, senti_score, senti_result, topic


# 신뢰도분석은 따로
def get_keyword_reliability(conn):
    cursor = conn.cursor()

    tb_reliability = '학습데이터_신뢰도'

    query = f"SELECT * FROM {tb_reliability}"

    cursor.execute(query)
    result = cursor.fetchall()

    title = []
    content = []
    for i in result:
        title.append(i[1])
        content.append(i[2])

    conn.close()
    return title, content