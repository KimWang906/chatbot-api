import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, Response, jsonify, request
import flask_cors

app = Flask(__name__)
flask_cors.CORS(app)

flask_cors.CORS(app, resources={r'*': {'origins': '*'}})

def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model


def get_dataset():
    df = pd.read_csv('servey_master finish.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df


res_generated = []
res_past = []

model = cached_model()
df = get_dataset()


@app.route("/chatbot", methods=["POST"])
def get_data():
    user_input = request.get_json()
    embedding = model.encode(user_input['buffer'])

    df['distance'] = df['embedding'].map(
        lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    res_past.append(user_input)
    if answer['distance'] > 0.5:
        res_generated.append(answer['답변'])
    else:
        res_generated.append(
            '적절한 답변이 없습니다. 정확한 답변을 듣고 싶으시다면 051-971-2153으로 연락주세요.')

    for i in range(len(res_past)):
        user_req = res_past[i]
        bot_res = res_generated[i]

    response = {
        "user_question": user_input['buffer'],
        "bot_answer": bot_res
    }

    return jsonify(response), 200


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
