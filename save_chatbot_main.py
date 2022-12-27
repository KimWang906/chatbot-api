import json
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

now = datetime.now()
st.set_page_config(layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css("style.css")


@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model


@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('servey_master finish.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df


model = cached_model()
df = get_dataset()

st.header('부산소프트웨어마이스터고 챗봇')
st.subheader("안녕하세요 소마고 챗봇입니다.")

with st.form('form', clear_on_submit=True):
    user_input = st.text_input('사용자 : ', '')
    submitted = st.form_submit_button('전송')

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(
        lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    if answer['distance'] > 0.2:
        st.session_state.generated.append(answer['답변'])
    else:
        st.session_state.generated.append(
            '적절한 답변이 없습니다. 정확한 답변을 듣고 싶으시다면 051-971-2153으로 연락주세요.')

for i in range(len(st.session_state['past'])):
    # message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    # if len(st.session_state['generated']) > i:
    st.markdown(
        """
                <div class="msg right-msg>
                    <div class="msg-img></div>
                    <div class="msg-bubble">
                        <div class="msg-info">
                            <div class="uname">지나가던 유저1</div>
                        </div>
                        <p>{0}</p>
                    </div>
                </div>

                <div class="msg left-msg">
                    <div class="msg-img"></div>
                    <div class="msg-img></div>
                    <div class="msg-bubble">
                        <div class="bot-name">
                            <div class="bname">소마</div>
                        </div>
                        <p>{1}</p>
                    </div>
                </div>
            """.format(st.session_state['past'][i], st.session_state['generated'][i]), unsafe_allow_html=True)


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://d3kxs6kpbh59hp.cloudfront.net/community/COMMUNITY/1552aee2f1704b62b7e7628cff0cbc2c/9106f352eb2545a2b2c9f17c646160dc_1620190538.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 

st.sidebar.title("BSSM")
st.sidebar.info(
    """
    [HomePage](https://school.busanedu.net/bssm-h/main.do)  |
    [Instagram](https://www.instgram.com/bssm.hs/)  |
    [Facebook](https://www.facebook.com/BusanSoftwareMeisterHighschool)
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    T : 051-971-2153
    """
)

tab1, tab2, tab3 = st.tabs(["학교소개", "입학안내", "문의"])

with tab1:
    st.header("저희 소마고를 소개합니다")
    st.text("학생 수 : 125명(남 55%, 여 45%)")
    st.text("교원수 : 33명")
    st.text("부산광역시 강서구 가락대로 1393")

with tab2:
    st.header("입학 안내")

with tab3:
    st.header("챗봇에게 무엇이든 물어보세요!")
