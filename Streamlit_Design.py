import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
from streamlit_carousel import carousel
import get_youtube_playlist_songs as youtube
import requests
import base64
from yt_dlp import YoutubeDL
import feature_engineering as fe
import advanced_functional_eda as eda
import Eurovision_Contest_Predictor as predictor
import matplotlib.pyplot as plt
import seaborn as sns
import pydeck as pdk
import altair as alt
import random
import joblib


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

st.set_page_config(layout='wide', page_title='Eurovision Contest Predictor', page_icon='🎤')

@st.cache_data
def get_data():
    meta = pd.read_excel('datasets/Contest_Songs_Data.xlsx')
    world_happinies = pd.read_excel('datasets/worldhappinessdata.xlsx')
    meta.columns = [col.upper() for col in meta.columns]
    return meta, world_happinies

def fit_model():
    model , rmse , data = predictor.contest_predictor()
    return model , rmse, data

# model, rmse , data = fit_model()

# Modeli yükleme
loaded_model = joblib.load('random_forest_eurovision_predictor.joblib')

meta, world_happinies = get_data()
meta1 = meta[['PARTICIPATION_RATE','DANCEABILITY', 'ENERGY', 'LOUDNESS', 'SPEECHINESS', 'ACOUSTICNESS', 'INSTRUMENTALNESS', 'LIVENESS', 'VALENCE', 'TEMPO', 'GDP_PER_CAPITA', 'SOCIAL_SUPPORT', 'NEGATIVE_AFFECT', 'PERCEPTIONS_OF_CORRUPTION', 'GENEROSITY', 'HEALTHY_LIFE_EXPECTANCY_AT_BIRTH', 'FREEDOM_TO_MAKE_LIFE_CHOICES']]

# video_html = """
#        <style>
#
#        #myVideo {
#          position: fixed;
#          right: 0;
#          bottom: 0;
#          min-width: 100%;
#          min-height: 100%;
#          object-fit: fit;
#
#        }
#
#        .content {
#          position: fixed;
#          bottom: 0;
#          background: rgba(0, 0, 0, 0.5);
#          color: #f1f1f1;
#          width: 50%;
#          height: 50%;
#          padding: 20px;
#        }
#
#        </style>
#        <video autoplay muted loop id="myVideo">
#          <source src="https://cdn.pixabay.com/video/2021/12/08/100000-654636859_large.mp4">
#          Your browser does not support HTML5 video.
#        </video>
#         """

# st.markdown(video_html, unsafe_allow_html=True)

# HTML ve CSS ile ses oynatıcıyı özelleştirme
st.markdown(
     f"""
        <style>
         audio {{
            max-height: 15px;
            margin-top: 5px;
            max-width:200px;
        }}


        audio::-webkit-media-controls-panel,
        audio::-webkit-media-controls-enclosure {{
            background-color: #a825ac;
            border-radius: 25px;
            margin-left: 10px;
            margin-right: 10px;
        }}

        div[data-testid="stVerticalBlock"] {{
            gap: 0px;
        }}
        </style>
        """,
    unsafe_allow_html=True
)
# st.audio("eurodatasong.mp3", loop=True)


# CSS ile carousel arka plan rengini ayarlama - css leri ezemedim sonra bir daha bakacağım.
st.markdown(
    """
    <style>
    .carousel-control-prev {
      opacity: 0.5;
      background-color: #a825ac;
    }
    
    .carousel-item {
        opacity: 0.5;
        background-color: #a825ac;  /* Arka plan rengini transparan ayarlama */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# CSS yazı stilini tanımla
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Calligraffitti&family=Kalnia+Glaze:wght@100..700&family=Playwrite+DK+Loopet:wght@100..400&display=swap');

    .calligraffitti-text {
        font-family: 'Calligraffitti', cursive;
        font-size: 40px;
        color: #f0f0f0; /* Beyaz bir renk */
        font-weight: bold;
    }
    .lato-text {
        font-family: 'Calibri ';
        font-size: 30px;
        color: #f0f0f0; /* Beyaz bir renk */
    }

    .container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Metinler
# st.markdown('''
#     <div class="container">
#         <div class="calligraffitti-text">EURO🤍ISION</div>
#         <div class="lato-text">Contest Predictor</div>
#     </div>
# ''', unsafe_allow_html=True)
col1, col2, col3 =st.columns([1, 2, 1])
with col1:
    st.audio("eurodatasong.mp3", loop=True)
with col2:
    st.markdown('''
        <div class="container">
            <div class="calligraffitti-text">EURO🤍ISION</div>
            <div class="lato-text">Contest Predictor</div>
        </div>
    ''', unsafe_allow_html=True)
with col3:
    st.image("Miuul_crop.jpg", width=250)

video_url = "https://cdn.pixabay.com/video/2017/03/19/8437-209165711_large.mp4"
# Arka plan videosunu HTML ve CSS ile ekleyin
st.markdown( f""" <style> .background {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; overflow: hidden; z-index: -1; background: rgba(0, 0, 0, 0.5); /* Opsiyonel: koyu bir arka plan rengi */ }} .background video {{ position: absolute; top: 50%; left: 50%; width: 100%; height: 100%; object-fit: cover; transform: translate(-50%, -50%); }} </style> <div class="background"> <video autoplay muted loop> <source src="{video_url}" type="video/mp4"> Your browser does not support the video tag. </video> </div> """, unsafe_allow_html=True )
# st.title('🎵:rainbow[Eurovision Contest Predictor]🎶', )
# Playlist URL'sini buraya ekleyin
playlist_url = 'https://www.youtube.com/playlist?list=PLhTunAmPoKKHUNtOC7fh-qdM1jHPGq0w4'

# Playlist'teki videoları çek
videos = youtube.get_playlist_videos(playlist_url)

home_tab, graph_tab, recommendation_tab = st.tabs(["Home", "Statistics", "Predictor "])

# home tab
test_items = [
    dict(
        title="",
        text="Switzerland's Nemo wins the Eurovision Song Contest 2024",
        img="https://www.srgssr.ch/fileadmin/_processed_/f/1/csm__c_Corinne_Cumming_EBU_6f065a028e.jpg",
        link="https://eurovision.tv/story/switzerland-wins-eurovision-2024"
    ),
    dict(
        title="",
        text="maNga - We Could Be The Same (Turkey) Live 2010 Eurovision Song Contest",
        img="https://i.ytimg.com/vi/HB_GnnhNz-8/maxresdefault.jpg",
        link="https://eurovision.tv/video/manga-we-could-be-the-same-turkey-live-2010-eurovision-song-contest"
    ),
    dict(
        title="",
        text="Eurovision 2024: All 37 songs",
        img="https://eurovision.tv/sites/default/files/styles/banner/public/media/image/2024-03/all37songs24.png?itok=I8_zBqcK",
        link="https://eurovision.tv/story/eurovision-2024-all-37-songs"
    ),
]

# col1, col2, col3 = home_tab.columns([1, 1, 1])

with home_tab:
    carousel(items=test_items, width=0.8, container_height=500, fade=True)
    # Üçlü kolonlar
    # Her bir kolona video başlığı ve linki ekleme
    # Her 3 video için bir satırda 3 kolon oluşturma
    # for i in range(0, len(videos), 3):
    for i in range(0, 30, 3):
        cols = home_tab.columns(3)  # 3 kolon oluştur
        for j, col in enumerate(cols):
            if i + j < len(videos):  # Listenin sonunu aşmamak için kontrol
                video = videos[i + j]
                with col:
                    st.video(video['url'])
                    st.write(video['title'])
# Grafik Çizim Fonksiyonları
with graph_tab:
    st.subheader("Bölge Başına Yarışmaya Katılım Sayısı")
    st.divider()
    st.markdown(""" - Güney Avrupa bölgesi, en yüksek katılım sayısına sahip, bu da bu bölgenin Eurovision yarışmasına olan ilgisinin yüksek olduğunu gösteriyor.""")
    st.write("""- Kuzey Avrupa ve Batı Avrupa da yüksek katılım sayıları ile öne çıkıyor.""")
    st.write("""- Avustralya, Kuzey Asya ve Orta Doğu en düşük katılım sayılarına sahip bölgeler, bu da bu bölgelerin Eurovision'a daha az ilgi gösterdiğini veya daha az temsil edildiğini gösteriyor.""")


    # Her bölgenin kaç defa katıldığını saymak için gruplama yap
    region_counts = meta.groupby('AREA').size().reset_index(name='Count')
    # Scatter plot oluştur
    fig = px.scatter(
        region_counts,
        x='AREA',
        y='Count',
        text='Count',
        title='Number of Competition Entries per Region',
        labels={'AREA': 'Region', 'Count': 'Count'},
        color='AREA',  # Doğru sütun adı
        size='Count',
        size_max=100
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    st.divider()
    st.subheader("En Çok Top 10 Listesine Kalan Ülkeler")
    st.divider()
    st.write(""" - İsveç: Listenin başında yer alıyor ve Eurovision'da oldukça başarılı bir geçmişe sahip. İsveç, pop müzik sahnesinde de oldukça güçlü bir ülke olarak biliniyor ve bu başarıları, ülkedeki müzik endüstrisinin gücünü ve Eurovision'a verdikleri önemi yansıtıyor.""")
    st.write("""- Ukrayna: İkinci sırada yer alarak önemli bir başarıya imza atmış. Ukrayna, özellikle son yıllarda güçlü performanslar sergileyerek ilk 10'a girmeyi başarmış.""")
    st.write("""- Bu veriler, Eurovision'da başarılı olan ülkelerin çoğunlukla Avrupa'nın müzik açısından zengin ve güçlü ülkeleri olduğunu gösteriyor. İsveç, Ukrayna ve İtalya gibi ülkeler, sürekli olarak yüksek kaliteli müzik ve sahne performansları sunarak yarışmada başarılı oluyor.""")
    st.write(" ")  # Boş satır
    filtered_data = meta[meta['FINAL_PLACE'] < 10]
    country_counts = filtered_data['COUNTRY'].value_counts()
    top_countries = country_counts[country_counts > 5]

    top_countries_data = [{'country': 'Sweden', 'lat': 59.6749712, 'lon': 14.5208584, 'count': 17},
                          {'country': 'Ukraine', 'lat': 49.4871968, 'lon': 31.2718321, 'count': 12},
                          {'country': 'Italy', 'lat': 42.6384261, 'lon': 12.674297, 'count': 11},
                          {'country': 'Russia', 'lat': 64.6863136, 'lon': 97.7453061, 'count': 10},
                          {'country': 'Greece', 'lat': 38.9953683, 'lon': 21.9877132, 'count': 10},
                          {'country': 'Norway', 'lat': 64.5731537, 'lon': 11.52803643954819, 'count': 9},
                          {'country': 'Denmark', 'lat': 55.670249, 'lon': 10.3333283, 'count': 8},
                          {'country': 'Armenia', 'lat': 40.0691, 'lon': 45.0382, 'count': 7},
                          {'country': 'Estonia', 'lat': 58.7523778, 'lon': 25.3319078, 'count': 7},
                          {'country': 'Israel', 'lat': 30.8124247, 'lon': 34.8594762, 'count': 6},
                          {'country': 'Germany', 'lat': 51.1638175, 'lon': 10.4478313, 'count': 6},
                          {'country': 'France', 'lat': 46.603354, 'lon': 1.8883335, 'count': 6},
                          {'country': 'Azerbaijan', 'lat': 40.3936294, 'lon': 47.7872508, 'count': 6},
                          {'country': 'Turkey', 'lat': 38.9597594, 'lon': 34.9249653, 'count': 6}]

    # Koordinatları ve katılım sayılarını içeren verileri oluştur
    top_countries_df = pd.DataFrame(top_countries_data)

    # Streamlit ve Pydeck ile harita oluşturma
    st.pydeck_chart(
        pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(
                latitude=38.95,
                longitude=34.92,  # Türkiye'nin koordinatlarını merkeze alalım
                zoom=3,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    "ColumnLayer",
                    data=top_countries_df,
                    get_position="[lon, lat]",
                    get_elevation="count",
                    elevation_scale=50000,
                    radius=50000,
                    get_fill_color="[255, 140, 0]",
                    pickable=True,
                    extruded=True,
                ),
                pdk.Layer(
                    "TextLayer",
                    data=top_countries_df,
                    get_position="[lon, lat]",
                    get_text="count",
                    get_color=[0, 0, 0, 200],
                    get_size=16,
                    get_alignment_baseline="'bottom'",
                ),
            ],
            tooltip={
                'html': '<b>Country:</b> {country} <br> <b>Count:</b> {count}',
                'style': {
                    'backgroundColor': 'steelblue',
                    'color': 'white'
                }
            }
        )
    )
    st.divider()
    st.subheader("Ülkelerin Birincilik Sayıları")
    st.divider()
    st.write(""" - İsveç : Grafikte en yüksek değere sahip olan İsveç, Eurovision'da en fazla birinci olan ülke olarak dikkat çekiyor.""")
    st.write("""- Ukrayna: İsveç'ten sonra en yüksek değere sahip olan Ukrayna, Eurovision'da birçok kez birinci olmuştur. Ukrayna'nın Eurovision performansları, özgün müzik tarzları ve güçlü sahne performanslarıyla tanınır.""")
    st.write("""- Grafikte Danimarka'nın iki kez birincilik derecesi aldığı ve Almanya, İsviçre ve Türkiye  gibi  diğer ülkeler de grafikte birer kez birincilik dereceleri aldığı görülmektedir.""")
    st.write(" ")  # Boş satır

    filtered_finalist = meta[meta['FINAL_PLACE'] == 1]
    country_finalist_counts = filtered_finalist['COUNTRY'].value_counts()
    country_finalist_counts = pd.DataFrame(country_finalist_counts)
    type(country_finalist_counts)
    country_finalist_counts = country_finalist_counts.reset_index()
    # country_finalist_counts.columns = ['Country','Count']

    def random_hex_color():
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    # Renk sütununu rastgele renklerle doldurun
    country_finalist_counts['color'] = [random_hex_color() for _ in range(len(country_finalist_counts))]

    st.bar_chart(country_finalist_counts, x="COUNTRY", y="count", color="color",width=500,height=450)

# Predector
with recommendation_tab:
    # Client ID ve Client Secret bilgilerinizi girin
    client_id = 'e8d269c91630491c8e568b0883a956d1'
    client_secret = 'e6b74e0a1e4e4c7f80a12cc65a01dbcb'


    def get_access_token(client_id, client_secret):
        auth_url = 'https://accounts.spotify.com/api/token'
        auth_header = base64.b64encode((client_id + ':' + client_secret).encode('ascii'))
        auth_data = {'grant_type': 'client_credentials'}

        response = requests.post(auth_url, headers={'Authorization': 'Basic ' + auth_header.decode('ascii')},
                                 data=auth_data)
        access_token = response.json()['access_token']
        return access_token


    get_access_token(client_id, client_secret)


    def search_track(query, access_token):
        headers = {
            "Authorization": f"Bearer {access_token}"
        }
        params = {
            "q": query,
            "type": "track",
            "limit": 10
        }
        response = requests.get("https://api.spotify.com/v1/search", headers=headers, params=params)
        return response.json()


    def get_track_audio_features(track_id, token):
        base_url = 'https://api.spotify.com/v1/audio-features/'
        headers = {
            'Authorization': 'Bearer ' + token
        }
        response = requests.get(base_url + track_id, headers=headers)
        return response.json()


    def get_youtube_url(query):
        ydl_opts = {
            'format': 'best',
            'noplaylist': True,
            'quiet': True,
            'extract_flat': True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(f"ytsearch:{query}", download=False)
            video_info = result['entries'][0]
            video_url = f"https://www.youtube.com/watch?v={video_info['id']}"
            return video_url

    print("Deneme")
    # query = 'ceza fark var'
    cols1, cols2 = st.columns(2)
    audio_features = []
    with cols1:
        # Kullanıcıdan şarkı adı girmesini isteyin
        query = st.text_input('Şarkı Adı Girin:')
        # query = 'yerli plaka ceza'
        if query:
            access_token = get_access_token(client_id, client_secret)
            results = search_track(query, access_token)

            if results.get('tracks'):
                track_options = [f"{track['name']} - {track['artists'][0]['name']}" for track in
                                 results['tracks']['items']]

                selected_track = st.selectbox('Şarkı Seçin:', track_options)

                if selected_track:
                    track_index = track_options.index(selected_track)
                    selected_track_info = results['tracks']['items'][track_index]

                    video_url = get_youtube_url(selected_track)
                    audio_features = get_track_audio_features(selected_track_info['id'], access_token)
                    st.write(f"Sanatçı: {selected_track_info['artists'][0]['name']}")
                    st.write(f"Albüm: {selected_track_info['album']['name']}")
                    st.video(video_url)
            else:
                st.write('Şarkı bulunamadı.')

    with cols2:
        # Eurovision'a katılan ülkelerin listesi
        countries = ['Ukraine', 'Turkey', 'Finland', 'Albania', 'North Macedonia',
       'Serbia', 'Slovenia', 'Slovakia', 'Russia', 'Montenegro',
       'Serbia & Montenegro', 'Moldova', 'Romania', 'Belgium', 'Portugal',
       'Spain', 'Poland', 'Norway', 'Netherlands', 'Latvia', 'Lithuania',
       'Estonia', 'Italy', 'Switzerland', 'Iceland', 'Azerbaijan',
       'Hungary', 'Bosnia & Herzegovina', 'Bosnia and Herzegovina',
       'Croatia', 'Israel', 'Greece', 'Armenia', 'Cyprus', 'France',
       'Luxembourg', 'Malta', 'Monaco', 'Georgia', 'Australia', 'Austria',
       'Belarus', 'Bulgaria', 'Czech Republic', 'Denmark', 'Germany',
       'Ireland', 'Sweden', 'United Kingdom']

        # Ülke seçimi
        selected_country = st.selectbox("Ülke Seçin:", countries)
        # selected_country = 'Turkey'
        if query:
            print(audio_features)
            audio_features_df = pd.DataFrame([audio_features])
            audio_features_predict = audio_features_df.drop(
                ["analysis_url", 'type', 'id', "uri", "track_href", "key", "mode", "time_signature","duration_ms"],
                axis=1).reset_index()
            world_happinies_predict1 = world_happinies[world_happinies['COUNTRY'] == selected_country]
            # Türkiye'yi ilk gördüğümüz satırı bulma
            first_country_row = meta[meta['COUNTRY'] == selected_country].iloc[0]
            # 'oran' sütununun değerini alma
            oran_value = first_country_row['PARTICIPATION_RATE']
            world_happinies_predict = world_happinies_predict1.drop(["COUNTRY"], axis=1).reset_index()
            # world_happinies_predict['PARTICIPATION_RATE'] = oran_value
            predict_data = pd.concat([audio_features_predict, world_happinies_predict], axis=1).drop(["index"], axis=1)
            predict_data.insert(0, 'PARTICIPATION_RATE', oran_value)
            predict_data.columns = [col.upper() for col in predict_data.columns]
            predict_data = predict_data.drop(['POSITIVE_AFFECT'], axis=1)

            standatrlasancols = ['DANCEABILITY', 'ENERGY',
                                 'LOUDNESS', 'SPEECHINESS', 'ACOUSTICNESS', 'INSTRUMENTALNESS',
                                 'LIVENESS', 'VALENCE', 'TEMPO',
                                 'GDP_PER_CAPITA', 'HEALTHY_LIFE_EXPECTANCY_AT_BIRTH']
            sil = fe.num_cols_standardization(meta1, standatrlasancols, "ss")
            meta1.loc[len(meta)] = predict_data.iloc[0].tolist()
            for col in meta1.columns:
                fe.replace_with_thresholds(meta1, col)

            standatrlasancols = ['DANCEABILITY', 'ENERGY',
                                 'LOUDNESS', 'SPEECHINESS', 'ACOUSTICNESS', 'INSTRUMENTALNESS',
                                 'LIVENESS', 'VALENCE', 'TEMPO',
                                 'GDP_PER_CAPITA', 'HEALTHY_LIFE_EXPECTANCY_AT_BIRTH']


            predict_data_finish = fe.num_cols_standardization(meta1, standatrlasancols, "ss")
            # Tahmin yapma
            predictions = loaded_model.predict(predict_data_finish.tail(1))
            # Tahmin işlemi bittiğinde predict_data değişkenini silmek için:
            predict_data = pd.DataFrame()
            # predicted_class = model[0].predict(predict_data.tail(1))
            # Mesajı oluştur ve ekranda göster
            st.write(f"Tahmini sıralama değeri : {int(predictions)} 'dır.")



