import pandas as pd
import requests
import base64
import streamlit as st
from yt_dlp import YoutubeDL


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

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


st.title('Spotify Şarkı Seçici')

cols1, cols2 = st.columns([2, 1])

with cols1:
    # Kullanıcıdan şarkı adı girmesini isteyin
    query = st.text_input('Şarkı Adı Girin:')
    # query = 'Çakkıdı'
    if query:
        access_token = get_access_token(client_id, client_secret)
        results = search_track(query, access_token)

        if results.get('tracks'):
            track_options = [f"{track['name']} - {track['artists'][0]['name']}" for track in results['tracks']['items']]

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
