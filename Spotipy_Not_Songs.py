import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import time

# Spotify API kimlik bilgilerinizi buraya ekleyin
client_id = 'e8d269c91630491c8e568b0883a956d1'
client_secret = 'e6b74e0a1e4e4c7f80a12cc65a01dbcb'

# Spotipy ile kimlik doğrulama
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Excel dosyasını okuma
df = pd.read_excel('sarkilar.xlsx')

# Şarkıcı ve şarkı isimlerinin bulunduğu sütunları kontrol edin
print(df.head())

def get_track_features(artist_name, track_name):
    query = f"artist:{artist_name} track:{track_name}"
    result = sp.search(q=query, type='track', limit=1)

    if result['tracks']['items']:
        track = result['tracks']['items'][0]
        track_id = track['id']
        track_features = sp.audio_features(track_id)[0]
        return {
            'artist': artist_name,
            'track': track_name,
            'id': track_id,
            'danceability': track_features['danceability'],
            'energy': track_features['energy'],
            'key': track_features['key'],
            'loudness': track_features['loudness'],
            'mode': track_features['mode'],
            'speechiness': track_features['speechiness'],
            'acousticness': track_features['acousticness'],
            'instrumentalness': track_features['instrumentalness'],
            'liveness': track_features['liveness'],
            'valence': track_features['valence'],
            'tempo': track_features['tempo'],
            'duration_ms': track_features['duration_ms'],
            'time_signature': track_features['time_signature']
        }
    else:
        return {
            'artist': artist_name,
            'track': track_name,
            'id': None,
            'danceability': None,
            'energy': None,
            'key': None,
            'loudness': None,
            'mode': None,
            'speechiness': None,
            'acousticness': None,
            'instrumentalness': None,
            'liveness': None,
            'valence': None,
            'tempo': None,
            'duration_ms': None,
            'time_signature': None
        }

features_list = []

for index, row in df.iterrows():
    artist_name = row['Artist']
    track_name = row['Track']
    features = get_track_features(artist_name, track_name)
    features_list.append(features)
    # API çağrılarını sınırlandırmak için kısa bir bekleme süresi ekleyin
    time.sleep(0.5)

features_df = pd.DataFrame(features_list)

# Sonuçları yeni bir Excel dosyasına kaydetme
features_df.to_excel('sarki_ozellikleri.xlsx', index=False)
