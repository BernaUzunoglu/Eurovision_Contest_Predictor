import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Spotify ile kimlik doğrulama
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id='senin_client_id',
    client_secret='senin_client_secret',
    redirect_uri='senin_redirect_uri',
    scope='playlist-read-private'
))


def get_playlist_tracks_features(playlist_id):
    # Playlist'teki şarkıları al
    playlist_tracks = sp.playlist_tracks(playlist_id)

    # Şarkı özelliklerini tutmak için boş bir liste oluştur
    track_features_list = []

    # Şarkıları listele ve her bir şarkının özelliklerini al
    for item in playlist_tracks['items']:
        track = item['track']
        # Şarkı özelliklerini al
        features = sp.audio_features(track['id'])[0]
        # Şarkı ve özellikleri birleştir
        track_info = {
            'name': track['name'],
            'artist': track['artists'][0]['name'],
            'album': track['album']['name'],
            'release_date': track['album']['release_date'],
            'duration_ms': track['duration_ms'],
            'popularity': track['popularity'],
            'danceability': features['danceability'],
            'energy': features['energy'],
            'key': features['key'],
            'loudness': features['loudness'],
            'mode': features['mode'],
            'speechiness': features['speechiness'],
            'acousticness': features['acousticness'],
            'instrumentalness': features['instrumentalness'],
            'liveness': features['liveness'],
            'valence': features['valence'],
            'tempo': features['tempo'],
            'time_signature': features['time_signature']
        }
        # Listeye ekle
        track_features_list.append(track_info)

    return track_features_list


# Örnek kullanım
playlist_id = '4nZ7HUx52tSPf2Yw9TbSqy'
track_features = get_playlist_tracks_features(playlist_id)
for track in track_features:
    print(track)
