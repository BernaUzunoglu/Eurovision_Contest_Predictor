import csv
import pandas as pd
import requests
import base64
import requests

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
def get_track_details(track_id, token):
    base_url = 'https://api.spotify.com/v1/tracks/'
    headers = {'Authorization': 'Bearer ' + token}

    response = requests.get(base_url + track_id, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None
def get_track_audio_features(track_id, token):
    base_url = 'https://api.spotify.com/v1/audio-features/'
    headers = {
        'Authorization': 'Bearer ' + token
    }
    response = requests.get(base_url + track_id, headers=headers)
    return response.json()

def get_track_info(track_id):
    token = get_access_token(client_id, client_secret)

    track_details = get_track_details(track_id, token)
    track_features = get_track_audio_features(track_id, token)

    if track_details and track_features:
        track_info = {
            'name': track_details['name'],
            'artist': ', '.join([artist['name'] for artist in track_details['artists']]),
            'album': track_details['album']['name'],
            'release_date': track_details['album']['release_date'],
            'duration_ms': track_details['duration_ms'],
            'popularity': track_details['popularity'],
            'audio_features': track_features
        }
        return track_info
    else:
        return None

def get_playlist_tracks(playlist_id, token):
    base_url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
    headers = {'Authorization': 'Bearer ' + token}
    response = requests.get(base_url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None


def get_playlist_tracks_features(playlist_id):
    token = get_access_token(client_id, client_secret)

    playlist_tracks = get_playlist_tracks(playlist_id, token)
    if playlist_tracks:
        tracks_features = []
        tracks_features_None = []
        for item in playlist_tracks['items']:
            print(item)
            if item['track'] == None:
                continue
            track = item['track']
            print(track)
            track_id = track['id']
            track_name = track['name']
            track_artist = ', '.join([artist['name'] for artist in track['artists']])

            audio_features = get_track_audio_features(track_id, token)
            if audio_features:
                track_info = {
                    'name': track_name,
                    'artist': track_artist,
                    'audio_features': audio_features
                }
                tracks_features.append(track_info)

        return tracks_features
    else:
        return None


# Id si verilen şarkının bilgilerini getirme
# track_id = '22E0QImA8MZuxadgAHIBet'
# track_id = '2BMOV8UZtc6Ex04SEXW54j'
# track_id = '7MJdPvhjeRwNuJQuyh3euM?si'
track_id = '3aUtmu4VTqAFkk4sFQ5kNa'
track_info = get_track_info(track_id)
if track_info:
    print("Track Name:", track_info['name'])
    print("Artist(s):", track_info['artist'])
    print("Album:", track_info['album'])
    print("Release Date:", track_info['release_date'])
    print("Duration (ms):", track_info['duration_ms'])
    print("Popularity:", track_info['popularity'])
    print("Audio Features:", track_info['audio_features'])
else:
    print("Track not found or unable to fetch details.")


# Çalma listesindeki şarkıların verilerini getirme
# playlist_id = '4nZ7HUx52tSPf2Yw9TbSqy'
playlist_id = '2UXTknLnk9j2MgNxW9mBpG'
tracks_features = get_playlist_tracks_features(playlist_id)

if tracks_features:
    for track in tracks_features:
        print("Track Name:", track['name'])
        print("Artist(s):", track['artist'])
        print("Audio Features:", track['audio_features'])
        print(track)
        print("\n")
else:
    print("Unable to fetch playlist tracks or their features.")


# CSV dosyasına yazma
csv_file = 'song_data.csv'

if tracks_features:
    # Dosyayı aç ve başlıkları yaz
    with open(csv_file, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        headers = ['track_name', 'artist_name'] + list(tracks_features[0]['audio_features'].keys())
        writer.writerow(headers)

        for track in tracks_features:
            row = [track['name'], track['artist']] + list(track['audio_features'].values())
            writer.writerow(row)

    print(f"Data has been written to {csv_file}")

df = pd.read_csv("track_info.csv")


# Excel dosyasını okuma
df = pd.read_excel('sarkilar.xlsx')

# Şarkıcı ve şarkı isimlerinin bulunduğu sütunları kontrol edin
print(df.head())

def get_track_features(token, artist_name, track_name):
    search_url = 'https://api.spotify.com/v1/search'
    headers = {
        'Authorization': f'Bearer {token}'
    }
    params = {
        'q': f'artist:{artist_name} track:{track_name}',
        'type': 'track',
        'limit': 1
    }
    search_response = requests.get(search_url, headers=headers, params=params)
    search_results = search_response.json()

    if search_results['tracks']['items']:
        track = search_results['tracks']['items'][0]
        track_id = track['id']
        features_url = f'https://api.spotify.com/v1/audio-features/{track_id}'
        features_response = requests.get(features_url, headers=headers)
        track_features = features_response.json()


features_list = []

for index, row in df.iterrows():
    artist_name = row['Artist']
    track_name = row['Track']
    features = get_track_features(token, artist_name, track_name)
    features_list.append(features)
    # API çağrılarını sınırlandırmak için kısa bir bekleme süresi ekleyin
    time.sleep(0.5)

features_df = pd.DataFrame(features_list)

# Sonuçları yeni bir Excel dosyasına kaydetme
features_df.to_excel('sarki_ozellikleri.xlsx', index=False)