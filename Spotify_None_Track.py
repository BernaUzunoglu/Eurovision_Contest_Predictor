import requests
from urllib.parse import urlencode
import pandas as pd
import csv


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Spotify API credentials
client_id = 'e8d269c91630491c8e568b0883a956d1'
client_secret = 'e6b74e0a1e4e4c7f80a12cc65a01dbcb'

# Function to get access token
def get_access_token(client_id, client_secret):
    url = 'https://accounts.spotify.com/api/token'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'grant_type': 'client_credentials'
    }
    response = requests.post(url, headers=headers, data=data, auth=(client_id, client_secret))
    return response.json()['access_token']

# Function to search for a track
def search_track(token, artist_name, track_name):
    query = f'artist:{artist_name} track:{track_name}'
    url = f'https://api.spotify.com/v1/search?{urlencode({"q": query, "type": "track", "limit": 1})}'
    headers = {
        'Authorization': f'Bearer {token}'
    }
    response = requests.get(url, headers=headers)
    results = response.json()
    if results['tracks']['items']:
        return results['tracks']['items'][0]['id']
    else:
        return None


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

# Get access token
token = get_access_token(client_id, client_secret)

df = pd.read_excel('datasets/Spotipy_None_Songs.xlsx')
df.head()

tracks_features = []
# for döngüsü ile 'artist_name' ve 'song_name' kolonlarındaki değerlere ulaşma
for index, row in df.iterrows():
    artist_name = row['artist_name']
    song_name = row['song_name']
    track_id = search_track(token, artist_name, song_name)
    print(track_id)
    if track_id:
        track_info = get_track_info(track_id)
        track_values = {
            'name': song_name,
            'artist': artist_name,
            'audio_features': track_info['audio_features']
        }
        tracks_features.append(track_info)
        print(track_info)
    else:
        continue
    print(f"Artist: {artist_name}, Song: {song_name}")

# CSV dosyasına yazma
csv_file = 'song_data_not.csv'

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


df_merge1 = pd.read_excel('datasets/Spotipy_None_Songs.xlsx')
df_merge2 = pd.read_csv('song_data_not.csv')

# Birleştirme işlemi (inner join)
merged_df = pd.merge(df_merge1, df_merge2, left_on=['artist_name'], right_on=['artist_name'], how='inner')


print(merged_df)



df1 = pd.DataFrame(tracks_features)
df1.head()
df1.shape

# Main script
artist_name = 'Aivaras'
track_name = 'Happy You'

# Search for track
track_id = search_track(token, artist_name, track_name)
if track_id:
    print(f'Track ID: {track_id}')
else:
    print('Track not found')
