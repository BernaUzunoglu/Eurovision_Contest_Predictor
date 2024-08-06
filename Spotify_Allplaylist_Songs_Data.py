import csv
import pandas as pd
import requests
import base64

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
    if response.status_code == 200:
        access_token = response.json().get('access_token')
        return access_token
    else:
        print("Failed to get access token")
        return None

def get_track_details(track_id, token):
    if track_id is None or token is None:
        print("Invalid track_id or token.")
        return None
    base_url = 'https://api.spotify.com/v1/tracks/'
    headers = {'Authorization': 'Bearer ' + token}

    response = requests.get(base_url + track_id, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to get track details.")
        return None

def get_track_audio_features(track_id, token):
    if track_id is None or token is None:
        print("Invalid track_id or token.")
        return None
    base_url = 'https://api.spotify.com/v1/audio-features/'
    headers = {'Authorization': 'Bearer ' + token}
    response = requests.get(base_url + track_id, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to get audio features.")
        return None

def get_track_info(track_id):
    token = get_access_token(client_id, client_secret)

    if token is None:
        print("No valid access token.")
        return None

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
        print("Failed to get track info.")
        return None

def get_playlist_tracks(playlist_id, token, offset=0, limit=100):
    base_url = f'https://api.spotify.com/v1/playlists/{playlist_id}/tracks'
    headers = {'Authorization': 'Bearer ' + token}
    params = {'offset': offset, 'limit': limit}
    response = requests.get(base_url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print("Failed to get playlist tracks.")
        return None

def get_all_playlist_tracks(playlist_id):
    token = get_access_token(client_id, client_secret)

    if token is None:
        print("No valid access token.")
        return None

    all_tracks = []
    offset = 0
    limit = 100
    while True:
        playlist_tracks = get_playlist_tracks(playlist_id, token, offset, limit)
        if playlist_tracks is None or len(playlist_tracks['items']) == 0:
            break
        all_tracks.extend(playlist_tracks['items'])
        if len(playlist_tracks['items']) < limit:
            break
        offset += limit
    return all_tracks

def get_playlist_tracks_features(playlist_id):
    all_tracks = get_all_playlist_tracks(playlist_id)
    if all_tracks:
        token = get_access_token(client_id, client_secret)

        if token is None:
            print("No valid access token.")
            return None

        tracks_features = []
        for item in all_tracks:
            if item['track'] is None:
                continue
            track = item['track']
            track_id = track['id']
            track_name = track['name']
            track_artist = ', '.join([artist['name'] for artist in track['artists']])
            album_name = track['album']['name']
            release_date = track['album']['release_date']

            # Orijinal versiyon kontrolü
            version_keywords = ['live', 'remix', 'acoustic', 'version', 'edit', 'mix']
            if any(keyword in track_name.lower() or keyword in album_name.lower() for keyword in version_keywords):
                continue

            audio_features = get_track_audio_features(track_id, token)
            if audio_features:
                track_info = {
                    'name': track_name,
                    'artist': track_artist,
                    'album': album_name,
                    'release_date': release_date,
                    'audio_features': audio_features
                }
                tracks_features.append(track_info)

        return tracks_features
    else:
        return None

# Id si verilen şarkının bilgilerini getirme
track_id = '2jCcKU8BV8eUC1ju0hGNCv'
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
csv_file = 'song_data_all.csv'

if tracks_features:
    # Dosyayı aç ve başlıkları yaz
    with open(csv_file, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        headers = ['track_name', 'artist_name', 'album', 'release_date'] + list(tracks_features[0]['audio_features'].keys())
        writer.writerow(headers)

        for track in tracks_features:
            row = [track['name'], track['artist'], track['album'], track['release_date']] + list(track['audio_features'].values())
            writer.writerow(row)

    print(f"Data has been written to {csv_file}")

# CSV dosyasını okuma ve görüntüleme
df = pd.read_csv(csv_file)
print(df)
