import yt_dlp
def get_playlist_videos(playlist_url):
    ydl_opts = {
        'extract_flat': True,  # Sadece bilgileri çek, video indirme
        'skip_download': True,  # Videoları indirme
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(playlist_url, download=False)
        videos = info_dict.get('entries', [])

        video_list = []
        for video in videos:
            video_title = video.get('title')
            video_url = f"https://www.youtube.com/watch?v={video.get('id')}"
            video_list.append({'title': video_title, 'url': video_url})

        return video_list

