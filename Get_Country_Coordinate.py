from geopy.geocoders import Nominatim

def get_country_coordinates(country_name):
    geolocator = Nominatim(user_agent="geoapiExercises")
    location = geolocator.geocode(country_name)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

coordinates_dict = {}
countries = ['Sweden', 'Ukraine', 'Italy', 'Russia', 'Greece', 'Norway', 'Denmark', 'Armenia', 'Estonia', 'Israel', 'Germany', 'France', 'Azerbaijan', 'Turkey']
for country in countries:
    latitude, longitude = get_country_coordinates(country)
    print(f"{country}: Latitude = {latitude}, Longitude = {longitude}")
    coordinates_dict[country] = [latitude, longitude]

data = []  # veri setinden gelen veriler
filtered_data = data[data['final_place'] < 10]
country_counts = filtered_data['country'].value_counts()
top_countries = country_counts[country_counts > 5]

# Ülke koordinatları (örnek olarak, gerçek koordinatlarınızı buraya ekleyin)
coordinates_dict = {'Sweden': [59.6749712, 14.5208584],
                    'Ukraine': [49.4871968, 31.2718321],
                    'Italy': [42.6384261, 12.674297],
                    'Russia': [64.6863136, 97.7453061],
                    'Greece': [38.9953683, 21.9877132],
                    'Norway': [64.5731537, 11.52803643954819],
                    'Denmark': [55.670249, 10.3333283],
                    'Armenia': [4.536307, -75.6723751],
                    'Estonia': [58.7523778, 25.3319078],
                    'Israel': [30.8124247, 34.8594762],
                    'Germany': [51.1638175, 10.4478313],
                    'France': [46.603354, 1.8883335],
                    'Azerbaijan': [40.3936294, 47.7872508],
                    'Turkey': [38.9597594, 34.9249653]}


# Top 10 ülke verilerini oluştur
top_countries_data = [{'country': country, 'lat': coordinates_dict[country][0], 'lon': coordinates_dict[country][1], 'count': top_countries.loc[top_countries['country'] == country, 'count'].values[0]} for country in top_countries['country']]
