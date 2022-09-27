import os

from bs4 import BeautifulSoup
import requests
import pandas as pd

theMagicians_episodes = []
cast_names = []
ep_cast = []
for season in range(1, 3):
    response = requests.get('https://www.imdb.com/title/tt4254242/episodes?season=' + str(season))

    soup = BeautifulSoup(response.text, "html.parser")

    episode_containers = soup.find_all('div', class_='info')

    for e in episode_containers:
        season = season
        title = e.a['title']
        ep_number = e.meta['content']
        airdate = e.find('div', class_='airdate').text.strip()

        rating = e.find('span', class_='ipl-rating-star__rating').text

        votes = e.find('span', class_='ipl-rating-star__total-votes').text
        for r in ((',', ''), ('(', ''), (')', '')):
            votes = votes.replace(*r)

        description = e.find('div', class_='item_description').text.strip()

        episode_containers = soup.find_all('div', class_='image')

        for epi in episode_containers:
            project_href = [i['href'] for i in epi.find_all('a', href=True)]

            for p in project_href:
                for ep in range(1, 14):
                    res = requests.get('https://www.imdb.com' + str(p) + '?ref_=ttep_ep' + str(ep))
                    soup = BeautifulSoup(res.text, "html.parser")

                    ep_container = soup.find_all('div', class_='sc-36c36dd0-8 fSYMLK')

                    for i in ep_container:
                        cast_names.append([cast.get_text() for cast in i.find_all(href=True)][0])
                    ep_cast.append(cast_names.copy())
                    cast_names.clear()
        episode_data = [season, title, ep_number, airdate, rating, votes, description, ep_cast[int(ep_number)-1]]

        theMagicians_episodes.append(episode_data)

theMagicians_episodes = pd.DataFrame(theMagicians_episodes,
                                     columns=['season', 'title', 'episode_number', 'airdate', 'rating', 'total_votes',
                                              'desc', 'cast'])

# formatting
theMagicians_episodes['airdate'] = pd.to_datetime(theMagicians_episodes.airdate)

theMagicians_episodes.head()

if os.path.exists("theMagicians_episodes.csv"):
    os.remove("theMagicians_episodes.csv")
    print("The file has been deleted successfully")
    theMagicians_episodes.to_csv('theMagicians_episodes.csv', index=False)
else:
    theMagicians_episodes.to_csv('theMagicians_episodes.csv', index=False)
