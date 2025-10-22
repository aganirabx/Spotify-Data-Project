'''
Programmer: Xavier Barinaga  

This is a utility file containg functions for my Spotify Data JupyterNotebook
'''

import pandas as pd
from dotenv import load_dotenv
import os
import requests
from requests.utils import quote
import json
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree


#def 



def filter_non_playlist_songs(json_df_1):
    print("Number of Songs Before:", len(json_df_1))


    result_json = json.load(open("Playlist1.json", "r"))
    playlists_json = result_json["playlists"]

    playlist_artists = []

    for single_playlist in playlists_json:
        for item in single_playlist["items"]:
            if item.get("track") and item["track"].get("artistName"):
                artist_name = item["track"]["artistName"]
                if artist_name not in playlist_artists:
                    playlist_artists.append(artist_name)


    json_df_1 = json_df_1[json_df_1["artistName"].isin(playlist_artists)]

    print("Number of songs after:", (len(json_df_1)))

    return json_df_1

def get_nonrepeat_tracks(json_df_1):
    json_df_artist_names = json_df_1["artistName"]
    json_df_track_names = json_df_1["trackName"]
    nonrepeat_tracks = []
    nonrepeat_track_artists = []
    for i in range(len(json_df_track_names)):
        if json_df_track_names.iloc[i] not in nonrepeat_tracks:
            nonrepeat_tracks.append(json_df_track_names.iloc[i])
            nonrepeat_track_artists.append(json_df_artist_names.iloc[i])
    return nonrepeat_tracks, nonrepeat_track_artists

def get_nonrepeat_id(json_df_1):
    json_df_id = json_df_1["trackId"]
    nonrepeat_id = []
    for i in range(len(json_df_id)):
        if json_df_id.iloc[i] not in nonrepeat_id:
            nonrepeat_id.append(json_df_id.iloc[i])
    return nonrepeat_id

def get_song_uri(json_df_1):
    nonrepeat_tracks, nonrepeat_track_artists = get_nonrepeat_tracks(json_df_1)

    load_dotenv("spotify.env")
    token = os.getenv("SPOTIFY_TOKEN")
    headers = {
        "Authorization": f"Bearer {token}"
    }

    all_track_uris = {}

    count = 0

    for artist, track in zip(nonrepeat_track_artists, nonrepeat_tracks):
        query = f"track:{track} artist:{artist}"
        better_query = quote(query)
        url = f"https://api.spotify.com/v1/search?q={better_query}&type=track&limit=1"
        response = requests.get(url, headers=headers)

        

        data = response.json()
        # with open("test.json", "w") as file:
        #     json.dump(data, file, indent=4)
        items = data.get("tracks", {}).get("items", [])
        if items:
            track_info = items[0]
            track_uri = track_info.get("uri")
            song_name = track_info.get("name")
            all_track_uris[song_name] = track_uri

        print(count)
        count += 1
    
        
    with open("track_search_responses1.json", "w") as file:
        json.dump(all_track_uris, file, indent=4)

    

def get_artist_info(json_df_1):
    nonrepeat_tracks, nonrepeat_track_artists = get_nonrepeat_tracks(json_df_1)

    load_dotenv("spotify.env")
    token = os.getenv("SPOTIFY_TOKEN")
    headers = {
        "Authorization": f"Bearer {token}"
    }

    all_artist_popularity = {}
    all_artist_genre = {}
    count = 0

    for artist in nonrepeat_track_artists:
        query = f"artist:{artist}"
        better_query = quote(query)
        url = f"https://api.spotify.com/v1/search?q={better_query}&type=artist&limit=1"
        response = requests.get(url, headers=headers)

        data = response.json()

        items = data.get("artists", {}).get("items", [])

        artist_info = items[0]
        artist_name = artist_info.get("name")
        artist_popularity = artist_info.get("popularity")
        artist_genre = artist_info.get("genres")
        all_artist_popularity[artist_name] = artist_popularity
        all_artist_genre[artist_name] = artist_genre
        count += 1
        print(count)

    with open("artist_genre_responses.json", "w") as file:
        json.dump(all_artist_genre, file, indent=4)
    with open("artist_popularity_responses.json", "w") as file2:
        json.dump(all_artist_popularity, file2, indent=4)
        


def add_track_id_to_df(json_df_1):
    with open("track_search_responses.json", "r", encoding="utf-8") as file:
        uri_dict = json.load(file)

    uri_df = pd.DataFrame(uri_dict.items(), columns=["trackName", "trackId"])
    track_ids = []
    for uri in uri_df["trackId"]:
        split = uri.split(":")
        track_id = split[2]
        track_ids.append(track_id)

    uri_df["trackId"] = track_ids

    merged = json_df_1.merge(uri_df, on="trackName", how="left")

    return merged

def add_artist_pop_to_df(json_df_1):
    with open("artist_popularity_responses.json", "r", encoding="utf-8") as file:
        pop_dict = json.load(file)
    
    pop_df = pd.DataFrame(pop_dict.items(), columns=["artistName", "artistPopularity"])

    merged = json_df_1.merge(pop_df, on="artistName", how="left")

    return merged


def batch_Ids(track_ids):
    batches = []
    for i in range(0, len(track_ids), 100):
        batch = track_ids[i:i+100]
        batches.append(batch)

    print(len(batches))

    return batches

def get_song_features(json_df_1):
    nonrepeat_id = get_nonrepeat_id(json_df_1)
    
    batches = batch_Ids(nonrepeat_id)
    #print(batches)

    load_dotenv("spotify.env")
    token = os.getenv("SPOTIFY_TOKEN")
    headers = {
        "Authorization": f"Bearer {token}"
    }

    track_data = []
    count = 0

    for batch in batches:
        #print(batch)
        #print(len(batch))
        ids_string = ",".join(batch)
        #print(ids_string)
        url = f"https://api.spotify.com/v1/tracks?ids={ids_string}"
        response = requests.get(url, headers=headers)

        response_json = response.json()

        #with open("filey.json", "w") as file:
            #json.dump(response_json, file, indent=4)

        for track in response_json["tracks"]:
            if track:
                track_data.append({
                "trackId": track.get("id"),
                "popularity": track.get("popularity"),
                "explicit": track.get("explicit"),
                "duration_ms": track.get("duration_ms")
            })
        
        count += 1
        #print(count)


    track_df = pd.DataFrame(track_data)

    merged = json_df_1.merge(track_df, on="trackId", how="left")

    merged.to_csv("merged.csv")

    # return merged

def get_portion_played(json_df_1):
    msPlayed_ser = json_df_1["msPlayed"]
    duration_ser = json_df_1["duration_ms"]
    portion = []

    for i in range(len(msPlayed_ser)):
        ratio = msPlayed_ser.iloc[i] / duration_ser.iloc[i]
        if ratio < 0.2:
            portion.append(0)
        elif ratio < 0.4:
            portion.append(1)
        elif ratio < 0.6:
            portion.append(2)
        elif ratio < 0.8:
            portion.append(3)
        elif ratio <= 1.0:
            portion.append(4)
        else:
            portion.append(5)


    json_df_1["portion_played"] = portion

    return json_df_1



def pie_chart(y, title, labels):
    counts = y.value_counts()
    plt.figure()
    plt.pie(counts,  labels=labels, colors=["#ff746c", "#90d5ff"], autopct="%1.1f%%")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def scatter(x, y, title, xlabel, ylabel):
    plt.figure()
    plt.scatter(x, y, marker=".", s=15, c="#90d5ff")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout
    plt.show()

def box_plot(x1, x2, title, ylabel, labels):
    plt.figure()
    plt.title(title)
    plt.xlabel(ylabel)
    plt.xticks([1, 2], labels=labels)
    plt.boxplot([x1, x2])
    plt.show()

def get_artist_group(json_df_1):
    artist_ser = json_df_1["artistName"].copy()
    msPlayed_ser = json_df_1["msPlayed"].copy()
    kanye_ser = []
    drake_ser = []
    for i in range(len(json_df_1["artistName"])):
        if artist_ser.iloc[i] == "Kanye West":
            kanye_ser.append(msPlayed_ser.iloc[i])
        if artist_ser.iloc[i] == "Drake":
            drake_ser.append(msPlayed_ser.iloc[i])
    return kanye_ser, drake_ser

def ms_hypothesis_test(json_df_1, significance, groupby, group):

    df1 = json_df_1.groupby(groupby).get_group(group[0])
    df2 = json_df_1.groupby(groupby).get_group(group[1])

    result = stats.ttest_ind(df1["msPlayed"], df2["msPlayed"], equal_var=False)
    print("pvalue: ", result.pvalue / 2)
    print("T-Statistic", result.statistic)

    if result.pvalue / 2 < significance:
        print("Reject H0")
    else:
        print("Failed to reject H0")

def artist_ms_hypothesis_test(ser1, ser2, significance):

    result = stats.ttest_ind(ser1, ser2, equal_var=False)
    print("pvalue: ", result.pvalue / 2)
    print("T-Statistic", result.statistic)

    if result.pvalue / 2 < significance:
        if result.statistic < 0:
            print("Reject H0")
        else:
            print("Failed to reject H0")

def preprocessing(json_df_1):
    le = LabelEncoder()
    json_df_1["artistName"] = le.fit_transform(json_df_1["artistName"])
    json_df_1["trackName"] = le.fit_transform(json_df_1["trackName"])

    json_df_1.drop("trackId", axis=1, inplace=True)
    

    scaler = MinMaxScaler()
    art_pop_ser = json_df_1["artistPopularity"]
    art_pop_df = art_pop_ser.to_frame()
    art_pop_df = scaler.fit_transform(art_pop_df)

    pop_ser = json_df_1["popularity"]
    pop_df = pop_ser.to_frame()
    pop_df = scaler.fit_transform(pop_df)

    # merged = json_df_1.merge(art_pop_df, on="endTime" , how="left")
    # merged2 = merged.merge(pop_df, on="endTime" , how="left")
    json_df_1["artistPopularity"] = art_pop_ser
    json_df_1["popularity"] = pop_ser
    
    json_df_1.drop("endTime", axis=1, inplace=True)

    

    return json_df_1

def knn(json_df_1, k):
    X = json_df_1.drop(["msPlayed", "portion_played"], axis=1)
    y = json_df_1["portion_played"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y, train_size=0.75)

    knn_clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    
    knn_clf.fit(X_train, y_train)

    y_predicted = knn_clf.predict(X_test)

    knn_acc = accuracy_score(y_test, y_predicted)

    matrix = confusion_matrix(y_test, y_predicted)

    plt.imshow(matrix, cmap="Reds")


    # plt.scatter(y_test, y_predicted)
    # plt.title("Predicted and Actual Categories")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("kNN Confusion Matrix")

    return knn_acc



def decision_tree(json_df_1):
    X = json_df_1.drop(["msPlayed", "portion_played"], axis=1)
    y = json_df_1["portion_played"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y, train_size=0.75)

    tree_clf = DecisionTreeClassifier()

    tree_clf.fit(X_train, y_train)

    y_predicted = tree_clf.predict(X_test)

    tree_acc = accuracy_score(y_test, y_predicted)

    matrix = confusion_matrix(y_test, y_predicted)

    plt.imshow(matrix, cmap="Reds")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Decision Tree Confusion Matrix")
    # plt.figure(figsize=(20, 10))
    # plot_tree(tree_clf, feature_names=X.columns, class_names={0: "r<1/5", 1: "1/5<=r<2/5", 2: "2/5<=r<3/5", 3: "3/5<=r<4/5", 4: "4/5<=r<=1", 5: "r>1"}, filled=True)
    # plt.show()

    return tree_acc