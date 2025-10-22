This project goes through my Spotify Data to find interesting stats and see what Spotify Wrapped missed.

The project's purpose is to help me see some tendencies I have when listening to music. This is just for my own enjoyment and will likely not affect anything.

**Very Important**

In my second code cell I have the functions that call the API. These are `utils.get_song_uri(json_df_1)` which ended up becoming obselete, `utils.get_artist_info(json_df_1)`, and `utils.get_song_features(json_df_1)`. All of these only needed to be ran once and I have their outputs written to files that other parts of the code can access. **THEY WERE THEN COMMENTED OUT**. You can not run these functions without your own Spotify Web API credentials. But the functions should still work without making the calls because I have the written files in the folder.

Files from the API Calls include:
artist_genre_responses.json  
artist_popularity_responses.json  
Playlist1.json  
StreamingHistory_music_1.json  
track_search_responses.json  

**These files contain my private data so I have kept these files private for the sake of presenting the project on github.**

I am the only one who maintains this project and adds to it
