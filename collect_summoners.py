import requests

import databaseQuries
from RiotApiCalls import *
import config

def collect_summoner_names(start_summoner_name, region_start, tagline, output_file):
    collected_names = set()
    current_summoner_name = start_summoner_name
    
    try:
        for _ in range(1):  # Adjust the range as needed
            Summoner = getPuuid(region_start, current_summoner_name, tagline)
            print(Summoner)
            puuid = Summoner['puuid']

            collected_names.add((current_summoner_name, tagline))
            print(f"Collected: {current_summoner_name}#{tagline}")
            
            # Collect other players' names in the matches
            MatchIDs = requests.get(f"https://{region_start}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?start=0&count=20&api_key=" + api_key).json()
            for MatchId in MatchIDs:
                MatchData = requests.get(f"https://{region_start}.api.riotgames.com/lol/match/v5/matches/{MatchId}?api_key=" + api_key).json()
                print(f"https://{region_start}.api.riotgames.com/lol/match/v5/matches/{MatchId}?api_key=" + api_key)
                participants = MatchData['info']['participants']
                for participant in participants:
                    collected_names.add((participant['summonerName'], participant['riotIdTagline']))
     
    except:    
        pass

    with open(output_file, 'a', encoding='utf-8') as f:  # Append mode to avoid overwriting existing names
        for name, tagline in collected_names:
            if tagline == "EUW":
                f.write(f"{name} #{tagline}\n")




if __name__ == "__main__":
    start_summoner_name = "Hiκari"
    region_start = "europe"
    tagline = "1337"
    output_file = "summoner_names.txt"
    collect_summoner_names(start_summoner_name, region_start, tagline, output_file)