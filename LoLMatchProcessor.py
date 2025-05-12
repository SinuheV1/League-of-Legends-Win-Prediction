

import requests
import pandas as pd
import time
from collections import defaultdict
import os

class LoLMatchProcessor:
    def __init__(self, api_key=None, region='americas'):
        self.api_key = api_key or os.getenv("RIOT_API_KEY")
        if not self.api_key:
            raise ValueError("API key not provided. Set RIOT_API_KEY in .env or pass directly.")
        self.headers = {"X-Riot-Token": self.api_key}
        self.base_url = f"https://{region}.api.riotgames.com/lol/match/v5/matches"

    def get_apex_puuids_ids(self, region='na1'):
        #gets apex tiers puuids, to then get match ids
        tiers = ['challenger', 'grandmaster', 'master']
        tier_endpoints = {
            'challenger': f'https://{region}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5',
            'grandmaster': f'https://{region}.api.riotgames.com/lol/league/v4/grandmasterleagues/by-queue/RANKED_SOLO_5x5',
            'master': f'https://{region}.api.riotgames.com/lol/league/v4/masterleagues/by-queue/RANKED_SOLO_5x5'}
        all_entries = []

        for tier in tiers:
            url = tier_endpoints[tier]
            resp = requests.get(url, headers=self.headers)

            if resp.status_code == 200:
                entries = pd.json_normalize(resp.json().get("entries", []))
                all_entries.append(entries)
            else:
                print(f"Failed to fetch {tier} data. Status: {resp.status_code}")

        if all_entries:
            combined = pd.concat(all_entries, ignore_index=True)
            return combined['puuid'].dropna().unique()
        else:
            print("No players retrieved.")
            return []

    def get_apex_tiers_match_ids(self, puuids, region='americas', delay=10, max_retries=5, return_samples = False):
        #get apex tier match ids by passing puuids to api. Returns 100 match ids and ranked 5v5 queue only(queue = 420)
        
        #return_samples = True returns 4 match ID samples of 5k each
        match_id_list = []

        for i, puuid in enumerate(puuids):
            print(f"Fetching matches for PUUID {i+1}/{len(puuids)}")

            url = f"https://{region}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids?queue=420&start=0&count=50"

            for attempt in range(max_retries):
                resp = requests.get(url, headers=self.headers)
                print(f"Status {resp.status_code} for PUUID {puuid}")

                if resp.status_code == 200:
                    match_ids = resp.json()
                    match_id_df = pd.DataFrame(match_ids, columns=['match_id'])
                    match_id_list.append(match_id_df)
                    break

                elif resp.status_code == 429:
                    print(f"Rate limit hit. Sleeping {delay} seconds (Attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)

                else:
                    print(f"Failed for PUUID: {puuid} with status {resp.status_code}")
                    break
        #very low api rate limits so reducing all match ids to random sample of 20k for processing
        #big enough sample size to get meaningful model results after filtering out bad matches
        if match_id_list:
            full_df = pd.concat(match_id_list, ignore_index=True).drop_duplicates()
            if return_samples:
                full_df = full_df.sample(frac=1, random_state=42)
                sample_dict = {
                    "sample_1": full_df.iloc[:5000],
                    "sample_2": full_df.iloc[5000:10000],
                    "sample_3": full_df.iloc[10000:15000],
                    "sample_4": full_df.iloc[15000:20000]}
                return full_df, sample_dict
            return full_df
        else:
            print("No match IDs retrieved.")
            return pd.DataFrame(columns=['match_id'])

        
    def fetch_with_retry(self, url):
        while True:
            resp = requests.get(url, headers = self.headers)
            print(f" Status {resp.status_code} for {url}")

            try:
                data = resp.json()
            except Exception:
                print(" Could not parse JSON")
                return None

            if resp.status_code == 200:
                print(" Data received:", list(data.keys()))
                return data
            elif resp.status_code == 429:
                print(" 429 rate limit. Sleeping 10s")
                time.sleep(10)
            elif resp.status_code == 403:
                print(" 403 Forbidden. Is your API key expired?")
                return None
            else:
                print(f" Unexpected response: {data}")
                return None
            

    def fetch_match_data(self, match_id):
        url = f"{self.base_url}/{match_id}"
        return self._safe_get(url)

    def fetch_timeline_data(self, match_id):
        url = f"{self.base_url}/{match_id}/timeline"
        return self._safe_get(url)

    def _safe_get(self, url, max_retries=3):
        for _ in range(max_retries):
            resp = requests.get(url, headers=self.headers)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep(10)
            elif resp.status_code in (403, 401, 404):
                return None
        return None

    def get_14_min_stats(self, match_id, match_data, timeline_data):
        #get 14 min stats of all games passed through. filter out non ranked 5v5 games(queue = 420)
        #and anything on patch 15.9
        
        queue_id = match_data['info'].get('queueId')
        game_version = match_data['info'].get('gameVersion', '')
        
        #remove newest patch only (15.9 at this time)
        if game_version.startswith("15.9"):
            print(f"Skipping {match_id} — patch 15.9 (version: {game_version})")
            return pd.DataFrame()
        
        # Skip non-ranked matches
        if queue_id != 420:
            return pd.DataFrame()

        frames = timeline_data['info']['frames']
        
        #moving info we need to participants df
        participants = pd.json_normalize(match_data['info']['participants'])[
            ['participantId', 'championName', 'teamId','firstBloodKill',
            'teamPosition','win']]
        #checking to make sure game does last 14 minutes
        if len(frames) <= 14 or 'participantFrames' not in frames[14]:
            print(f"{match_id} is too short or missing frame 14")
            return pd.DataFrame()
        
        #for loop to get all values from each key for the 14th minute data of the game
        #store it in an empty list and append to it
        frame14_data = []
        for id, data in frames[14]['participantFrames'].items():
            data['participantId'] = int(id)
            frame14_data.append(data)

        minute14_df = pd.json_normalize(frame14_data)

        #merging both dataframes on participantID 
        players = participants.merge(minute14_df, on='participantId')
        #adding gold per minute and cs(creep score) per minute
        players['goldPerMinute'] = players['totalGold']/ 14
        players['csPerMinute'] = (players['minionsKilled'] + players['jungleMinionsKilled'])/ 14
        players['match_id'] = match_data['metadata']['matchId']

        kills_14 = defaultdict(int)
        deaths_14 = defaultdict(int)
        assists_14 = defaultdict(int)
        plates_taken = defaultdict(int)
        towers_killed = defaultdict(int)
        dragons = defaultdict(int)
        horde_kills = defaultdict(int)
        wards_placed = defaultdict(int)

        #minute 0-14
        for frame in frames[:15]:
                for event in frame.get('events', []):
                    t = event.get('timestamp', 0)
                    if t > 840000:
                        continue
                    #get champion kill events and assign to correct participant
                    if event['type'] == 'CHAMPION_KILL':
                        killer = event.get('killerId')
                        victim = event.get('victimId')
                        assist = event.get('assistingParticipantIds',[])
                        if killer:
                            kills_14[killer] +=1
                        if victim:
                            deaths_14[victim] +=1
                        for aid in assist:
                            assists_14[aid] +=1

                    #plate destruction and assign to correct participant
                    if event['type'] == 'TURRET_PLATE_DESTROYED':
                        id = event.get('killerId')
                        if id:
                            plates_taken[id] += 1
        
                    #tower kills and assign to correct participant
                    if event['type'] == 'BUILDING_KILL' and event.get('buildingType') == 'TOWER_BUILDING':
                        id = event.get('killerId')
                        if id:
                            towers_killed[id] += 1
        
                    #track dragon kills and assign to correct team
                    #track void grub kills(HORDE) and assign to correct team
                    if event['type'] == 'ELITE_MONSTER_KILL':
                        team_id = event.get('killerTeamId')
                        if team_id:
                            if event.get('monsterType') == 'DRAGON':
                                dragons[team_id] += 1
                            elif event.get('monsterType') == 'HORDE':
                                horde_kills[team_id] += 1
                    #track wards placed and assign to correct participant       
                    if event['type'] == 'WARD_PLACED':
                        ward_type = event.get('wardType')
                        pid = event.get('creatorId')
                        if ward_type in ['YELLOW_TRINKET', 'BLUE_TRINKET', 'CONTROL_WARD']:
                            wards_placed[pid] += 1
                        
        #map all of these features tracked  to the correct participant/team and return them                  
        players['platesTaken_14'] = players['participantId'].map(plates_taken).fillna(0).astype(int)
        players['towersKilled_14'] = players['participantId'].map(towers_killed).fillna(0).astype(int)
        players['teamDragonKills_14'] = players['teamId'].map(dragons).fillna(0).astype(int)
        players['teamHordeKills_14'] = players['teamId'].map(horde_kills).fillna(0).astype(int)
        players['kills_14'] = players['participantId'].map(kills_14).fillna(0).astype(int)
        players['assists_14'] = players['participantId'].map(assists_14).fillna(0).astype(int)
        players['deaths_14'] = players['participantId'].map(deaths_14).fillna(0).astype(int)
        players['wards_placed'] = players['participantId'].map(wards_placed).fillna(0).astype(int)
        #adding gold per minute and cs(creep score) per minute
        players['goldPerMinute'] = players['totalGold']/ 14
        players['csPerMinute'] = (players['minionsKilled'] + players['jungleMinionsKilled'])/ 14
        players['match_id'] = match_data['metadata']['matchId']
        #rename the team position utility to support. convert TRUE/FALSE win to 1/0 
        players['teamPosition'] = players['teamPosition'].replace({'UTILITY': 'SUPPORT'})
        players['win'] = players['win'].astype(int)
        players['firstBloodKill'] = players['firstBloodKill'].astype(int)
        players['totalMinionsKilled'] = players['minionsKilled'] + players['jungleMinionsKilled']
        return players[[
                'match_id','participantId','championName',
                'totalGold', 'goldPerMinute',
                'minionsKilled', 'jungleMinionsKilled',
                'totalMinionsKilled','csPerMinute',
                'xp', 'level','wards_placed',
                'kills_14', 'deaths_14', 'assists_14',
                'platesTaken_14', 'towersKilled_14', 'firstBloodKill',
                'teamDragonKills_14', 'teamHordeKills_14','teamId','teamPosition','win']]


    def process_matches_batch(self, match_ids, batch_size=50, pause_time=120, checkpoint_path=None):
        #if checkpoint_path = True then periodic saving to the checkpoint path after each batch processed.
        
        all_data = []
        start_time = time.time()

        for i in range(0, len(match_ids), batch_size):
            batch = match_ids[i:i + batch_size]
            print(f"\n Processing matches {i+1} to {i+len(batch)} of {len(match_ids)}")

            for j, match_id in enumerate(batch, 1):
                try:
                    print(f"{match_id} ({i+j}/{len(match_ids)})")

                    match_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}"
                    match_data = self.fetch_with_retry(match_url)

                    if not match_data or 'info' not in match_data:
                        print(f" Skipping {match_id} — match data invalid")
                        continue

                    timeline_url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
                    timeline_data = self.fetch_with_retry(timeline_url)

                    if not timeline_data or 'info' not in timeline_data:
                        print(f"Skipping {match_id} — timeline data invalid")
                        continue

                    df = self.get_14_min_stats(match_id, match_data, timeline_data)

                    if df.empty:
                        print(f"Match {match_id} returned no valid 14-min stats (skipped)")
                    else:
                        all_data.append(df)

                except Exception as e:
                    print(f"Error processing {match_id}: {e}")
                    continue

            #save partial results after each batch
            if checkpoint_path and all_data:
                partial_df = pd.concat(all_data, ignore_index=True)
                partial_df.to_csv(checkpoint_path, index=False)
                print(f"Checkpoint saved to {checkpoint_path}")

            #throttle to avoid Riot rate limits
            if i + batch_size < len(match_ids):
                print(f" Sleeping {pause_time} seconds to respect Riot API rate limits...")
                time.sleep(pause_time)

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            print(f"\nDone. Processed {len(final_df)} rows in {round(time.time() - start_time, 2)} seconds.")
            return final_df
        else:
            print("\n No valid matches were processed in any batch.")
            return pd.DataFrame()
        
    def role_gold_diff(players):
        #calculates role gold diffs for all 5 roles in each match and zero centers them
        players = players.copy()
        players['roleGoldDiff'] = 0.0

        #group by match id
        for match_id, match_df in players.groupby('match_id'):
            #for each match id compare role gold diff
            for role, role_group in match_df.groupby('teamPosition'):
                #validate there is 2 of the same roles for each of the 5 roles in that match
                if len(role_group) != 2:
                    continue

                p1, p2 = role_group.iloc[0], role_group.iloc[1]
                diff = p1['totalGold'] - p2['totalGold']
                half_diff = abs(diff) / 2

                if diff > 0:
                    players.loc[p1.name, 'roleGoldDiff'] = +half_diff
                    players.loc[p2.name, 'roleGoldDiff'] = -half_diff
                else:
                    players.loc[p1.name, 'roleGoldDiff'] = -half_diff
                    players.loc[p2.name, 'roleGoldDiff'] = +half_diff
        return players
    
    def role_xp_diff(players):
        #calculates role xp diff for all 5 roles in each match and zero centers them
        players = players.copy()
        players['roleXpDiff'] = 0.0
        #group by match id and validate there is 2 of the same roles for each of the 5 roles in that match
        for match_id, match_df in players.groupby('match_id'):
            for role, role_group in match_df.groupby('teamPosition'):
                if len(role_group) !=2:
                    continue 
                p1,p2 = role_group.iloc[0], role_group.iloc[1]
                diff = p1['xp'] - p2['xp']
                half_diff = abs(diff) / 2
        
                if diff > 0:
                    players.loc[p1.name, 'roleXpDiff'] = +half_diff
                    players.loc[p2.name, 'roleXpDiff'] = -half_diff
                else:
                    players.loc[p1.name, 'roleXpDiff'] = -half_diff
                    players.loc[p2.name, 'roleXpDiff'] = +half_diff
        return players
    
    def role_cs_diff(players):
        #calculates role cs diff for all 5 roles in each match and zero centers them
        players = players.copy()
        players['roleCsDiff'] = 0.0
        #group by match id and validate there is 2 of the same roles for each of the 5 roles in that match
        for match_id, match_df in players.groupby('match_id'):
            for role, role_group in match_df.groupby('teamPosition'):
                if len(role_group) !=2:
                    continue
                p1,p2 = role_group.iloc[0], role_group.iloc[1]
                diff = p1['totalMinionsKilled'] -p2['totalMinionsKilled']
                half_diff = abs(diff) / 2

                if diff > 0:
                    players.loc[p1.name, 'roleCsDiff'] = +half_diff
                    players.loc[p2.name, 'roleCsDiff'] = -half_diff
                else:
                    players.loc[p1.name, 'roleCsDiff'] = -half_diff
                    players.loc[p2.name, 'roleCsDiff'] = +half_diff 
        return players

    def role_kill_diff(players):
        #calulates role kill diff for all 5 roles in each match and zero centers them
        players = players.copy()
        players['roleKillDiff'] = 0.0
        #group by match id and validate there is 2 of the same roles for each of the 5 roles in that match
        for match_id, match_df in players.groupby('match_id'):
            for role, role_group in match_df.groupby('teamPosition'):
                if len(role_group) !=2:
                    continue
                p1,p2 = role_group.iloc[0], role_group.iloc[1]
                diff = p1['kills_14'] - p2['kills_14']
                half_diff = abs(diff) / 2

                if diff >0:
                    players.loc[p1.name, 'roleKillDiff'] = +half_diff
                    players.loc[p2.name, 'roleKillDiff'] = -half_diff
                else:
                    players.loc[p1.name, 'roleKillDiff'] = -half_diff
                    players.loc[p2.name, 'roleKillDiff'] = +half_diff
        return players
    
    def role_deaths_diff(players):
        #caluclate role deaths diff for all 5 roles in each match and zero centers them
        players = players.copy()
        players['roleDeathsDiff'] = 0.0
        #group by match id and validate there is 2 of the same roles for each of the 5 roles in that match
        for match_id, match_df in players.groupby('match_id'):
            for role, role_group in match_df.groupby('teamPosition'):
                if len(role_group) !=2:
                    continue
                p1,p2 = role_group.iloc[0], role_group.iloc[1]
                diff = p1['deaths_14'] - p2['deaths_14']
                half_diff = abs(diff) / 2

                if diff >0:
                    players.loc[p1.name, 'roleDeathsDiff'] = +half_diff
                    players.loc[p2.name, 'roleDeathsDiff'] = -half_diff
                else:
                    players.loc[p1.name, 'roleDeathsDiff'] = -half_diff
                    players.loc[p2.name, 'roleDeathsDiff'] = +half_diff
        return players
        
    def role_vision_diff(players):
        #calculate role vision diff for all 5 roles in each match and zero centers them
        players = players.copy()
        players['roleVisionDiff'] = 0.0
        #group by match id and validate there is 2 of the same roles for each of the 5 roles in that match
        for match_id, match_df in players.groupby('match_id'):
            for role, role_group in match_df.groupby('teamPosition'):
                if len(role_group) !=2:
                    continue
                p1,p2 = role_group.iloc[0], role_group.iloc[1]
                diff = p1['wards_placed'] - p2['wards_placed']
                half_diff = abs(diff) / 2

                if diff >0:
                    players.loc[p1.name, 'roleVisionDiff'] = +half_diff
                    players.loc[p2.name, 'roleVisionDiff'] = -half_diff
                else:
                    players.loc[p1.name, 'roleVisionDiff'] = -half_diff
                    players.loc[p2.name, 'roleVisionDiff'] = +half_diff
        return players
