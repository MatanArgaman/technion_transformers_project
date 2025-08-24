import argparse
import os
import pickle
import numpy as np
import json
from typing import List, Dict, Tuple

def filter_on_topic(x: List[Dict], docs: List[Dict], ordered_topics, topic_index: int) -> Tuple[List[Dict], Dict]:
    topic_id = ordered_topics[topic_index]
    indices = set(np.where(data['topics'] == topic_id)[0])
    doc_set = set([d['id'] for i, d in enumerate(docs) if i in indices])
    y = [a for a in x if a['relevant_docs'][0] in doc_set]

    topic_name = data['dataframe'][data['dataframe']['Topic'] == topic_id]['Name'].item()
    return y, {'topic_id': int(topic_id), 'topic_name': topic_name}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_topics', help='to topics.pkl')
    parser.add_argument('--path_to_data', type=str, default="C:\\projects\\transformers\\236004-HW1-GPT\\NQ10k\\", help='path to test.json/docs.json')
    parser.add_argument('-k', type=int, default=10, help='top k highest mean confidence topics will be selected')
    parser.add_argument('-min_queries', type=int, default=10, help='minimum queries for selected topic')
    parser.add_argument('-min_samples', type=int, default=10, help='only topics with document count >= min_samples are selected')
    args =parser.parse_args()

    out_path = os.path.join(os.path.dirname(args.path_to_topics), 'selected.npy')

    with open(args.path_to_topics, 'rb') as fp:
        data = pickle.load(fp)

    with open(os.path.join(args.path_to_data, 'test_queries-10000-7423.json')) as fp:
        test = json.load(fp)
    with open(os.path.join(args.path_to_data, 'documents-10000-7423.json')) as fp:
        docs = json.load(fp)

    df = data['dataframe']
    filter = df['Count'] >= args.min_samples
    df_filtered = df[filter]
    topics = []
    confidence = []
    for i in range(len(df_filtered['Topic'])):
        l = np.array(data['confidence'])[np.array(data['topics']) == df_filtered['Topic'].iloc[i]]
        topics.append(df_filtered['Topic'].iloc[i])
        confidence.append(l.mean())

    order = (np.argsort(confidence))[::-1]
    ordered_topics = np.array(topics)[order]
    selected_topics = []
    for i, s in enumerate(ordered_topics):
        y, d = filter_on_topic(test, docs, ordered_topics, i)
        if len(y) >= args.min_queries:
            print('queries: ', len(y), f'documents: {df[df["Topic"]==s]["Count"].item()}', f'confidence: {confidence[order[i]]:.3f}', f'topic: {d["topic_name"]}', 'index: ', s)
            selected_topics.append(s)
        if len(selected_topics) >= args.k:
            break
    np.save(out_path, selected_topics)
    '''
    queries:  89 documents: 1000 confidence: 0.998 topic: 0_song_album_singles_chart index:  0
    queries:  47 documents: 501 confidence: 0.978 topic: 1_film_films_disney_movie index:  1
    queries:  13 documents: 78 confidence: 0.894 topic: 14_bank_financial_business_banking index:  14
    queries:  17 documents: 205 confidence: 0.871 topic: 2_season_series_tv_2017 index:  2
    queries:  14 documents: 59 confidence: 0.708 topic: 24_game_games_nintendo_xbox index:  24
    queries:  10 documents: 80 confidence: 0.698 topic: 12_hai_dil_singh_ki index:  12
    queries:  12 documents: 89 confidence: 0.684 topic: 9_india_indian_gandhi_party index:  9
    queries:  12 documents: 85 confidence: 0.604 topic: 10_nerve_muscle_anterior_thyroid index:  10
    queries:  21 documents: 165 confidence: 0.587 topic: 3_actor_he_episode_film index:  3
    queries:  13 documents: 63 confidence: 0.553 topic: 21_olympic_olympics_games_skating index:  21

    '''
    '''
    indices = np.load('C:\\projects\\transformers\\236004-HW1-GPT\\DSI-large-7423\\selected.npy')
    for v in indices:
        l = np.array(data['confidence'])[np.array(data['topics']) == v]
        print(f'mean: {l.mean()}, std: {l.std()}')
    print(df[df['Topic'].isin(selected)][['Name','Count']])
    '''
    '''
    mean: 0.9975256429865508, std: 0.022418379775810442
    mean: 0.996167973725796, std: 0.011031984924444252
    mean: 0.9782765197207695, std: 0.061889573309460426
    mean: 0.9746339526600691, std: 0.038171051769998916
    mean: 0.9721919465099397, std: 0.03682472164760808
    mean: 0.9721826259385886, std: 0.06350812729068184
    mean: 0.9694696242795242, std: 0.038871915588136456
    mean: 0.9601036791771655, std: 0.048801636593269616
    mean: 0.9588462713327437, std: 0.050109980806786296
    mean: 0.9439674479749384, std: 0.07872826290538176
    
                                    Name  Count
    1         0_song_album_singles_chart   1000
    2          1_film_films_disney_movie    501
    9          8_nfl_bowl_super_patriots     90
    12    11_river_trail_oregon_mountain     82
    38        37_spanish_de_spain_mexico     42
    66   65_soil_water_erosion_pollution     32
    81    80_sesame_jem_spongebob_street     27
    90    89_logic_philosophy_history_of     25
    93     92_india_labour_indian_sector     24
    103     102_ikea_stores_target_store     22

    '''

