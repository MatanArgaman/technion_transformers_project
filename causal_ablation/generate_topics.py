from bertopic import BERTopic
import json
import os
import pickle

path_to_data = "C:\\projects\\transformers\\236004-HW1-GPT\\NQ10k\\"
with open(os.path.join(path_to_data, 'documents-10000-7423.json')) as fp:
  docs = json.load(fp)

doc_list = [d['text'] for d in docs]

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(doc_list)

df = topic_model.get_topic_info()
with open('topics.pkl', 'wb') as fp:
    pickle.dump({'topics': topics, 'confidence':probs, 'dataframe':df}, fp)

#result downloadable in: https://1drv.ms/u/c/dfade318d5bbe66c/EVVoKtvOndlJjrNWAXKmNdMBc3RIF_Y3eYE2JF5boAiKGA?e=SYJU35