import json
import tqdm
import random
with open("tiage/test.json",'r',encoding='utf-8') as f:
    dataset = json.load(f)

res = []
for data in tqdm.tqdm(dataset['dial_data']['tiage']):
    data = data['turns']
    pre_memory = ""
    pre_topic = -1
    for utterance in data:
        if utterance['role'] == 'user':
            utterance_data = "USER: " + utterance['utterance']
        else:
            utterance_data = "Chatbot: " + utterance['utterance']
         
        if utterance['topic_id'] != pre_topic and pre_topic != -1:
            res.append({
                "input": pre_memory + "[SEP]" + utterance_data,
                "label": 1 
            })
            pre_topic = utterance['topic_id']
            pre_memory = utterance_data
        elif pre_topic == -1:

            pre_topic = utterance['topic_id']
            pre_memory = pre_memory + "\n" + utterance_data
        else:
            res.append({
                "input": pre_memory + "[SEP]" + utterance_data,
                "label": 0
            })
            pre_memory = pre_memory + "\n" + utterance_data
random.shuffle(res)
with open("output/test_dataset.json",'w',encoding='utf-8') as f:
    f.write(json.dumps(res, indent=4, ensure_ascii=False))