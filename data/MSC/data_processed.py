import json
import tqdm
import copy
context = {}
dataset = []

"""
with open("dataset/session_1/valid.txt",'r',encoding='utf-8') as f:
    for line in f.readlines(): 
        line = json.loads(line)
        id = line['initial_data_id']
        if id not in context.keys():
            context[id] = []
            for i in range(4):
                context[id].append([])
        dialogs = line['dialog']
        for dialog in dialogs:
            context[id][0].append(dialog['text'])
        
with open("dataset/session_2/valid.txt",'r',encoding='utf-8') as f:
    for line in f.readlines():
        line = json.loads(line)
        id = line['initial_data_id']
        if id not in context.keys():
            print(id)
        dialogs = line['dialog']
        for dialog in dialogs:
            context[id][1].append(dialog['text'])

with open("dataset/session_3/valid.txt",'r',encoding='utf-8') as f:
    for line in tqdm.tqdm(f.readlines()):
        line = json.loads(line)
        id = line['initial_data_id']
        if id not in context.keys():
            print(id)
        init_personas = line['init_personachat']['init_personas']
        #init_personas = line['init_personachat']['personas']
        #init_personas[0] = init_personas[0] + line['init_personachat']['init_personas'][0]
        #init_personas[1] = init_personas[1] + line['init_personachat']['init_personas'][1]
        agg_persona_list_1 = line['dialog'][-2]['agg_persona_list']
        agg_persona_list_2 = line['dialog'][-1]['agg_persona_list']
        agg_persona_list = []
        agg_persona_list.append(agg_persona_list_1)
        agg_persona_list.append(agg_persona_list_2)
        for i in range(0,len(line['dialog']),2):
            if i+1 >= len(line['dialog']):
                continue
            query = line['dialog'][i]['text']
            response = line['dialog'][i+1]['text']
            personas = []
            if "problem_data" in line['dialog'][i]:
                if "persona" in line['dialog'][i]["problem_data"]:
                    personas.append(line['dialog'][i]["problem_data"]["persona"])
                else:
                    personas.append("")
            else:
                personas.append("")
            
            if "problem_data" in line['dialog'][i+1]:
                if "persona" in line['dialog'][i+1]["problem_data"]:
                    personas.append(line['dialog'][i+1]["problem_data"]["persona"])
                else:
                    personas.append("")
            else:
                personas.append("")
            
            dataset.append({
                'id':id + "_" + str(int(i/2)),
                'query':query,
                'response':response,
                'context': copy.deepcopy(context[id]),
                'personas': personas,
                'init_personas': init_personas,
                'agg_persona_list': agg_persona_list
            })
            context[id][2].append(query)
            context[id][2].append(response)

with open("/home/liudongshuo/Lora-test/data/session_3/valid.json",'w',encoding='utf-8') as f:
    f.write(json.dumps(dataset, indent=4, ensure_ascii=False))"""

json_dataset = []
with open("/home/liudongshuo/Lora-test/dataset/session_2/test.txt",'r',encoding='utf-8') as f:
    for line in tqdm.tqdm(f.readlines()): 
        line = json.loads(line)
        init_personas = line['personas']
        speak_1 = []
        speak_2 = []
        for p in init_personas[0]:
            for p_split in p.split(". "):
                if not p_split.endswith("."):
                    p_split =  p_split + "."
                speak_1.append(p_split)  

        for p in init_personas[1]:
            for p_split in p.split(". "):
                if not p_split.endswith("."):
                    p_split =  p_split + "."
                speak_2.append(p_split)      

        context = [[],[],[],[],[]]
        for session_num in range(len(line['previous_dialogs'])):
            session = line['previous_dialogs'][session_num]['dialog']
            for utterance in session:
                context[session_num].append(utterance['text'])
        
        for utterance_num in range(0, len(line['dialog']), 2):
            if utterance_num + 1 == len(line['dialog']):
                break
            id = line['dialog'][utterance_num]['convai2_id']
            query = line['dialog'][utterance_num]['text']
            response = line['dialog'][utterance_num + 1]['text']

            json_dataset.append({
                'id':id + "_" + str(int(utterance_num/2)),
                'query':query,
                'response':response,
                'context': copy.deepcopy(context),
                'init_personas': [speak_1, speak_2],
            })

            context[len(line['previous_dialogs'])].append(query)
            context[len(line['previous_dialogs'])].append(response)


with open("/home/liudongshuo/Lora-test/data/session_2/test.json",'w',encoding='utf-8') as f:
    f.write(json.dumps(json_dataset, indent=4, ensure_ascii=False))



