import json
import copy
original_dataset = []
path = "" # your dataset path
mode = "both" # choose SELF or BOTH in DuLeMon
type_ = "dev" # choose train, test or dev
path = path + mode + "/" + type_ + ".json" # concat path
with open(path,'r',encoding='utf-8') as f:
    for line in f.readlines():
        original_dataset.append(json.loads(line))
    
dataset = []
for conversation in original_dataset:
    conversation_history = []
    if len(conversation['conversation']) % 2 != 0:
        continue
    for utterance_num in range(0, len(conversation['conversation']), 2):
        query = conversation['conversation'][utterance_num].split(":")[1].strip(" ")
        response = conversation['conversation'][utterance_num + 1].replace("：",":")
        golden_response = response.split(":")[1].strip(" ").split("\t")[0]
        
        if mode == "self":
            personas = [[], []]
            for p in conversation['p1_persona'] + conversation['p2_persona']:
                if ":" not in p:
                    personas[1].append(p.strip(" "))
                else:
                    personas[1].append(p.split(":")[1].strip(" "))

        elif mode == "both":
            personas = [[], []]
            for p in conversation['bot_persona']:
                if ":" not in p:
                    personas[1].append(p.strip(" "))
                else:
                    personas[1].append(p.split(":")[1].strip(" "))

            for p in conversation['user_said_persona']:
                if ":" not in p:
                    personas[0].append(p.strip(" "))
                else:
                    personas[0].append(p.split(":")[1].strip(" "))
        else:
            raise TypeError("Unknown mode")
            
        dataset.append({
            "query": query,
            "response": golden_response,
            "context": copy.deepcopy(conversation_history),
            "init_personas": personas
        })

        conversation_history.append("User: " + query)
        conversation_history.append("ChatBot: " + golden_response)

output_path = "" # your output path
with open(output_path,'w',encoding='utf-8') as f:
    f.write(json.dumps(dataset, indent=4, ensure_ascii=False))
