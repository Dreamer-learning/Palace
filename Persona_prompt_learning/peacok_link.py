import json
from memory import BiEncoderRetriever
import tqdm 
def get_relations_dict():
    relations_dict = {
                      "characteristic": "here is my character trait",
                      "characteristic_relationship": "here is my character trait related to other people or social groups",
                      #"routine_habit": 4,
                      "routine_habit_relationship": "here is what I regularly or consistently do related to other people or social groups",
                      #"goal_plan": 6,
                      "goal_plan_relationship": "here is what I will do or achieve in the future related to other people or social groups",
                      "experience_relationship": "here is what I did in the past related to other people or social groups",
                      #"experience": 9
                      }
    return relations_dict


def main():
    retrieval = BiEncoderRetriever()
    with open("./peacok_triple.json",'r',encoding='utf-8') as f:
        kg = json.load(f)
    with open("./peacok_triple_dict.json",'r',encoding='utf-8') as f:
        kg_head = json.load(f)

    head = list(kg_head.keys())    
    
    kg_triple_sentence = {}
    relations_dict = get_relations_dict()
    for triple in kg:
        if triple[1] not in relations_dict.keys():
            continue
        if triple[0] not in kg_triple_sentence.keys():
            kg_triple_sentence[triple[0]] = []
        sentence = triple[0] + "," + relations_dict[triple[1]] + "," + triple[2]
        kg_triple_sentence[triple[0]].append(sentence)
    #print(kg_triple_sentence)

    encode_summaries = retrieval.encode_summaries(list(head))
    encode_summaries = encode_summaries.cpu().numpy()

    with open("./session_5/test.json",'r',encoding='utf-8') as f: # original dataset
        dataset = json.load(f)
    
    res = {}
    for data in tqdm.tqdm(dataset):
        """context = data['context']
        for session in context:
            for utterance in session:
                memory_retrieved_query = retrieval.retrieve_top_summaries(question = utterance, summaries=list(head), encoded_summaries = encode_summaries)
                if len(memory_retrieved_query) == 0:
                    res[utterance] = "None"
                else :
                    res[utterance] = memory_retrieved_query[0]
                    memory_retrieved_triple = retrieval.retrieve_top_summaries(question = utterance, summaries=list(kg_triple_sentence[memory_retrieved_query[0]]), topk = 20)
                    res[utterance] = memory_retrieved_triple"""

        query = data['query']
        if query not in res.keys():
            memory_retrieved_query = retrieval.retrieve_top_summaries(question = query, summaries=list(head), encoded_summaries = encode_summaries)
            if len(memory_retrieved_query) == 0:
                res[query] = "None"
            else :
                res[query] = memory_retrieved_query[0]
                memory_retrieved_triple = retrieval.retrieve_top_summaries(question = query, summaries=list(kg_triple_sentence[memory_retrieved_query[0]]), topk = 20)
                res[query] = memory_retrieved_triple

        init_personas = data['init_personas']
        for persona_list in init_personas:
            for persona in persona_list:
                if persona not in res.keys():
                    memory_retrieved_query = retrieval.retrieve_top_summaries(question = persona, summaries=list(head), encoded_summaries = encode_summaries)
                    if len(memory_retrieved_query) == 0:
                        res[persona] = "None"
                    else :
                        res[persona] = memory_retrieved_query[0]
                        memory_retrieved_triple = retrieval.retrieve_top_summaries(question = persona, summaries=list(kg_triple_sentence[memory_retrieved_query[0]]), topk = 20)
                        res[persona] = memory_retrieved_triple
        
    with open("./entity_link/session_5/peacok_linked_query_and_persona_test.json",'w',encoding='utf-8') as f:
        f.write(json.dumps(res, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
