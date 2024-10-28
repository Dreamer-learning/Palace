# Persona Prompt Learning

## 1. Persona triples extraction
We follow the method in [Link](https://ojs.aaai.org/index.php/AAAI/article/view/26545), you can also train the model with DNLI or use another method.

You should extract all the persona triples from the defined personas, context, and current query, and place the corresponding generated JSON files in the `./entity_link` folder.

the format of JSON files is :

`  
{
  
    "I also do gymnastics, I love it": "<I, like_activity, gymnastics>",    .#"query/context/personas": <persona triples>
  
    ...
  
}
`
## 2. knowledge graph link
You need first to process the original knowledge graph file to generate the corresponding persona_triple.json and persona_triple_dict.json

the format of persona_triple.json is 

`
[
   [
        "i am an actor who play my part", # entity
        "characteristic", # relation
        "complicent" # entity
    ],
    ...
]
`

the format of persona_triple_dict.json is 

`
{
  "i am an actor who play my part": [ # entity
        [
            "characteristic", # relation
            "complicent" # entity
        ],
        ...
}
`

You can also download the processed knowledge graph in this link [Link]()


You can modify the file path in the code to point to the JSON files of the extracted persona triples and run:

`python peacok_link.py`
## 3. Initialize representation 
To accelerate training, we pre-store the vectors that need to be initialized as .npy files. After modifying the corresponding file paths, you can run get_npy.py to obtain the initial representations of the current query and context:

`python get_npy.py`

You can run get_npy_triples.py to get the initial representations of the nodes in the graph.

`python get_npy_triples.py`


## 4. training
After completing the above steps and obtaining the corresponding files, modify the file paths and run:

·bash ppl.sh`
