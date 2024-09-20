import json
import re
import pandas as pd


triples = pd.read_csv(f"archive/test.txt", delimiter="\t", names=["head", "relation", "tail"])
entity2definition_df = pd.read_csv(f"archive/entity2definition.txt", delimiter="\t", names=["id", "desc"])
entity2definition = dict(zip(entity2definition_df['id'], entity2definition_df['desc']))
entity2label_df = pd.read_csv(f"archive/entity2label.txt", delimiter="\t", names=["id", "desc"])
entity2label = dict(zip(entity2label_df['id'], entity2label_df['desc']))

with open(f"archive/relation2label.json") as f:
    relation2label = json.load(f)
with open(f"archive/relation2template.json") as f:
    relation2template = json.load(f)

def get_match_location(sentence, pattern):
    match = re.search(pattern, sentence)
    if match:
        return match.start()
    else:
        return None

def do_verb(template, head):
    pattern_y = r'\[Y\]'
    pattern_x = r'\[X\]'
    y_location = get_match_location(template, pattern_y)
    x_location = get_match_location(template, pattern_x)
    if y_location < x_location:
        return f'{template}'.replace(
            '[X]', f'[{head}] ?').replace(
            '[Y]', f'___ ')
    else:
        return f'{template}'.replace(
            '[X]', f'[{head}]').replace(
            '[Y]', f'?')
sent = []
entity_used = dict()
for i in range(10):
    h, r, t = triples.loc[i]['head'], triples.loc[i]['relation'], triples.loc[i]['tail']

    # entity_used[entity2label[h]] = entity2definition[h]
    entity_used[entity2label[t]] = entity2definition[t]

    template  = relation2template[r]
    text = do_verb(template, entity2label[h])
    # text = f'{template}'.replace(
    #     '[X]',f'[{entity2label[h]}]').replace(
    #     '[Y]', f'?')

    text = f'{text}\n\n Where description of [{entity2label[h]}] is [{entity2definition[h]}].'

    sent.append({
        'head':h,
        'head_ent':entity2label[h],
        'relation':r,
        'tail':t,
        'tail_ent': entity2label[t],
        'input_text': text
    })
ent_desc = (f'Here, I am providing {len(entity_used)} target entity those are enclosed by [] bracket, also their description is given below. Read it carefully.'
            f'I will ask you some question from those entities. \n\n')
for i, k in enumerate(entity_used):
    ent_desc += f'{i+1}. Description of [{k}] is [{entity_used[k]}].\n'

with open('archive/questions_and_desc.json', 'w') as f:
    json.dump({
        'question':sent,
        'target_ent':ent_desc
    }, f, indent=2)