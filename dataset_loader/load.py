import json
from dataset_loader.config import DATASET_DIR


class Loader():
    def get_data(self, file_path):
        pass


class Wikidata5mLoader(Loader):

    def get_data(self):

        file_path = f"{DATASET_DIR}/wikidata5m_alias/"
        
        with open(file_path+'wikidata5m_entity.txt', "r") as f:
            entities = f.readlines()
        
        with open(file_path+'wikidata5m_relation.txt', "r") as f:
            relations = f.readlines()

        with open(file_path+'wikidata5m_transductive/wikidata5m_transductive_test.txt', "r") as f:
            triples = f.readlines()

        return triples, entities, relations
    
class Wikidata5mAliasLoader(Wikidata5mLoader):

    def id_to_text(self, data):
        result = {}
        for line in data:
            parts = line.strip().split('\t')
            id_key = parts[0]  # The ID will be the key
            value_text = '\t'.join(parts[1:])
            if id_key in result:
                result[id_key] += f' {value_text}'
            else:
                result[id_key] = value_text
        return result

    def get_data(self):
        triples, entities, relations = super().get_data()

        entities = self.id_to_text(entities)
        relations = self.id_to_text(relations)

        triple_verbal = []
        for i, triple in enumerate(triples):
            try:
                h,r,t = triple.strip().split('\t') 
                triple_verbal.append({"head":entities[h], "relation":relations[r], "tail":entities[t]})
                # if i==10:
                #     break
            except Exception as e:
                print(e)
        return triple_verbal






