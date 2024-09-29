import json


def json_to_text(json_data):
    text = ""
    for item in json_data:
        for key in item:
            text += f"< {key.upper()} > \n{item[key]}\n"
        text += "\n\n\n\n***********************************************************************************************************************************************\n"
    return text 

def show_report(filename):

    with open(filename) as f:
        json_data = json.load(f)

    text = json_to_text(json_data)
   
    with open(f'{filename}.txt', "w") as f:
        f.write(text)

filename = "REPORT/responses_meta-llama_Meta-Llama-3.1-70B-Instruct_wikidata5m_alias_test.json"
show_report(filename)