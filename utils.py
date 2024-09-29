import json
import os


def add_to_json(file, new):
    if os.path.exists(file):
        with open(file) as f:
            old = json.load(f)
        old.append(new)
        with open(file, 'w') as f:
            json.dump(old, f, indent=2)
    else:
        with open(file, 'w') as f:
            json.dump([new], f, indent=2)