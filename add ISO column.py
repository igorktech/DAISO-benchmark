import os
import json
import pandas as pd

folder_names = ['ami', 'mrda', 'maptask', 'frames', 'swda', 'oasis', 'dyda',
                'dstc3',"dstc8-sgd"]

with open('iso_mapping.json', 'r') as file:
    iso_mapping = json.load(file)

for folder_name in folder_names:
    folder_path = os.path.basename(folder_name)

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        print(f"Files in {folder_name}:")
        current_iso_mapping = iso_mapping[folder_name]

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and os.path.splitext(file_name)[1] in ['.txt', '.csv']:
                print(folder_name, os.path.splitext(file_name)[0])

                if folder_name not in ['ami', 'dyda',"dstc8-sgd"]:
                    df = pd.read_csv(file_path, delimiter="|")
                    if folder_name == 'mrda':
                        df.columns = ['Speaker', 'Utterance', 'Basic_DA', 'General_DA', 'Dialogue_Act']
                    else:
                        df.columns = ['Speaker', 'Utterance', 'Dialogue_Act']
                else:
                    df = pd.read_csv(file_path)


                def map_iso(name):
                    if name is not None:
                        return current_iso_mapping.get(name, {}).get("ISO")
                    else:
                        return None


                df['Dialogue_Act_ISO'] = df['Dialogue_Act'].map(map_iso)
                df.to_csv(os.path.join(folder_path, os.path.splitext(file_name)[0] + '_N.csv'), index=False)
        print()
    else:
        print(f"Folder '{folder_name}' does not exist")