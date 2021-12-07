import json


def load_config_from_file():
    with open("config/config.json", "r") as jsonfile:
        config_data = json.load(jsonfile)
        print("Read successful")
    print(config_data)
    return config_data
