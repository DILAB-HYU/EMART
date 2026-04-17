import yaml

config = {
    "data_dir": {
        "iemocap":      "/data/iemocap/",
        "meld":         "/media/data/public-data/SER/meld/MELD.Raw/",
    },
    "project_dir":      "/media/data/projects/speech-privacy/trust-ser"
}

with open('config.yml', 'w') as outfile:
    yaml.dump(config, outfile, default_flow_style=False)