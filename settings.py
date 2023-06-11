import json
import os
from tinydb import TinyDB, Query

class Settings:
    def __init__(self):
        initialize_data = not os.path.exists("settings.json")
        if initialize_data:
            f = open("settings.json", "a+")
            f.write("{}")
            f.close()
        self.json_file = open("settings.json", "r+")
        self.settings = json.load(self.json_file)
        if initialize_data:
            self.settings["llm"] = "llama"
            self.settings["openai_key"] = ""

    def write_settings(self):
        self.json_file.seek(0)
        self.json_file.write(json.dumps(self.settings))

    def get_llm(self):
        return self.settings["llm"]

    def set_llm(self, value):
        if value not in ["llama", "openai"]:
            return
        else:
            self.settings["llm"] = value
            self.write_settings()

    def get_openai_key(self):
        return self.settings["openai_key"]

    def set_openai_key(self, value):
        self.settings["openai_key"] = value
        self.write_settings()
