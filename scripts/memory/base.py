"""Base class for memory providers."""
import abc
from config import AbstractSingleton
import openai
from config import Config
cfg = Config()


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    
    # return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]

    openai.api_type = "azure"
    openai.api_key = cfg.openai_api_key
    # print("aaa")
    # print(cfg.openai_api_base)
    # print(cfg.openai_api_key)
    openai.api_base = cfg.openai_api_base
    openai.api_version = "2022-12-01"
    return openai.Embedding.create(input=[text], engine="text-embedding-ada-002")["data"][0]["embedding"]


class MemoryProviderSingleton(AbstractSingleton):
    @abc.abstractmethod
    def add(self, data):
        pass

    @abc.abstractmethod
    def get(self, data):
        pass

    @abc.abstractmethod
    def clear(self):
        pass

    @abc.abstractmethod
    def get_relevant(self, data, num_relevant=5):
        pass

    @abc.abstractmethod
    def get_stats(self):
        pass
