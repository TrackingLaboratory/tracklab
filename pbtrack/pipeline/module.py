import re
from abc import ABC, ABCMeta
from typing import List

import logging

log = logging.getLogger(__name__)


class MetaModule(ABCMeta):
    @property
    def name(cls):
        name = cls.__name__
        return name  # re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

    @property
    def level(cls):
        name = cls.__bases__[0].__name__
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
        return name.split("_")[0]


class Module(metaclass=ABCMeta):
    input_columns = None
    output_columns = None
    forget_columns = []

    @property
    def name(self):
        name = self.__class__.__name__
        return name  # re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

    @property
    def level(self):
        name = self.__class__.__bases__[0].__name__
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
        return name.split("_")[0]

    def validate_input(self, dataframe):
        assert self.input_columns is not None, "Every model should define its inputs"
        for col in self.input_columns:
            if col not in dataframe.columns:
                raise AttributeError(f"The input detection should contain {col}.")

    def validate_output(self, dataframe):
        assert self.output_columns is not None, "Every model should define its outputs"
        for col in self.output_columns:
            if col not in dataframe.columns:
                raise AttributeError(f"The output detection should contain {col}.")


class Pipeline:
    def __init__(self, models: List[Module]):
        self.models = [model for model in models if model.name != "skip"]
        log.info("Pipeline: " + " -> ".join(model.name for model in self.models))
        self.validate()

    def validate(self):
        columns = set()
        for model in self.models:
            if model.input_columns is None or model.output_columns is None:
                raise AttributeError(
                    f"{type(model)} should contain input_ and output_columns"
                )
            if not set(model.input_columns).issubset(columns):
                raise AttributeError(
                    f"The {model} model doesn't have "
                    "all the input needed, "
                    f"needed {model.input_columns}, provided {columns}"
                )
            columns.update(model.output_columns)

    def __str__(self):
        return " -> ".join(model.name for model in self.models)

    def __getitem__(self, item: int):
        return self.models[item]


class Skip(Module):
    def __init__(self, **kwargs):
        pass

    @property
    def name(self):
        return "skip"
