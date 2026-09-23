"""Minimal addict.Dict stub."""


class Dict(dict):
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name not in self:
            self[name] = Dict()
        v = self[name]
        return Dict(v) if isinstance(v, dict) and not isinstance(v, Dict) else v

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]
