"""Minimal addict.Dict stub.

ckg08 has no `addict` and its Python is externally managed (PEP 668), so it
cannot be installed. OUTPOST uses exactly one feature of it -- `Dict(cfg)` in
main.py, giving attribute access over a config dict -- so the real package is not
needed to run. Same mechanism as the torch_sparse/torch_scatter stubs already
here; reached via PYTHONPATH=.stubs.

Attribute access on a missing key returns an empty Dict in the real addict rather
than raising, and config lookups rely on that, so it is reproduced.
"""


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
