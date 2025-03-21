from dataclasses import dataclass

@dataclass
class ParamDataclass:
    a: int
    b: float

d = {"a": 1}

obj = ParamDataclass(**d)  # This should work as expected
print(obj)  # Output: ParamDataclass(a=1, b=2.0)