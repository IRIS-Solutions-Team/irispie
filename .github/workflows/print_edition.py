
import toml

_PYPROJECT_PATH = "./pyproject.toml"
_EDITION_SEPARATOR = "+"

with open(_PYPROJECT_PATH, "rt") as f:
    toml_content = toml.load(f, )

version = toml_content["project"]["version"]
mmp, edition = version.rsplit(_EDITION_SEPARATOR, maxsplit=1, )
print(edition)

