
import toml

_PYPROJECT_PATH = "./pyproject.toml"

with open(_PYPROJECT_PATH, "rt") as f:
    toml_content = toml.load(f, )

version = toml_content["project"]["version"]
mmp, edition = version.rsplit("-", maxsplit=1, )
print(mmp)

