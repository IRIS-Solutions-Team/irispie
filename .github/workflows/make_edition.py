
import argparse
import toml

_PYPROJECT_PATH = "./pyproject.toml"
_EDITION_SEPARATOR = "+"

def _change_edition_in_string(
    current_string: str,
    new_edition: str,
) -> str:
    whatever_before, edition, = current_string.rsplit(_EDITION_SEPARATOR, maxsplit=1, )
    return f"{whatever_before}{_EDITION_SEPARATOR}{new_edition}"

parser = argparse.ArgumentParser()
parser.add_argument("--edition", required=True, help="New edition to set (e.g., 'ce', 'de')")
args = parser.parse_args()

with open(_PYPROJECT_PATH, "rt") as f:
    toml_content = toml.load(f, )

current_version = toml_content["project"]["version"]
new_version = _change_edition_in_string(current_version, args.edition, )
toml_content["project"]["version"] = new_version

current_name = toml_content["project"]["name"]
new_name = _change_edition_in_string(current_name, args.edition, )
toml_content["project"]["name"] = new_name

with open(_PYPROJECT_PATH, "wt") as f:
    f.write(toml.dumps(toml_content, ), )

