
# Create a function with a similar logic as in bump_version to change the
# edition from "de" to whatever is passed as an argument to the script. The
# edition is encoded in pyproject.toml in the "name" key. Initially, it is
# "irispie-de", and it should be changed to "irispie-<edition>".

import argparse
import toml


def _change_edition_string(current_name: str, new_edition: str) -> str:
    """Change the edition part of the package name"""
    # Split on the last hyphen to separate 'irispie' from the edition
    if '-' in current_name:
        base_name = current_name.rsplit('-', 1)[0]
    else:
        base_name = current_name
    
    return f"{base_name}-{new_edition}"


def main(args):
    with open(args.source_path, "rt") as f:
        toml_content = toml.load(f)
    
    current_name = toml_content["project"]["name"]
    new_name = _change_edition_string(current_name, args.edition)
    toml_content["project"]["name"] = new_name
    
    print(new_name)
    
    if args.target_path:
        with open(args.target_path, "wt") as f:
            f.write(toml.dumps(toml_content))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edition", required=True, help="New edition to set (e.g., 'ce', 'de')")
    parser.add_argument("--source_path", required=True, help="Path to source pyproject.toml file")
    parser.add_argument("--target_path", default=None, help="Path to target pyproject.toml file (optional)")
    args = parser.parse_args()
    main(args)


