
class ListException(Exception):
    def __init__(self, messages: list[str]) -> None:
        messages = "\n\n" + "\n".join([ "![modiphy] " + m for m in messages ]) + "\n"
        super().__init__(messages) 


class UndeclaredName(Exception):
    def __init__(self, errors: list[tuple[str, str]]) -> None:
        message = "".join([
            f"\n*** Undeclared or mistyped name '{e[0]}' in equation {e[1]}"
            for e in errors
        ])
        message = "\n" + message + "\n"
        super().__init__(message)

class UnknownName(Exception):
    def __init__(self, name):
        super().__init__(f"Name not found in Model object: '{name}'")

