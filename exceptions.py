
class ListException(Exception):
    def __init__(self, messages: list[str]) -> None:
        messages = "\n\n" + "\n".join([ "![modiphy] " + m for m in messages ]) + "\n"
        super().__init__(messages) 

