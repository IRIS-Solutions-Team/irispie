"""
General mixins
"""

#[
#]


class DescriptionMixin:
    """
    """
    #[
    _description: str = ""

    def set_description(self, description: str, /, ) -> None:
        self._description = str(description)

    def get_description(self, /, ) -> str:
        return str(self._description)
    #]


