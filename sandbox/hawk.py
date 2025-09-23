
import re

shocks = "shk_a", "shk_b", "shk_ab", "shk_abc", "shk_abcd",
pattern = re.compile(r"\b(" + "|".join(shocks) + r")\b")

eqs = (
    " no_shk_b + shk_a + shk_ab ",
    " shk_b + shk_ab ",
    " shk_abc + shk_abcd ",
)

replace_dict = {
    shk: f"(ant_{shk}+{shk})"
    for shk in shocks
}

eqs1 = tuple(
    pattern.sub(lambda n: replace_dict[n.group()], i)
    for i in eqs
)

