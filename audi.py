
from numpy import log, exp, zeros, NaN
from re import findall


def create_Audi(func):
    def wrapper(*args):
        return Audi(*func(*args))
    return wrapper


class Audi():

    data = None

    def __init__(self, value=None, diff=None, name=None, index=None):
        self.name = name
        self._index = index
        self._value = value
        self.diff = diff


    @property
    def value(self):
        return self._value if self._value is not None else self.data[self._index]


    @create_Audi
    def __neg__(self):
        new_value = -self.value
        new_diff = -self.diff
        return new_value, new_diff


    @create_Audi
    def __add__(self, other):
        if isinstance(other, Audi):
            new_value = self.value + other.value
            new_diff = self.diff + other.diff
        else:
            new_value = self.value + other
            new_diff = self.diff
        return new_value, new_diff


    @create_Audi
    def __sub__(self, other):
        if isinstance(other, Audi):
            new_value = self.value - other.value
            new_diff = self.diff - other.diff
        else:
            new_value = self.value - other
            new_diff = self.diff
        return new_value, new_diff


    @create_Audi
    def __mul__(self, other):
        if isinstance(other, Audi):
            new_value = self.value * other.value
            new_diff = self.diff * other.value + self.value * other.diff
        else:
            new_value = self.value * other
            new_diff = self.diff * other
        return new_value, new_diff


    @create_Audi
    def __pow__(self, other):
        """
        Differentiate self ** other 
        """
        if isinstance(other, Audi):
            # self ** other where diff(self)≠0 and diff(other)≠0
            new_value = self.value ** other.value
            _, new_diff_powf = self._powf(other.value)
            _, new_diff_expf = self._expf(other.value)
            new_diff = new_diff_powf + new_diff_expf
        else:
            # self ** other where diff(self)≠0 and diff(other)=0
            new_value, new_diff = self._expf(other)
        return new_value, new_diff


    def _powf(self, other):
        """
        Differentiate k^x
        """
        new_value = other ** self.value
        new_diff = other ** self.value * log(other) * self.diff
        return new_value, new_diff


    def _expf(self, other):
        """
        Differentiate x^k
        """
        new_value = self.value ** other
        new_diff = other * (self.value ** (other-1)) * self.diff
        return new_value, new_diff


    __rpow__ = create_Audi(_powf)

    __rmul__ = __mul__

    __radd__ = __add__


    def __rsub__(self, other):
        return self.__neg__().__sub__(-other)


    @create_Audi
    def log(self):
        new_value = log(self.value)
        new_diff = 1 / self.value * self.diff;
        return new_value, new_diff


    @create_Audi
    def exp(self):
        new_value = exp(self.value)
        new_diff = new_value * self.diff;
        return new_value, new_diff

    # @staticmethod
    # def populate_namespace(ns, wrt_names, other_names):
        # num_wrt_names = len(wrt_names) 
        # num_other_names = len(other_names) 
        # z = np.zeros(num_wrt_names)
        # for i, n in enumerate(wrt_names):
            # zi = z.copy()
            # ns.__setattr__(n, Audi(i, None, 


def create_Audi_space(all_names: tuple[str], wrt_names: tuple[str], at_values=None) -> dict[str: float]:
    num_all_names = len(all_names)
    num_wrt_names = len(wrt_names)
    if not at_values:
        at_values = dict()

    if set(wrt_names)-set(all_names):
        raise Exception("List of wrt names includes names that are not in the list of all names.")

    space = dict()
    for i, n in enumerate(all_names):
        if n in wrt_names:
            diff = zeros(num_wrt_names, dtype=bool)
            index_wrt = wrt_names.index(n)
            diff[index_wrt] = 1
        else:
            diff = 0
        value = at_values.get(n, NaN)
        space[n] = Audi(value, diff, n, i)

    return space


def extract_all_names_from_expression(expn: str) -> set[str]:
    variable_name_pattern = r"\b[A-Za-z]\w*\b(?!\()"
    return set(findall(variable_name_pattern, expn))


def differentiate(expn: str, wrt_names: set[str], at_values: dict[str: float]) -> tuple[float]:
    all_names = extract_all_names_from_expression(expn)
    audi_space = create_Audi_space(all_names, wrt_names, at_values) 
    result = eval(expn, dict(), audi_space)
    return result.diff


