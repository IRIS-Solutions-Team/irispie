
class Model:
    def __init__(self, context, ):
        self.context = context
        self.func_string = "def func(a, b, ): return a * cdf(a, b)"
        self._compile_function()

    def _compile_function(self, ):
        context = {"__builtins__": None, } | self.context
        exec(self.func_string, context, )
        self.func = context["func"]

    def eval(self, a, b, ):
        return self.func(a, b, )

    def __getstate__(self, ):
        return {"context": self.context, "func_string": self.func_string, }

    def __setstate__(self, state):
        self.context = state["context"]
        self.func_string = state["func_string"]
        self._compile_function()
