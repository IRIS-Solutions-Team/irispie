
import jinja2 as _jj

e = _jj.Environment()

context = {
    "segments": ["aa", "bb", ],
}

source = """

{% for s in segments %}

    {% set A = f'x_{{s}}' %}

    {{ A }}

{% endfor %}

"""


source = e.from_string(source, ).render(context, )

print(source)

