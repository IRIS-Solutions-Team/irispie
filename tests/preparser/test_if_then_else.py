
import irispie as ir

source = """

!equations{ :abc :_D }

    !for ?x = 0, 1, 2 !do
        !if aaa == ?x !then
            x = 1;
            !if aaa > 10 !then
                y = 1;
            !end
        !else
            !for ?w = a, b, c !do
                u?w?x = 2;
            !end
        !end
    !end

!equations

    z = 100;

!variables{:main}

    a

"""

p, info = ir.parsers.preparser.from_string(source, context={"aaa": 1}, )
q = ir.parsers.algebraic.from_string(p, )

a, *_ = ir.sources.AlgebraicSource.from_string(source, context={"aaa": 1}, )


