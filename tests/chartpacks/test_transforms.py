
import irispie as ir

db = ir.Databox()
db["x"] = ir.Series(start=ir.qq(2020,1), values=(1.3, 2.4, 4.5, 8.6, 16.7, ), )
db["y"] = ir.Series(start=ir.qq(2020,1), values=(1.3, 1.4, 1.4, 1.5, 1.6, ), )


def test_transform_round_as_lambda():
    ch = ir.Chartpack()
    f = ch.add_figure("Figure", )
    f.add_charts((
        "x",
        "Series y1: y",
    ), transform=lambda x: round(x))
    out = f.plot(db)
    assert _get_data(out, 0) == (1, 2, 4, 9, 17)
    assert _get_data(out, 1) == (1, 1, 1, 2, 2)


def test_transform_round_as_string():
    ch = ir.Chartpack()
    f = ch.add_figure("Figure", )
    f.add_charts((
        "x",
        "Series y2: y",
    ), transform="round(x)")
    out = f.plot(db)
    assert _get_data(out, 0) == (1, 2, 4, 9, 17)
    assert _get_data(out, 1) == (1, 1, 1, 2, 2)


def _get_data(fig, index=0):
    return fig.data[index].y


if __name__ == "__main__":
    test_transform_round_as_lambda()
    test_transform_round_as_string()

