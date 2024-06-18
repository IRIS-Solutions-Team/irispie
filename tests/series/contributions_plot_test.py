
import pytest
import irispie as ir
import random as rn


rn.seed(0)


def random_series():
    return ir.Series(
        periods=ir.qq(2020,1,...,2030,4),
        func=rn.gauss,
        num_variants=5,
    )

N = 5

@pytest.mark.parametrize(
    ["index", "x"], [(i, random_series(), ) for i in range(N, ) ]
)
def test_contributions_plot(index: int, x: ir.Series, ) -> None:
    y = ir.series.sum(x)

    figure_title = f"Test chart {index}"

    # Plot bar chart with contributions, bar_relative
    info = x.plot(
        show_figure=False,
        chart_type="bar_relative",
        figure_title=figure_title,
        return_info=True,
        legend=("A", "B", "C", "D", "E", ),
    )

    f = info["figure"]

    # Plot white background for total line
    y.plot(
        figure=f,
        show_figure=False,
        update_traces={"line.width": 6, "line.color": "white"},
    )

    # Plot total line
    y.plot(
        figure=f,
        show_figure=False,
        update_traces={"line.color": "black"},
        legend=["Total"],
    )

    # Move figure title a bit down (default is 0.98 for a reason related to subplots)
    f.update_layout({"title.y": 0.95, })

    assert f.layout.title.text == figure_title
    assert f.layout.title.y == 0.95


