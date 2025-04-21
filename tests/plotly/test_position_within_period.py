
import irispie as ir
import random as rn

# Generate random values
values = tuple(rn.gauss() for _ in range(16))

db = ir.Databox()

# Create a quarterly time series
db["q"] = ir.Series(start=ir.qq(2020,1), values=values, )

# Create a undated time series
db["i"] = ir.Series(start=ir.ii(1), values=values, )

# Plot the quarterly time series with observations and labels in the middle of
# the period
db["q"].plot(
    date_axis_mode="period",
    highlight=(ir.qq(2020,1), ir.qq(2021,2)),
    xline=ir.qq(2020,3),
    show_figure=False,
)

# Plot the quarterly time series with observations and labels at the start of
# the period
db["q"].plot(
    date_axis_mode="instant",
    highlight=(ir.qq(2020,1), ir.qq(2021,2)),
    xline=ir.qq(2020,3),
    show_figure=False,
)

# Plot the undated time series with observations and labels in the middle of
# the period
db["i"].plot(
    date_axis_mode="period",
    highlight=(ir.ii(1), ir.ii(6)),
    xline=ir.ii(3),
    show_figure=False,
)

# Plot the undated time series with observations and labels at the start of
# the period
db["i"].plot(
    date_axis_mode="instant",
    highlight=(ir.ii(1), ir.ii(6)),
    xline=ir.ii(3),
    show_figure=False,
)

