
import irispie as ir
import irispie.ez_plotly as ez
import random as rn
import plotly.graph_objects as pg

a = [rn.uniform(1, 2) for _ in range(10)]
b = [rn.uniform(1, 2) for _ in range(10)]

# c = ez.Chartbox(subplots=(2, 2))
# # c.update_layout(colorway=["xadsfa", "blue", "green", "orange", "purple", "brown", "pink", "gray", "cyan", "magenta"])
# 
# # c.add_bar(subplot=(0, 0), y=a, name="A", mode="relative");
# c.add_bar(subplot=(0, 0), y=b, name="B", marker=dict(coloraxis="coloraxis1", ), );
# 
# c.add_bar(subplot=(0, 1), y=a, name="A", marker=dict(coloraxis="coloraxis2", ), );
# #c.add_bar(subplot=(0, 1), y=b, name="B", marker=dict(coloraxis="coloraxis2", ), );
# c.show()

fig1 = pg.Figure()
fig1.add_scatter(y=a, mode="lines");

fig2 = pg.Figure()
fig2.add_scatter(y=a, mode="lines");




for i in range(len(fig1.data)):
    fig1.data[i].xaxis='x1'
    fig1.data[i].yaxis='y1'

fig1.layout.xaxis1.update({'anchor': 'y1'})
fig1.layout.yaxis1.update({'anchor': 'x1', 'domain': [.55, 1]})

for i in range(len(fig2.data)):
    fig2.data[i].xaxis='x2'
    fig2.data[i].yaxis='y2'

# initialize xaxis2 and yaxis2
fig2['layout']['xaxis2'] = {}
fig2['layout']['yaxis2'] = {}


fig2.layout.xaxis2.update({'anchor': 'y2'})
fig2.layout.yaxis2.update({'anchor': 'x2', 'domain': [0, .45]})


fig = go.Figure()
fig.add_traces([fig1.data[0], fig2.data[0]])

fig2.show()

fig.layout.update(fig1.layout)
fig.layout.update(fig2.layout)

fig.show()

