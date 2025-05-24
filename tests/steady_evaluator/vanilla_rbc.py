
import json as _js
import plotly as _pl
import irispie as _ir


m = _ir.Model.from_file(
    ["model-source/vanilla-rbc.model", "model-source/parameters.model"],
)

n = _ir.Model.from_file(
    ["model-source/vanilla-rbc-stationarized.model", "model-source/parameters.model"],
    flat=True,
)

n.postprocessor = _ir.Sequential.from_file("model-source/vanilla-rbc-stationarized-postprocessor.model", )

parameters = dict(
    alpha = 1.02**(1/4),
    beta = 0.95**(1/4),
    gamma = 0.40,
    delta = 0.05,
    rho = 0.8,
    std_shock_a = 0.9,
    std_shock_c = 0.9,
)

with open("parameters.json", "wt+", ) as fid:
    _js.dump(parameters, fid, indent=4, )

m.assign(**parameters, )
n.assign(**parameters, )

m.assign(a = 1, k = 20, )
n.assign(kk = 20, )

fix_level = ("a", )

info = m.steady(
    fix_level=fix_level,
    flat=False,
    iter_printer_settings={"every": 5, },
)

n.steady(
    iter_printer_settings={"every": 5, },
)

chk, info = m.check_steady(when_fails="warning", )
chk, info = n.check_steady(when_fails="warning", )

m.solve()
n.solve()

start_sim = _ir.qq(2020,1)
end_sim = _ir.qq(2040,4)
sim_range = start_sim >> end_sim


dm = _ir.Databox.steady(m, sim_range, deviation=True, )
dm0 = dm.copy()
dm0["shock_a"][start_sim] = 0.1
sm0, *_ = m.simulate(dm0, sim_range, deviation=True, )

dn = _ir.Databox.steady(n, sim_range, deviation=True, )
dn0 = dn.copy()
dn0["shock_a"][start_sim] = 0.1
sn0, *_ = n.simulate(dn0, sim_range, deviation=True, )

dm = _ir.Databox.steady(m, sim_range, )
dm1 = dm.copy()
dm1["shock_a"][start_sim] = 0.1
sm1, *_ = m.simulate(dm1, sim_range, )

dn = _ir.Databox.steady(n, sim_range, )
dn1 = dn.copy()
dn1["shock_a"][start_sim] = 0.1
sn1, *_ = n.simulate(dn1, sim_range, )

sn1['a'] = _ir.Series()
sn1['a'][start_sim-1] = sm1['a'][start_sim-1]
p = _ir.PlanSimulate(n.postprocessor, start_sim-1>>end_sim, )
p.exogenize(start_sim-1, "a", when_data=True, )
sn1, *_ = n.postprocess(sn1, start_sim-1>>end_sim, target_databox=sn1, plan=p, when_nonfinite="silent", )


chart_names = ("c", "i", "r", "a", )
descriptions = m.get_descriptions()
chart_titles = tuple(descriptions[n] for n in chart_names)


fig = _pl.subplots.make_subplots(
    rows=2, cols=2, shared_xaxes=True,
    subplot_titles=chart_titles,
)

for i, n in enumerate(chart_names):
    (sm1[n] | sn1[n]).plot(
        range=start_sim-1 >> end_sim,
        figure=fig,
        legend=["Full level model", "Stationarized model", ] if i == 0 else None,
        subplot=i,
        traces=({"mode": "lines+markers"}, {"mode": "lines"}, ),
    )

fig.show()


