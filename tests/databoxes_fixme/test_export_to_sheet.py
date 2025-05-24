
import irispie as ir
from irispie.databoxes import _imports2
import sys

d = ir.Databox()
d["a"] = ir.Series(dates=ir.qq(2020,1)>>ir.qq(2024,4), values=range(20))
d["b"] = d["a"] + 10
d["c"] = (d["a"] + 100)[-10]

d["x"] = ir.Series()

d.apply(
    lambda x: ir.aggregate(x, ir.Freq.YEARLY),
    source_names=["a", "b", "c"],
    target_names=lambda n: n + "_yearly",
)

d.apply(
    lambda x: x.redate(ir.ii(0)),
    source_names=["a", "b", "c"],
    target_names=lambda n: n + "_int",
    in_place=True,
)
sys.exit()

info = d.to_sheet("test_export_to_sheet.csv", )
e = ir.Databox.from_sheet("test_export_to_sheet.csv", )


for k, v in d.items():
    assert all(v.get_data(...) == e[k].get_data(...))


d.to_sheet(
    "test_export_to_sheet_monthly.csv",
    #frequency_range={ir.Freq.MONTHLY: ir.mm(2020,1)>>ir.mm(2022,12)},
    frequency_range={ir.Freq.MONTHLY: ...},
)

e1 = ir.Databox.from_sheet("test_export_to_sheet_monthly.csv", )


