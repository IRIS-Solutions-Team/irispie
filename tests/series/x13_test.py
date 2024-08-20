
import os
import irispie as ir


_THIS_DIR = os.path.dirname(os.path.abspath(__file__, ), )
_DATA_PATH = os.path.join(_THIS_DIR, "x13_test.csv", )


db = ir.Databox.from_csv(
    _DATA_PATH,
    name_row_transform=lambda x: x.replace("DATE", "__quarterly__"),
    period_from_string=ir.Period.from_iso_string,
)


db.rename(
    source_names=("ND000334Q", ),
    target_names=("GDP_NSA", ),
)


def test_vanilla():
    sa, info = ir.x13(
        db["GDP_NSA"],
        return_info=True,
    )
    assert info["success"]
    assert sa.start == db["GDP_NSA"].start
    assert sa.end == db["GDP_NSA"].end


def test_seasonal_factors():
    sf, info = ir.x13(
        db["GDP_NSA"],
        output="seasonal",
        return_info=True,
    )
    assert info["success"]
    assert sf.start == db["GDP_NSA"].start
    assert sf.end == db["GDP_NSA"].end


def test_in_sample_missing():
    miss = db["GDP_NSA"].copy()
    miss[ir.start+10>>ir.start+15] = None
    sf, info = ir.x13(
        miss,
        allow_missing=True,
        return_info=True,
    )
    assert info["success"]
    assert sf.start == db["GDP_NSA"].start
    assert sf.end == db["GDP_NSA"].end


def test_with_forecast_specs():
    sa, info = ir.x13(
        db["GDP_NSA"],
        return_info=True,
        add_to_specs={
            "pickmdl": {},
            "outlier": {},
            "forecast": {"maxlead": 12, "maxback": 0, },
        },
    )
    assert info["success"]
    assert sa.start == db["GDP_NSA"].start
    assert sa.end == db["GDP_NSA"].end


if __name__ == "__main__":
    # test_vanilla()
    # test_seasonal_factors()
    # test_in_sample_missing()
    test_with_forecast_specs()

