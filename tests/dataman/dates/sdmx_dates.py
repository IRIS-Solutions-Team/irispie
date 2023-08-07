

import sys
sys.path.append("../../../src")

import irispie as _ip

db = _ip.Databank()
db.date = _ip.dater_from_sdmx_string(_ip.Frequency.QUARTERLY, "2020-Q1")
db.date_string = db.date.to_sdmx_string()

print(db)

