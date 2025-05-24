
import irispie as ir


db = ir.Databox()
db['aa'] = ir.Series(num_variants=2, dates=ir.qq(2020,1,...,2020,4), values=[(1,2,3,4), 0], )
db['bb'] = 3
db['cc'] = [100, 200]

ds = ir.Dataslate.from_databox(
    db, tuple(db.keys()), ir.qq(2019,1,...,2020,4),
    num_variants=3,
    scalar_names=('bb', 'cc'),
)

db2 = ir.Databox()
db2['dd'] = 'abc'
db2 = ds.to_databox(target_databox=db2, )


