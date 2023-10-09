
# Date ranges (`Ranger` class)


## Create date ranges

### Resolved date ranges

Forward ranges

```
r = d1 >> d2
r = Ranger(d1, d2)
r = Ranger(d1, d2, 2)
```

Backward ranges

```
r = d1 << d2
r = Ranger(d2, d1, -1)
r = Ranger(d2, d1, -2)
```

```
list(r)
r[i]
len(r)
```

### Unesolved date ranges

