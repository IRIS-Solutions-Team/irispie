
import irispie as ir


db = ir.Databox()
db["a"] = 1
db["b"] = 2
db["c"] = "abc"
db["A"] = None
db["B"] = [10, 20, 30, ]


def test_has_single_name():
    assert db.has("a")
    assert db.has("b")
    assert db.has("c")
    assert db.has("A")
    assert db.has("B")
    assert not db.has("d")
    assert not db.has("C")
    assert not db.has("D")


def test_has_tuple_of_names():
    assert db.has(("a", "b"))
    assert db.has(("b", "c"))
    assert db.has(("c", "A"))
    assert db.has(("A", "B"))
    assert not db.has(("a", "C"))
    assert not db.has(("c", "D"))
    assert not db.has(("D", "C"))


def test_shallow_copy():
    sh = db.shallow(source_names=("B", ))
    sh["B"].append(40)
    assert db["B"] == [10, 20, 30, 40, ]


