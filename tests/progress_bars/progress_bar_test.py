
import irispie as ir
import pytest
import os


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def test_progress_bar():
    log_file = os.path.join(_THIS_DIR, "progress_bar_test.log")
    with open(log_file, "wt", ) as fid:
        p = ir.ProgressBar(num_steps=10, output_stream=fid, )
        p.start()
        for i in range(10):
            p.increment()
        p.finish()
    with open(log_file, "rt") as fid:
        log = fid.read()
    assert log[-1] == "\n"
    assert log[-2] == "s"

