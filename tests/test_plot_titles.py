from mdsview.plotting import format_diff_title, format_dod_title, format_field_title


def test_format_field_title_3d():
    assert format_field_title("T", 480, level=4, shape=(40, 160, 240)) == "T, iter 480, level 4"


def test_format_field_title_2d():
    assert format_field_title("Eta", 120, shape=(160, 240)) == "Eta, iter 120"


def test_format_field_title_default_level():
    assert format_field_title("T", 0, shape=(12, 60, 80)) == "T, iter 0, level 6"


def test_format_diff_title():
    assert (
        format_diff_title("T", 2520, 0, level=20, shape=(40, 160, 240))
        == "T, iter 2520 - 0, level 20"
    )


def test_format_diff_title_cross_run():
    title = format_diff_title(
        "T", 2520, 0, level=20, shape=(40, 160, 240),
        later_tag="@warm", earlier_tag="@ref",
    )
    assert title == "T, iter 2520@warm - 0@ref, level 20"


def test_format_dod_title():
    assert (
        format_dod_title("T", "S", 0, 2520, level=20, shape=(40, 160, 240))
        == "(S - T), t1=0, t2=2520, level 20"
    )
