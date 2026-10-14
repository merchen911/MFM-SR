import pytest

MODELS = ["GRU4Rec", "Caser", "BERT4Rec", "SASRec", "SASRecF", "SASRecD", "FDSA", "SASRecDM_MTL"]


@pytest.mark.parametrize("name", MODELS)
def test_table3_models_are_importable(name):
    from recbole.utils import get_model

    assert get_model(name).__name__ == name


def test_recbole_data_package_and_entry_point_are_importable():
    import recbole.data
    from recbole.quick_start import run_recbole

    assert callable(run_recbole)
    assert recbole.data.__name__ == "recbole.data"
