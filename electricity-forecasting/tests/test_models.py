def test_random_forest_imports():
    from src.models.random_forest import train_random_forest
    assert callable(train_random_forest)
