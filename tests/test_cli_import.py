def test_cli_imports():
    # Smoke test to ensure CLI module imports and exposes main
    import FESutils.cli as cli

    assert callable(cli.main)
