from src.user_settings import DEFAULT_USER_SETTINGS, load_user_settings, save_user_settings


def test_user_settings_round_trip(tmp_path):
    path = tmp_path / "settings.json"

    saved = save_user_settings(
        {
            "threshold": "0.53",
            "tp_pips": 9,
            "sl_pips": 4,
            "include_costs": False,
            "unknown_key": "ignored",
        },
        path=path,
    )
    loaded = load_user_settings(path=path)

    assert saved["threshold"] == 0.53
    assert loaded["threshold"] == 0.53
    assert loaded["tp_threshold"] == 0.0009
    assert loaded["sl_threshold"] == 0.0004
    assert loaded["include_costs"] is False
    assert "unknown_key" not in loaded


def test_user_settings_bad_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{bad json", encoding="utf-8")

    loaded = load_user_settings(path=path)

    assert loaded["threshold"] == DEFAULT_USER_SETTINGS["threshold"]
    assert loaded["initial_balance"] == DEFAULT_USER_SETTINGS["initial_balance"]
