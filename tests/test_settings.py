from comfytelegram.settings import Settings


def _settings(**overrides) -> Settings:
    # _env_file=None: don't let the real repo-root .env (if present) leak
    # into these tests — Settings should only reflect what's passed here.
    return Settings(_env_file=None, telegram_bot_token="test-token", **overrides)


def test_allowed_user_ids_parses_comma_separated_string():
    assert _settings(allowed_user_ids="123,456").allowed_user_ids == [123, 456]


def test_allowed_user_ids_drops_empty_segments():
    # a trailing/doubled comma shouldn't produce a bogus empty-string entry
    assert _settings(allowed_user_ids="123,,456,").allowed_user_ids == [123, 456]


def test_allowed_user_ids_empty_string_is_empty_list():
    assert _settings(allowed_user_ids="").allowed_user_ids == []


def test_allowed_user_ids_defaults_to_empty_list():
    assert _settings().allowed_user_ids == []


def test_comfyui_http_base_uses_http_by_default():
    settings = _settings(comfyui_host="example.com", comfyui_port=1234)
    assert settings.comfyui_http_base == "http://example.com:1234"


def test_comfyui_http_base_uses_https_when_tls_enabled():
    settings = _settings(comfyui_host="example.com", comfyui_port=1234, comfyui_use_tls=True)
    assert settings.comfyui_http_base == "https://example.com:1234"


def test_comfyui_ws_base_uses_ws_by_default():
    settings = _settings(comfyui_host="example.com", comfyui_port=1234)
    assert settings.comfyui_ws_base == "ws://example.com:1234"


def test_comfyui_ws_base_uses_wss_when_tls_enabled():
    settings = _settings(comfyui_host="example.com", comfyui_port=1234, comfyui_use_tls=True)
    assert settings.comfyui_ws_base == "wss://example.com:1234"
