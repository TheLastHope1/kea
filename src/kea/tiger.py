"""Connection helpers for the Tiger Brokers OpenAPI (optional `tigeropen` SDK).

Credentials are never read from Kea's own config files. The SDK picks them up from
either environment variables or the `tiger_openapi_config.properties` file you
download from Tiger's developer portal:

    TIGEROPEN_TIGER_ID     developer app id
    TIGEROPEN_PRIVATE_KEY  RSA private key (PEM text, or a path to the file)
    TIGEROPEN_ACCOUNT      account number (paper accounts are 17+ digits)
    TIGEROPEN_PROPS_PATH   optional: folder or file path of the properties file
"""

from __future__ import annotations

from typing import Any

PAPER_ACCOUNT_DIGITS = 17  # mirrors tigeropen.common.util.account_util


class TigerUnavailable(RuntimeError):
    """The Tiger SDK is missing or credentials are not configured."""


def is_paper_account(account: str | int | None) -> bool:
    text = str(account or "")
    return text.isdigit() and len(text) >= PAPER_ACCOUNT_DIGITS


def client_config() -> Any:
    try:
        from tigeropen.tiger_open_config import TigerOpenClientConfig
    except ImportError as exc:
        raise TigerUnavailable(
            "Tiger support needs the optional SDK: pip install 'kea-trader[tiger]'"
        ) from exc
    config = TigerOpenClientConfig()
    missing = [
        name
        for name, value in (
            ("TIGEROPEN_TIGER_ID", config.tiger_id),
            ("TIGEROPEN_PRIVATE_KEY", config.private_key),
            ("TIGEROPEN_ACCOUNT", config.account),
        )
        if not value
    ]
    if missing:
        raise TigerUnavailable(
            "Tiger credentials not configured; set "
            + ", ".join(missing)
            + " or point TIGEROPEN_PROPS_PATH at tiger_openapi_config.properties"
        )
    return config


def quote_client() -> Any:
    from tigeropen.quote.quote_client import QuoteClient

    return QuoteClient(client_config())


def trade_client() -> Any:
    from tigeropen.trade.trade_client import TradeClient

    return TradeClient(client_config())
