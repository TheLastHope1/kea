"""Brokers: the local paper account and Tiger Brokers."""

from __future__ import annotations

from kea.brokers.base import Account, Broker, OrderTicket
from kea.brokers.paper import PaperBroker
from kea.config import Config


def build_broker(config: Config) -> Broker:
    if config.broker.kind == "tiger":
        from kea.brokers.tiger import TigerBroker

        return TigerBroker(allow_live=config.broker.allow_live)
    return PaperBroker(
        config.broker.state_dir / "paper_account.json",
        config.broker.initial_cash,
        config.execution,
    )


__all__ = ["Account", "Broker", "OrderTicket", "PaperBroker", "build_broker"]
