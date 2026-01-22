"""
MT5 Client: подключение, проверка соединения, отключение
"""
import os
import logging
from typing import Optional, Dict, Any
import MetaTrader5 as mt5
from dotenv import load_dotenv

# Загружаем .env в переменные окружения
load_dotenv()

logger = logging.getLogger(__name__)


class MT5Client:
    def __init__(self):
        self._initialized = False
        self.account_info = None

    def connect(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Подключение к MT5 терминалу
        
        Args:
            config: {'login': int, 'password': str, 'server': str}
                   если None — берём из .env
            
        Returns:
            True если успешно подключился
        """
        if self._initialized:
            logger.warning("Уже подключён, пропускаем")
            return True

        # Конфиг из .env по умолчанию
        if config is None:
            login = int(os.getenv("MT5_LOGIN"))
            password = os.getenv("MT5_PASSWORD")
            server = os.getenv("MT5_SERVER")
        else:
            login = config["login"]
            password = config["password"]
            server = config["server"]

        if not all([login, password, server]):
            logger.error("Не заданы MT5 credentials в .env или config")
            return False

        # Инициализация MT5
        if not mt5.initialize():
            logger.error(f"mt5.initialize() failed: {mt5.last_error()}")
            return False

        # Логин в счёт
        authorized = mt5.login(login, password=password, server=server)
        if not authorized:
            logger.error(f"mt5.login() failed: {mt5.last_error()}")
            mt5.shutdown()
            return False

        # Проверяем информацию о счёте
        self.account_info = mt5.account_info()
        if self.account_info is None:
            logger.error(f"Не удалось получить info о счёте: {mt5.last_error()}")
            mt5.shutdown()
            return False

        self._initialized = True
        logger.info(f"✅ Подключён к MT5: {self.account_info.login} @ {self.account_info.server}")
        logger.info(f"   Баланс: {self.account_info.balance} {self.account_info.currency}")
        return True

    def is_connected(self) -> bool:
        """Проверка состояния соединения"""
        return self._initialized and mt5.terminal_info() is not None

    def disconnect(self):
        """Отключение"""
        if self._initialized:
            mt5.shutdown()
            self._initialized = False
            logger.info("🔌 Отключён от MT5")

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Инфо о торговом счёте"""
        return {
            "login": self.account_info.login if self.account_info else None,
            "balance": self.account_info.balance if self.account_info else None,
            "equity": self.account_info.equity if self.account_info else None,
            "currency": self.account_info.currency if self.account_info else None,
        } if self.account_info else None
