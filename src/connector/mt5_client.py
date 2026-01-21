"""
MT5 Client: подключение, проверка соединения, отключение
"""
import MetaTrader5 as mt5
from dotenv import load_dotenv
import os 
import logging
from typing import Optional, Dict, Any

# Загружаем .env в переменные окружения
load_dotenv()

logger = logging.getLogger(__name__)

class MT5Client:
   
    def __init__(self):
        self._initialized = False       # флаг подключены ли мы к MT5 ?
        self.account_info = None        # кэш информации о счете

    def connect(self, mt5_creds: Optional[Dict[str, Any]] = None) -> bool:
        """
        Connect to MT5

        2 way of getting creds: 1).env 2) directly connect(MT5_LOGIN = , MT5_PASSWORD = ,MT5_SERVER =)
    
        Args:
            mt5_creds: {'login': int, 'password': str, 'server': str}
                   если None — берём из .env
        
        Returns:
            True если успешно подключился
        """
        if self._initialized:
            logger.warnings('Уже подключен, пропускаем')
            return True
        
        if mt5_creds is None:   # если не передали в словарь - берем из .env
            login = int(os.getenv("MT5_LOGIN"))
            password = os.getenv("MT5_PASSWORD")
            server = os.getenv("MT5_SERVER")   
        else:
            login = mt5_creds["MT5_LOGIN"]
            password = mt5_creds["MT5_PASSWORD"]
            server = mt5_creds["MT5_SERVER"]
        
        if not all([login, password, server]): # если хоть одно пустое 
            logger.error('Не заданы MT5 credentials...')
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
        logger.info(f"Подключён к MT5: {self.account_info.login} @ {self.account_info.server}")
        logger.info(f"Баланс: {self.account_info.balance} {self.account_info.currency}")
        return True


