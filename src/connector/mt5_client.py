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
            logger.warning('Already connected, skip')
            return True
        
        if mt5_creds is None:   # если не передали в словарь - берем из .env
            login = int(os.getenv("MT5_LOGIN"))
            password = os.getenv("MT5_PASSWORD")
            server = os.getenv("MT5_SERVER")   
        else:
            login = mt5_creds["login"]
            password = mt5_creds["password"]
            server = mt5_creds["server"]
        
        if not all([login, password, server]): # если хоть одно пустое 
            logger.error('Have 1 or more empty field MT5 credentials...')
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
            logger.error(f"Eror, enabla to get account info: {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        self._initialized = True
        logger.info(f"Connected to MT5: {self.account_info.login} @ {self.account_info.server}")
        logger.info(f"Balance: {self.account_info.balance} {self.account_info.currency}")
        return True
    
    def is_connected(self) -> bool:
        """Проверка состояния соединения"""
        return self._initialized and mt5.terminal_info() is not None
    
    def disconnect(self):
        """Отключение"""
        if self._initialized:
            mt5.shutdown()
            self.account_info = None
            self._initialized = False
            logger.info('Disconnected from MT5')

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        if not self.account_info:
            return None
        
        return {
            "login" :self.account_info.login ,
            "balance": self.account_info.balance ,
            "equity": self.account_info.equity ,
            "currency": self.account_info.currency
        }




