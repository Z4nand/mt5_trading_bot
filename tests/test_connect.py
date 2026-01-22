import logging
from pathlib import Path
from src.connector.mt5_client import MT5Client

# Настройка логов
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),  # консоль
        logging.FileHandler("logs/test_mt5.log")  # файл
    ]
)
logger = logging.getLogger(__name__)

def main():
    # Создаём папку logs если нет
    Path("logs").mkdir(exist_ok=True)
    
    logger.info("Start test MT5Client...")
    
    # 1. Создаём клиент
    client = MT5Client()
    
    # 2. Подключаемся (из .env)
    logger.info("1. Test connect()...")
    if client.connect():
        logger.info("connect() SUCCESS")
    else:
        logger.error("connect() FAILED")
        return
    
    # 3. Проверяем статус
    logger.info("2. Test is_connected()...")
    status = client.is_connected()
    logger.info(f"   Status: {status}")
    
    # 4. Инфо о счёте
    logger.info("3. Test get_account_info()...")
    account_info = client.get_account_info()
    if account_info:
        logger.info(f"   Login: {account_info['login']}")
        logger.info(f"   Balance: {account_info['balance']}")
        logger.info(f"   Equity: {account_info['equity']}")
        logger.info(f"   Currency: {account_info['currency']}")
    else:
        logger.error("No account information")
    
    # 5. Пробуем повторное подключение
    logger.info("4. Retest connect()...")
    if client.connect():
        logger.info("Reconnect..")
    else:
        logger.error("Reconnect error")
    
    # 6. Отключаемся
    logger.info("5. Test disconnect()...")
    client.disconnect()
    
    # 7. Финальная проверка
    logger.info("6. Final status...")
    final_status = client.is_connected()
    logger.info(f"is_connected(): {final_status} (must be False)")
    
    logger.info("Test comleted!")

if __name__ == "__main__":
    main()