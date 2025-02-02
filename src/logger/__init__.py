import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Constants for log configuration
LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024
BACKUP_COUNT = 3

# Construct log file path
log_dir_path = os.path.join(os.getcwd(), LOG_DIR)
os.makedirs(log_dir_path, exist_ok= True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)

def configure_logger():
    """
    This function is reponsible for setting up the logging system. 
    It ensures that logs are recorded in a structure format and stored in both file handler (rotation) and console handler.
    """

    # Creating a customer logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Define the formatter
    formatter = logging.Formatter("[%(asctime)s]%(name)s - %(levelname)s - %(message)s")

    # File handler with rotation
    file_handler = RotatingFileHandler(filename=log_file_path, maxBytes= MAX_LOG_SIZE, backupCount= BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    # Adding handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Configure the logger
configure_logger()