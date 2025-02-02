# # below code is to check logging configuration
# from src.logger import logging

# logging.debug("This is a debug message")
# logging.debug("This is an info message")
# logging.warning("This is an warning message")
# logging.error("This is an error message")
# logging.critical("This is an Critical message")


# Below code is to check exception configuration
# from src.logger import logging
# from src.exception import MyException
# import sys

# try:
#     a = 1+'Z'
# except Exception as e:
#     logging.info(e)
#     raise MyException(e, sys) from e