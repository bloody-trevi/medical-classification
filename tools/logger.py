# import logging
# import time
# import os

# class Logger:
#     def __init__(self, exp: str, isStart=False) -> None:
#         if isStart:
#             log_filemode = "w"
#         else:
#             log_filemode = "a"

#         # path formatting
#         root_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "log")
#         now = time.localtime()
#         log_fname = os.path.join(root_path, exp + time.strftime('_%Y%m%d-%H%M%S', now) + ".log")

#         # init logging config
#         logging.basicConfig(filename=log_fname, filemode=log_filemode, level=logging.INFO, datefmt="%m/%d/%Y %I:%M:%S %p")

#     def logging(log_fname, epoch, loss, accuracy) -> None:
#         logging.info("")
