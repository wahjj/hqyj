# #                 from  n    params  module                                  arguments
# # Import the logging module
# import logging
#
# # Create a logger
# LOGGER = logging.getLogger()
# LOGGER.setLevel(logging.INFO)
#
# # Create a console handler and set the level to info
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
#
# # Create a formatter and set it to the console handler
# formatter = logging.Formatter('%(message)s')
# console_handler.setFormatter(formatter)
#
# # Add the console handler to the logger
# LOGGER.addHandler(console_handler)
#
# # Header
# header = f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}"
# LOGGER.info(header)
#
# # Data rows
# data = [
#     ('ModuleA', 3, 100, 'SomeModuleName', 'SomeArguments'),
#     ('ModuleB', 5, 200, 'AnotherModuleName', 'OtherArguments')
# ]
#
# for row in data:
#     LOGGER.info(f"{row[0]:>18}{row[1]:>3}{row[2]:>10}  {row[3]:<40}{row[4]:<30}")
#
m = "co"
m = eval(m) if isinstance(m, str) else m
