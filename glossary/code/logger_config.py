import logging

# Configure logging
logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()  # Logs to the console

# Modes for FileHandler:
# "a" - Append mode (default, adds logs to the end of the file if it exists)
# "w" - Write mode (overwrites the file if it exists)
# "x" - Exclusive creation mode (fails if the file already exists)

file_handler = logging.FileHandler("issues.log", mode="w")

# Set log levels for handlers
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)

# Define a common log format
formatter = logging.Formatter("%(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)
