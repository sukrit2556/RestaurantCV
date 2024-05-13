# ANSI escape codes for text colors
COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_YELLOW = '\033[93m'
COLOR_BLUE = '\033[94m'
COLOR_PURPLE = '\033[95m'
COLOR_CYAN = '\033[96m'
COLOR_WHITE = '\033[97m'
COLOR_RESET = '\033[0m'  # Reset to default color

# Example usage
print('\033[92m' + 'hello' + '\033[0m')
print(COLOR_GREEN + 'This text will be printed in green.' + COLOR_RESET)
print(COLOR_BLUE + 'This text will be printed in blue.' + COLOR_RESET)
