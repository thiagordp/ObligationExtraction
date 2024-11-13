from dotenv import load_dotenv
from src.deontic_extraction import (
    divide_sample,
    obligation_extraction,
    organize_filtered_data,
    setup_logging, OBLIGATIONS_DATA
)
import logging

def main():
    """Main function to load environment, setup logging, and execute data processing steps."""
    try:
        load_dotenv()
        setup_logging()
        logging.info("Environment loaded and logging set up successfully.")
    except Exception as e:
        logging.error("Failed to load environment or set up logging.", exc_info=True)
        return

    try:
        logging.info("Organizing filtered data...")
        organize_filtered_data(OBLIGATIONS_DATA)
        logging.info("Extracting obligations...")
        obligation_extraction()
        logging.info("Dividing sample data...")
        divide_sample()
        logging.info("Process completed successfully.")
    except Exception as e:
        logging.error("An error occurred during the processing steps.", exc_info=True)

if __name__ == '__main__':
    main()