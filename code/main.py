from app.app import App
from app.logger import setup_logger


def main():
    setup_logger()
    app = App()
    app.run()


if __name__ == '__main__':
    main()
