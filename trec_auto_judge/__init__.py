from click import group
from ._commands._evaluate import evaluate


@group()
def main():
    pass


main.command()(evaluate)


if __name__ == '__main__':
    main()
