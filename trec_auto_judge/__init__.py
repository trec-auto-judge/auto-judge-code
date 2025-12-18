from click import group
from .request import Request, load_requests_from_irds, load_requests_from_file
from ._commands._evaluate import evaluate
from ._commands._export_corpus import export_corpus

__version__ = '0.0.1'


@group()
def main():
    pass


main.command()(evaluate)
main.command()(export_corpus)


if __name__ == '__main__':
    main()
