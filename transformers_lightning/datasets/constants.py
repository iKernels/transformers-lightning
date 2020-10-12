import csv

QUOTING_MAP = {
    'none': csv.QUOTE_NONE,
    'non_numeric': csv.QUOTE_NONNUMERIC,
    'minimal': csv.QUOTE_MINIMAL,
    'all': csv.QUOTE_ALL
}