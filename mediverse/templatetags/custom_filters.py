from django import template
import ast

register = template.Library()

def parse_list(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
def convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return value

register.filter('convert_to_int', convert_to_int)
register.filter('parse_list', parse_list)
