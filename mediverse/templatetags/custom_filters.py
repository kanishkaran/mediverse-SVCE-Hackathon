from django import template
import json
import ast

register = template.Library()

def check_type(value):
    if isinstance(value, list):
        return 'list'
    else:
        return 'string'
    
def json_decode(value):
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value

def parse_list(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value
    
register.filter('check_type', check_type)
register.filter('json_decode', json_decode)
register.filter('parse_list', parse_list)