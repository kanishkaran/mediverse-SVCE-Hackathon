from django import template
import json
import ast

register = template.Library()

def parse_list(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value

register.filter('parse_list', parse_list)