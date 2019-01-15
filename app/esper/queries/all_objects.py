from esper.prelude import *
from .queries import query

@query("All objects")
def all_objects():
    from query.models import Object
    from esper.stdlib import qs_to_result
    return qs_to_result(Object.objects.all(), stride=1000)
