from esper.prelude import *
#from esper.spark import *
import django.apps

models = [m._meta.db_table for m in django.apps.apps.get_models(include_auto_created=True)]

with Timer('Exporting models'):
    def export_model(model):
        try:
            sp.check_call("/app/scripts/export-table.sh {}".format(model), shell=True)
        except Exception:
            import traceback
            print(model)
            traceback.print_exc()
    par_for(export_model, models, workers=8)
