"""
Deploy with Azure:
https://learn.microsoft.com/en-us/azure/app-service/quickstart-python
"""

from flask import Flask, render_template_string, request, render_template
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import base64
from io import BytesIO
import numpy as np
import ast
#########
import AutoSys2D

app = Flask(__name__)

@app.route('/')
def index():
    # Default values
    sys_id = "1"
    parameters = str({'a': 0.7, 
                  'b': 0.8,
                  'tau': 12.5,
                  'r': 0.1,
                  'i_ext': 3.2418,})
    
    # If 'a' is provided in the query string, use that value
    if 'sys_id' in request.args:
        sys_id = request.args['sys_id']
    if 'parameters' in request.args:
        parameters = request.args['parameters']
    
    # decode values
    sys_id = int(sys_id)
    params = ast.literal_eval(parameters)
    
    FHN_std_parameters = {'a': 0.7, 
                  'b': 0.8,
                  'tau': 12.5,
                  'r': 0.1,
                  'i_ext': 3.2418,
    }
    
    dynsys = AutoSys2D.FitzHugh_Nagumo(x0=-1.5, y0=-1., parameters=FHN_std_parameters, run_time=500, front_end=False)  
    dynsys.solve()
    animation = dynsys.webapp_animation()
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Matplotlib Animation with Flask</title>
    </head>
    <body>
        <div>
            <!-- Render the Matplotlib animation -->
            {{ animation | safe }}
        </div>
    </body>
    </html>
    ''', 
   animation=animation)

if __name__ == '__main__':
    app.run(debug=False)