# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:51:37 2024

@author: kalpavruksh_sjo
"""
from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "<center><h1>Flask app deployment of Azure</h1></center>"

if __name__ == "__main__":
    app.run()
