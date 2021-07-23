# Rethink priorities job application

To run (pick a name for your virtual environment other than venv)

```$python -m venv projectname
$ source projectname/bin/activate
(venv) $ pip install ipykernel
(venv) $ pip install -r requirements.txt
(venv) $ ipython kernel install --user --name=projectname
(venv) $ jupyter notebook
```

Select the "venv" kernel and run the notebook.