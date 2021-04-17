# Fake-News Profiling Reporter
This launches a command-line fake-news profiling reporter client, which fetch data from a profiling service and 
presents the data as an HTML report.

## Build and Import
Build with `python setup.py sdist bdist_wheel`.

Import into a local virtual environment with `python setup.py install`.

## Run
The program takes as positional arguments:
* `api_url` - URL of the Fake News Profiling API service (e.g. `http://localhost:8080/user_profiler`)
* `report_dir` - Path to the report directory (where the HTML reports are saved)

Once installed into a local virtual environment, run the server with 
`python -m fnpreporter <api_url> <report_dir>`
