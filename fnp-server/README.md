# Fake-News Profiling Server
This launches a local server which runs various fake-news profiling services, providing an interface to them.

## Build and Import
Build with `python setup.py sdist bdist_wheel`.

Import into a local virtual environment with `python setup.py install`.

## Run
The program takes as positional arguments:
* `config` - Filepath to the services configuration file
* `server_port` - The server's port (defaults to 8080)

Once installed into a local virtual environment, run the server with 
`python -m fnpserver <config> <server_port>`

### Config
An example config JSON file can be found in `fnp-server/data/example_service_config.json`.

This file contains a list of services to run. Each service should provide: 
* `import_path` - The pythonic import path used to import the service's Python class, this is passed to `importlib` 
so the service is dynamically imported at run-time - allowing for custom service classes.
* `endpoint` - The service's API endpoint
* `config` - The service's config
