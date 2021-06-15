import os
from webbrowser import open_new_tab
from jinja2 import Environment, FileSystemLoader


class UserProfilerTemplateHandler:
    """ Compiles data into a Twitter user fake news spreader report """

    def __init__(self, report_dir: str):
        template_env = Environment(loader=FileSystemLoader("data/templates"))
        self.template = template_env.get_template("report_template.html")
        self.report_dir = report_dir

    def generate_report(self, filename: str, data: dict, open_in_web_browser=True):
        """ Generate a report, given JSON service data """
        full_filename = filename + ".html"
        report = self.template.render(data)
        os.makedirs(self.report_dir, exist_ok=True)
        filepath = os.path.abspath(os.path.join(self.report_dir, full_filename))
        print(filepath)

        with open(filepath, "w", encoding="UTF-8") as report_file:
            print(report, file=report_file)
            print(f"Created report at: {filepath}")
            if open_in_web_browser:
                open_new_tab(filepath)
