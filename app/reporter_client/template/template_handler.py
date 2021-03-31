import os
from webbrowser import open_new_tab
from jinja2 import Environment, FileSystemLoader


class UserProfilerTemplateHandler:
    """ Compiles data into a Twitter user fake news spreader report """

    def __init__(self):
        template_env = Environment(loader=FileSystemLoader("template/templates"))
        self.template = template_env.get_template("report_template.html")

    def generate_report(self, filename: str, data: dict, open_in_web_browser=True):
        """ Generate a report, given JSON service data """
        full_filename = filename + ".html"
        report = self.template.render(data)
        os.makedirs("reports", exist_ok=True)
        filepath = os.path.join("reports", full_filename)

        with open(filepath, "w", encoding="UTF-8") as report_file:
            print(report, file=report_file)
            print(f"Created report at: {filepath}")
            if open_in_web_browser:
                open_new_tab(filepath)
