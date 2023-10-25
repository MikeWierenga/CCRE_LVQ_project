import csv
import os

class CSV:
    def __init__(self, name):
        self.name = name
        self.file = self.create_csvfile()

    def create_csvfile(self):
        return open(self.name, 'w')

    def write_to_csv_file(self, text_as_list):
        writer = csv.writer(self.file)
        if type(text_as_list) == list:

            writer.writerow(text_as_list)
        else:
            print(f'please enter a list of item(s)')


