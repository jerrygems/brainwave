class Opr:
    def __init__(self, data):
        self.data = data

    def mean(self):
        columns = input("enter the columns (use \";\" as separator) : ").split(";")
        means = {}
        for col in columns:
            if col not in self.data.columns:
                print(f"There is no column named {col}.")
            else:
                means[col] = self.data[col].mean()
        print(means)

    def median(self):
        columns = input("enter the columns (use \";\" as separator) : ").split(";")
        medians = {}
        for col in columns:
            if col not in self.data.columns:
                print(f"There is no column named {col}.")
            else:
                medians[col] = self.data[col].median()
        print(medians)

    def dataInfo(self):
        print(self.data.info())

    def showNull(self):
        print(self.data.isna().sum())

    def showHead(self):
        print(self.data.head())

    def showTail(self):
        print(self.data.tail())

    def describeCsv(self):
        print(self.data.describe())

    def showColumns(self):
        print(self.data.columns)