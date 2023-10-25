class neighbours:
    def __init__(self, df):
        self.df  = df

    def connect_neighbours(self):
        connections = []
        for i in range(len(self.df)):
            for j in range(len(self.df)):
                if i == j:
                    continue

                connections.append([i, j])
        return connections