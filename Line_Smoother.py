class LineSmoother:

    def __init__(self, data, iterations=1, values=[[20, 20]]):
        self.data = data.copy()
        self.iterations = iterations
        self.values = values

    def smooth_line(self):

        tmp_res = self.data

        for iter in range(self.iterations):

            final_res = []
            back = self.values[iter][0]
            forward = self.values[iter][1]

            for i in range(len(self.data)):
                if i < back:
                    final_res.append((sum(tmp_res[0:i]) + tmp_res[i] + sum(tmp_res[i + 1: i + 1 + forward])) /
                                     (len(tmp_res[0:i]) + len(tmp_res[i + 1: i + 1 + forward]) + 1))
                elif i + forward >= len(tmp_res):
                    final_res.append((sum(tmp_res[i - back:i]) + tmp_res[i] + sum(tmp_res[-len(tmp_res) + i - 1: -1])) /
                                     (len(tmp_res[i - back:i]) + len(tmp_res[-len(tmp_res) + i - 1: -1]) + 1))
                else:
                    final_res.append((sum(tmp_res[i - back:i]) + tmp_res[i] + sum(tmp_res[i + 1: i + 1 + forward])) /
                                     (len(tmp_res[i - back:i]) + len(tmp_res[i + 1: i + 1 + forward]) + 1))

            tmp_res = final_res.copy()

        return final_res

