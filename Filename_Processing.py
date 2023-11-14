import os

class Data_Processing():
    def __init__(self,data):
        super(Data_Processing, self).__init__()
        self.data = data

    def label_generation(self,data):
        self.data = data
        label_list = []
        label_repeat = []
        for i in range(len(self.data )):
            label = os.path.split(self.data [i])[-1]
            label = label.replace(".png", '')  # 删除尾部文件类型名
            label = label.replace(".jpg", '')
            if "-" in label:
                label = label.split("-")[0]
                label_list.append(label)
                # label_repeat.append(label)
            else:
                label_list.append(label)
        keys = data
        values = label_list
        data_dic = dict(zip(keys, values))
        # return data_dic , label_repeat
        return data_dic