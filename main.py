import sys
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtCore import *
from GUI_main import Ui_MainWindow
from Filename_Processing import Data_Processing
from Main_Core import Main_core



class MyMainForm(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MyMainForm, self).__init__(parent)

        self.setupUi(self)
        # self.pushButton.clicked.connect(self.Thread_openfile) # 按钮1绑定线程的触发事件
        # self.pushButton_2.clicked.connect(self.Thread_choose_output_path)
        # input_items = self.openfile
        # output_path = self.choose_output_path
        # global input_items
        # global output_path
        self.pushButton_3.clicked.connect(lambda :self.image_recognition(self.openfile(),self.choose_output_path()))


    def openfile(self):
        openfile_name = QFileDialog.getOpenFileNames(self,'选择文件','','PNG Files(*.png))')

        self.textBrowser.setText(str(openfile_name[0]))

        ###### 创建表格项
        items = openfile_name[0]
        for i in range(len(items)):
            row = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row)
            # 设置列名
            self.tableWidget.setColumnCount(1)
            self.tableWidget.setHorizontalHeaderLabels(['需转换文件路径'])
            # 表格铺满整个QTableWidget控件
            self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            # 设置双击单元格不可编辑
            self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
            input_str = str(items[i])
            item = QTableWidgetItem(input_str)
            self.tableWidget.setItem(i,0,item)

        input_items = openfile_name[0]
        return input_items # input_items数据类型是{list},里面的每一项是str


    def choose_output_path(self):
        openfile_name = QFileDialog.getExistingDirectory(None,"选取文件夹","D:/REC_BD/MXB/2")
        output_path = openfile_name
        self.textBrowser_2.setText(str(output_path))
        return output_path # output_path数据类型是str


    def image_recognition(self,input_items,output_path):
        DP = Data_Processing(input_items)
        MC = Main_core()
        data_dic = DP.label_generation(input_items)
        label_not_repeat = list(data_dic.values())
        value = []
        [value.append(i) for i in label_not_repeat if not i in value]

        for i in value:
            df = []
            df = pd.DataFrame(df)
            for keys, values in data_dic.items():
                if values == i:
                    # 识别的程序，出excel的程序
                    df_1 = MC.parse_pic_to_excel_data(keys)
                    df = pd.concat( [df,df_1] , axis = 0 ).reset_index(drop=True)
                    df_1 = []
                    df_1 = pd.DataFrame(df_1)
            MC.make_excel(output_path,df,i)

        return



if __name__ == "__main__":
    # 固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    # 初始化
    myWin = MyMainForm()
    # 将窗口控件显示在屏幕上
    myWin.show()
    #程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())




