import numpy as np
import cv2
import paddle.fluid as fluid
import paddle
import paddle.vision.transforms as T
from PIL import Image
from Minipic_Model import DNN

class kernel_minipic():
    def __init__(self, rawimg):
        self.CLASS_DIM = 10  # 图像种类数
        self.INIT_LR = 2e-4  # 初始学习率
        self.MODEL_PATH = "dnn.pdparams"  # 模型参数保存路径


    def core(self, rawimg):
        self.rawimg = rawimg

        model = DNN(n_classes=self.CLASS_DIM)
        # opt = Adam(learning_rate=self.INIT_LR,
                   # parameters=model.parameters())  # 定义Adam优化器

        # rawimg = cv2.imread("D:\REC_BD\MXB\MXB_DL\dataset\img_0_0_30.jpg", 1)

        ### 去除图片的黑色边界
        gray = cv2.cvtColor(self.rawimg, cv2.COLOR_RGB2GRAY)  # 转换为灰度图像
        threshold = 40  # 阈值

        nrow = gray.shape[0]  # 获取图片尺寸
        ncol = gray.shape[1]

        rowc = gray[:, int(1 / 2 * nrow)]  # 无法区分黑色区域超过一半的情况
        colc = gray[int(1 / 2 * ncol), :]

        rowflag = np.argwhere(rowc > threshold)
        colflag = np.argwhere(colc > threshold)

        left, bottom, right, top = rowflag[0, 0], colflag[-1, 0], rowflag[-1, 0], colflag[0, 0]

        rawimg_after = gray[left:right, top:bottom]

        ### 数字部分为255 白色部分显示为零
        rawimg = 255 - rawimg_after

        row_nz = []
        for row in rawimg.tolist():
            counter = 0  # 计数器，用于记录每一行中小于阈值的数量
            for i in row:
                if i < threshold:
                    counter = counter + 1
            row_nz.append(len(row) - counter)
        # # counting non-zero value by column
        col_nz = []
        for col in rawimg.T.tolist():
            counter = 0
            for i in col:
                if i < threshold:
                    counter = counter + 1
            col_nz.append(len(col) - counter)
        ## 找x方向的非零边界
        upper_y = 0
        for i, x in enumerate(row_nz):
            # i是索引id，key
            # x是实际的数值value
            if x != 0:
                upper_y = i
                break
        lower_y = 0
        for i, x in enumerate(row_nz[::-1]):
            # row_nz[::-1] 是将row_nz数组倒过来  [::-1] 顺序相反操作
            if x != 0:
                lower_y = len(row_nz) - i
                break
        sliced_y_img = rawimg[upper_y:lower_y, :]
        #
        # print(row_nz[::-1])

        ## 找y方向的非零边界
        column_boundary_list = []
        record = False
        for i, x in enumerate(col_nz[:-1]):
            # col_nz[:-1] 去除数组最后一个数字
            if (col_nz[i] == 0 and col_nz[i + 1] != 0) or col_nz[i] != 0 and col_nz[i + 1] == 0:
                column_boundary_list.append(i + 1)

        img_list = []
        xl = [column_boundary_list[i:i + 2] for i in range(0, len(column_boundary_list), 2)]
        for x in xl:
            img_list.append(sliced_y_img[:, x[0]:x[1]])

        # 图像尺寸重塑
        width = 32
        height = 32

        res = 0
        # 识别图片并且输出结果
        for i, img in enumerate(img_list):
            # new_img = img.resize((width, height), Image.BILINEAR)
            new_img = cv2.resize(img, (width, height), interpolation=Image.BILINEAR)
            path = "D:\REC_BD\MXB\MXB_DL\graphs\{}.jpg".format(i)
            new_img = Image.fromarray(new_img)
            new_img.save(path)

            infer_img = self.load_image(path)  # 3*32*32
            infer_img = fluid.layers.unsqueeze(input=infer_img, axes=[0])
            model.eval()  # 开启评估模式
            model.set_state_dict(
                paddle.load(self.MODEL_PATH)
            )  # 载入预训练模型参数
            result = model(infer_img)
            infer_lab = np.argmax(result)  # 返回数组result中的最大值的索引值

            res = res * 10 + infer_lab

        # print("预测结果：%d" % (res))
        return  res


    def load_image(self, path):  # 图片预处理
        img = cv2.imread(path)  # 32*32*3
        # plt.imshow(img)
        # plt.show()  # 显示图像
        transform = T.ToTensor()
        img = transform(img)  # 将图像转化为Tensor类型
        return img




