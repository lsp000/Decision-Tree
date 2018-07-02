# 从指定训练数据中生成决策树并对数据进行预测
1.运行环境：
Python3、
Numpy

2.运行方式:

从终端进入项目文件夹，输入：python3，运行Python解释器，在Python解释器中输入以下命令：

    import decisionTree 

    decisionTree.getArrageError(0,1)

3.参数说明：

    # 指定需要读取的训练数据文件名，应保证该文件与decisionTree.py在同一文件夹下
	self.fileName
	# 指定训练数据使用的分割符号
	self.fileSplitStr
	# 指定训练数据各特征值名称
	self.attribute_names
	# 指定训练数据各特征值类型：continuous表示该特征值为连续型；discrete表示该特征值为离散型
	self.attribute_types
	# 指定训练数据中缺省值的表示符号
	self.unknownMark
	# 指定得到的决策树中，若该节点特征值取值不全时，其它项的表示文字
	self.resultTreeOtherSituationMark = "其它:"
	# 指定用于测试的样本所占比例，默认为0，若为0，则不对树进行剪枝处理
	self.testNumPercent = 0.0
  
4.提供的测试数据中包含连续值、离散值以及缺失值，共32561条训练数据，数据详细信息参考：(http://archive.ics.uci.edu/ml/datasets/Adult)

5.从指定训练数据中生成决策树并预测：

decisionTree.py中testA方法（倒数第二个方法），提供了设置参数的方式，按照指定数据集的数据格式，修改该方法相关内容，设定相应的参数，在Python解释器中运行：
     
     myTree = decisionTree.testA()
     
即可得到生成的决策树对象。
生成决策树对象后，预测单条数据可运行：

     myTree.predictDataWithTree([特征1,特征2,特征3...特征n],myTree.tree)
    
其中myTree.tree为生成的决策树，其它参数参考decisionTree.py中的属性注释

getArrageError（最后一个方法）,提供了利用测试数据集，测试平均错误率的方法

    

