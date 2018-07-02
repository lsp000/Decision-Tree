from numpy import *
import random

class Tree():
	"""docstring for Tree"""
	def __init__(self):
		# 数据文件名
		self.fileName = ""
		# 缺失值标记
		self.unknownMark = ""
		# 生成决策树中，其它值的表示符号
		self.resultTreeOtherSituationMark = "Others:"
		# 文件数据分割符
		self.fileSplitStr = ""
		# 训练集
		self.trainSet = []
		# 测试集（用于树剪枝）
		self.testSet = []
		# 属性名称
		self.attribute_names = []
		# 属性数据类型'continuous','discrete'
		self.attribute_types = []
		# 离散型属性可能取值{属性名：[可能取值1,可能取值2,可能取值3...]}
		self.attributes_dicSet = {}
		# 测试数据所占比例
		self.testNumPercent = 0.0
		# 属性索引，用于预测
		self.predictAttrIndexDic = {}


		self.canPrint = True
		# 生成的决策树
		self.tree = {}


	# 从指定文件中读取文件，参数：文件名，分割符
	def readFile(self,fileName,splitStr):
		self.fileName = fileName
		self.fileSplitStr = splitStr
		dataSet = []
		attr_length = len(self.attribute_names) + 2
		fr = open(fileName)
		index = 0
		for line in fr.readlines():
			newLine = line.strip().split(splitStr)
			if len(newLine) == 1:
				# 如果分割后只有一条数据，则提示分割符是否正确，并退出
				print("分割符不正确！")
				# 文件名、分割符置空
				self.fileName = ""
				self.fileSplitStr = ""
				break
			index += 1
			attributes = [item.strip() for item in newLine]
			# 添加数据权重
			attributes.append(float(1))
			if len(attributes) != attr_length:
				print("第",index,"条数据格式可能不正确")
				continue
			dataSet.append(attributes)
		return dataSet
		pass
	# 训练决策树
	def trainTree(self):
		if not self.checkPropoty():
			return
		dataSet = self.readFile(self.fileName,self.fileSplitStr)
		self.trainSet,self.testSet = self.create_train_test_DataSet(dataSet)
		self.create_attributes_dicSet(mat(self.trainSet))
		self.create_attributes_indexDic()
		labelSet = self.attribute_names[:]
		labelTypes = self.attribute_types[:]
		tree = self.getTree(self.trainSet,labelSet,labelTypes)
		self.post_Prune(tree,self.testSet)
		return tree
		pass

	def post_Prune(self,tree,dataSet):
		key = list(tree.keys())[0]
		index = self.predictAttrIndexDic[key]
		attr_type = self.attribute_types[index]
		isContinuous = (attr_type == "continuous")
		treeSplitKeys = list(tree[key].keys())
		if isContinuous:
			sub_key = treeSplitKeys[0]
			if "<=" in sub_key:
				attr_value = float(sub_key.split("<=")[-1])
			else:
				attr_value = float(sub_key.split(">")[-1])
			splitDic = self.splitDataWithContinuouAttr(dataSet,attr_value,index,key,False)
		else:
			splitDic,w_d,v_w,t_w = self.splitInfoWithDiscreteAttr(dataSet,index,True,False)	
		
		for key_item in splitDic.keys():
			if key_item in treeSplitKeys:
				if self.isTree(tree[key][key_item]) and self.isList(splitDic[key_item]):
					tree[key][key_item] = self.post_Prune(tree[key][key_item],splitDic[key_item])
		for key_item in tree[key].keys():
			if self.isTree(tree[key][key_item]) and key_item != self.unknownMark:
				return tree
		errorNum = 0
		for item in dataSet:
			copyItem = item[:]
			del(copyItem[-1])
			result = self.predictDataWithTree(copyItem,tree)
			if result != item[-2]:
				errorNum += 1
		prune_errorNum = 0
		prune_value = self.chooseTheMostItems(dataSet)
		for item in dataSet:
			if prune_value != item[-2]:
				prune_errorNum += 1
		if prune_errorNum < errorNum:
			print("prune:",tree)
			return prune_value
		else:
			return tree


	# 取出一部分训练样本用于树剪枝
	def create_train_test_DataSet(self,dataSet):
		totalNum = len(dataSet)
		dataIndexDic = {}
		for index in range(totalNum):
			data = dataSet[index]
			dataIndexN = dataIndexDic.get(data[-2],[])
			dataIndexN.append(index)
			dataIndexDic[data[-2]] = dataIndexN
		testIndexList = []
		for key in dataIndexDic.keys():
			indexList = dataIndexDic[key]
			dataNum = int(len(indexList) * self.testNumPercent + 0.5)
			randomList = random.sample(indexList,dataNum)
			testIndexList.extend(randomList)
		testIndexList.sort()
		t_index = 0
		trainSet = []
		testSet = []
		testNum = len(testIndexList)
		for index in range(totalNum):
			copyData = dataSet[index]
			if (t_index == testNum) or (index < testIndexList[t_index]):
				trainSet.append(copyData)
			else:
				testSet.append(copyData)
				print("test_index:",index)
				t_index += 1

		return trainSet,testSet


	# 从指定数据生成树
	def getTree(self,dataSet,labelSet,labelTypes):
		print("~~~~~~~~~~~~~~~~new~~~~~~~~~~~~~~~~~~")
		classList = [item[-2] for item in dataSet]
		# 如果属性都划分完了
		if len(dataSet[0])==2:
			return self.chooseTheMostItems(dataSet)
		# 如果全都为同一分类
		if len(set(classList)) == 1:
			return classList[0]
		bestSplitAttributeIndex,bestSplitValue = self.chooseBestAttributeToSplit(dataSet,labelTypes,labelSet)
		# 没有符合要求的最佳划分
		if bestSplitAttributeIndex == -1:
			return self.chooseTheMostItems(dataSet)
		isContinuous = (labelTypes[bestSplitAttributeIndex] == "continuous")
		bestSplitAttr = labelSet[bestSplitAttributeIndex]
		tree = {bestSplitAttr:{}}
		if isContinuous:
			splitDic = self.splitDataWithContinuouAttr(dataSet,bestSplitValue,bestSplitAttributeIndex,bestSplitAttr)
		else:
			splitDic = self.splitDataWithDiscreteAttr(dataSet,bestSplitAttributeIndex)
		del(labelTypes[bestSplitAttributeIndex])
		del(labelSet[bestSplitAttributeIndex])
		totalNum = float(0)
		splitCountDic = {}
		for key in splitDic.keys():
			copyLabelTypes = labelTypes[:]
			copyLabelSet = labelSet[:]
			tree[bestSplitAttr][key] = self.getTree(splitDic[key],copyLabelSet,copyLabelTypes)
			classifyNum = len(splitDic[key])
			totalNum += classifyNum
			splitCountDic[key] = float(classifyNum)
			pass
		# 补全属性缺失值
		if not isContinuous:
			attrs = self.attributes_dicSet[bestSplitAttr][:]
			keys = tree[bestSplitAttr].keys()
			if len(attrs) > len(keys):
				tree[bestSplitAttr][self.resultTreeOtherSituationMark] = self.chooseTheMostItems(dataSet)
		unknownRadioDic = {}
		for key in splitCountDic.keys():
			unknownRadioDic[key] = splitCountDic[key]/totalNum
			pass
		tree[bestSplitAttr][self.unknownMark] = unknownRadioDic
		return tree
		pass

	# 选择最佳属性切分数据
	def chooseBestAttributeToSplit(self,dataSet,labelTypes,labelSet):
		attributeNum = len(labelTypes)
		maxGain = 0
		mostUsedKeyNum = 0
		bestSplitAttributeIndex = -1
		bestSplitValue = -1
		for attribute_index in range(attributeNum):
			isContinuous = (labelTypes[attribute_index] == "continuous")
			gain,usedKeyNum,splitValue = self.ID3_CalculateGain(dataSet,attribute_index,isContinuous)
			print(attribute_index,labelSet[attribute_index],",","gain:",gain,"splitValue:",splitValue)
			if (gain > maxGain) or (gain == maxGain and usedKeyNum > mostUsedKeyNum):
				maxGain = gain
				bestSplitAttributeIndex = attribute_index
				bestSplitValue = splitValue
				if ((gain == maxGain) and (not isContinuous)):
					mostUsedKeyNum = usedKeyNum
				pass
		return bestSplitAttributeIndex,bestSplitValue
		pass

	# 计算信息增益ID3,返回增益值以及该属性索引下属性数量
	def ID3_CalculateGain(self,baseDataSet,index,isContinuous):
		# 使用到的属性数量
		usedKeyNum = 0
		bestSplitValue = -1
		if isContinuous:
			bestSplitValue,validWeight,totalWeight,baseEnt,gain = self.bestBinSplitInfoWithContinuouAttr(baseDataSet,index)
		else:
			splitDic,weightDic,validWeight,totalWeight = self.splitInfoWithDiscreteAttr(baseDataSet,index)
			validDataSet = []
			gain = 0
			for key in splitDic.keys():
				usedKeyNum += 1
				validDataSet.extend(splitDic[key])
				ent = self.calculateEnt(splitDic[key])
				gain += (weightDic[key]/validWeight)*ent
			baseEnt = self.calculateEnt(validDataSet)
		gain = baseEnt - gain
		gain *= (validWeight/totalWeight)
		return gain,usedKeyNum,bestSplitValue

	# 离散型数据划分信息,delLost==True,删除缺失项，delLost==Flase,将缺失项保留，并划分入每一类，修改权值
	def splitInfoWithDiscreteAttr(self,dataSet,index,delLost=True,delSpecifyData=True):
		returnDic = {}
		lostArray = []
		# 权重计数
		weightDic = {}
		# 总有效权重
		validWeight = float(0)
		totalWeight = float(0)
		for data in dataSet:
			copyData = data[:]
			attribute = copyData[index]
			if delSpecifyData:
				del(copyData[index])
			totalWeight += copyData[-1]
			if attribute != self.unknownMark:
				attributeArray = returnDic.get(attribute,[])
				attributeArray.append(copyData)
				returnDic[attribute] = attributeArray
				weightDic[attribute] = weightDic.get(attribute,float(0)) + copyData[-1]
				validWeight += copyData[-1]
			else:
				lostArray.append(copyData)
		if not delLost:
			for lost in lostArray:
				for key in returnDic.keys():
					copyLost = lost[:]
					copyLost[-1] *= (weightDic[key]/validWeight)
					returnDic[key].append(copyLost)
		return returnDic,weightDic,validWeight,totalWeight
		pass
	# 连续型数据划分信息,二分法
	def bestBinSplitInfoWithContinuouAttr(self,dataSet,index):
		values = []
		validWeight = float(0)
		totalWeight = float(0)
		validDataSet = []
		for data in dataSet:
			copyData = data[:]
			value = copyData[index]
			totalWeight += copyData[-1]
			if value != self.unknownMark:
				values.append(value)
				validWeight += copyData[-1]
				validDataSet.append(copyData)
		values = sort([float(item) for item in set(values)])
		maxGain = 100
		bestSplitValue = -1
		baseEnt = self.calculateEnt(validDataSet)
		if len(values) == 1:
			bestSplitValue = values[0]
			maxGain = baseEnt 
		else:
			leftValidWeight = 0 ; rightValidWeight = validWeight
			leftDataSet = [] ; rightDataSet = sorted(validDataSet,key=lambda x:x[index])
			leftClassifyCountDic = {} ; rightClassifyCountDic = {}
			for data in rightDataSet:
				leftClassifyCountDic[data[-2]] = float(0)
				rightClassifyCountDic[data[-2]] = rightClassifyCountDic.get(data[-2],float(0)) + data[-1]
				pass
			for index_ in range(len(values)-1):
				splitValue = (values[index_] + values[index_+1])/2
				# print("bestBinSplitInfoWithContinuouAttr:",index_,",,",len(values))
				leftEnt,rightEnt,leftValidWeight,rightValidWeight = self.scanSortedContinuouAttr(\
					rightDataSet,splitValue,index,leftValidWeight,rightValidWeight,leftClassifyCountDic,rightClassifyCountDic)
				

				gain = leftEnt*leftValidWeight/validWeight + rightEnt*rightValidWeight/validWeight
				if gain <= maxGain:
					bestSplitValue = splitValue
					maxGain = gain
					pass
		return bestSplitValue,validWeight,totalWeight,baseEnt,maxGain	
		pass

	# 离散型数据划分
	def splitDataWithDiscreteAttr(self,dataSet,index):
		returnDic,weightDic,validWeight,totalWeight = self.splitInfoWithDiscreteAttr(dataSet,index,False)
		return returnDic
		pass

	# 扫描数据集，计算信息熵
	def scanSortedContinuouAttr(self,dataSet,specifyValue,attrIndex,leftWeight,rightWeight,\
		leftClassifyCountDic,rightClassifyCountDic):
		lessCount = 0
		for data in dataSet:
			value = float(data[attrIndex])
			if value <= specifyValue:
				lessCount += 1
				leftWeight += data[-1]
				rightWeight -= data[-1]
				leftClassifyCountDic[data[-2]] = leftClassifyCountDic[data[-2]] + data[-1]
				rightClassifyCountDic[data[-2]] = rightClassifyCountDic[data[-2]] - data[-1]
			else:
				break
		for index in range(lessCount):
			del(dataSet[0])
		leftEnt = self.calculateEntWithClassifyCountDic(leftClassifyCountDic,leftWeight)
		rightEnt = self.calculateEntWithClassifyCountDic(rightClassifyCountDic,rightWeight)
		return leftEnt,rightEnt,leftWeight,rightWeight

	# 连续型数据划分(二分法)
	def splitDataWithContinuouAttr(self,dataSet,specifyValue,index,splitAttr,delSpecifyData=True):
		leftList = [] ; leftWeight = float(0)
		rightList = [] ; rightWeight = float(0)
		lostList = []
		for data in dataSet:
			copyData = data[:]
			value = copyData[index]
			if delSpecifyData:
				del(copyData[index])
			if value != self.unknownMark:
				value = float(value)
				if value <= specifyValue:
					leftList.append(copyData)
					leftWeight += copyData[-1]
				else:
					rightList.append(copyData)
					rightWeight += copyData[-1]
			else:
				lostList.append(copyData)
		# 在构建决策树时，理论上totalWeight不会为0，\
		# （若为0，代表分割数据集dataSet的指定值全为未知量，若全为未知量，则按照信息增益原则，不会选该项作为划分项）,\
		# 但是在后剪枝过程中，totalWeight可能为0，这种情况非常少见，常见于数据集过少，暂未处理，重新运行即可
		totalWeight = leftWeight + rightWeight
		for lost in lostList:
			leftCopyLost = lost[:] ; rightCopyLost = lost[:]
			leftCopyLost[-1] *= leftWeight/totalWeight ; rightCopyLost[-1] *= rightWeight/totalWeight ; 
			leftList.append(leftCopyLost)
			rightList.append(rightCopyLost)
			pass
		splitDic = {}
		leftKey = str(splitAttr)+'<='+str(specifyValue)
		rightKey = str(splitAttr)+'>'+str(specifyValue)
		if len(leftList) > 0:
			splitDic[leftKey] = leftList
		else:
			splitDic[leftKey] = self.chooseTheMostItems(rightList)
		if len(rightList) > 0:
			splitDic[rightKey] = rightList
		else:
			splitDic[rightKey] = self.chooseTheMostItems(leftList)
		return splitDic
		pass

	# 计算信息熵
	def calculateEnt(self,dataSet):
		classifyCountDic = {}
		totalWeights = float(0)
		for data in dataSet:
			classifyCountDic[data[-2]] = classifyCountDic.get(data[-2],float(0)) + data[-1]
			totalWeights += data[-1]
		ent = self.calculateEntWithClassifyCountDic(classifyCountDic,totalWeights)
		return ent
	# 根据分类好的字典和总权重计算信息熵
	def calculateEntWithClassifyCountDic(self,classifyCountDic,totalWeights):
		ent = 0.0
		for key in classifyCountDic.keys():
			weight = classifyCountDic[key]
			if weight != 0:
				p = weight/totalWeights
				ent -= p*log2(p)
		return ent
		pass


	# 离散型属性可能取值{属性名：[可能取值1,可能取值2,可能取值3...]}
	def create_attributes_dicSet(self,dataMat):
		dic = {}
		length = len(self.attribute_types)
		for index in range(length):
			if self.attribute_types[index] == "discrete":
				sets = set(dataMat[:,index].T.tolist()[0])
				listSet = []
				for item in sets:
					# 删除缺失值标记
					if item != self.unknownMark:
						# self.debugPrint("item:",item,"unknownMark:",self.unknownMark)
						listSet.append(item)
				dic[self.attribute_names[index]] = listSet
				pass
			pass
		self.attributes_dicSet = dic
		pass

	def create_attributes_indexDic(self):
		index = 0
		for attr_name in self.attribute_names:
			self.predictAttrIndexDic[attr_name] = index
			index += 1
			pass
		pass

	def myIteritems(self,dic:dict):
		keys = dic.keys()
		values = dic.values()
		lst = [(key,value) for key , value in zip(keys,values)]
		return lst
	# 选择列表中最多的类型
	def chooseTheMostItems(self,classList):
		r = {}
		for item in classList:
			r[item[-2]] = r.get(item[-2],0) + item[-1]
		sortedR = sorted(self.myIteritems(r),key=lambda x:x[1],reverse=True)
		return sortedR[0][0]

	# 检测必要属性是否为空
	def checkPropoty(self):
		if self.checkEmpty(self.fileName):
			print("fileName为空")
			return False
		if self.checkEmpty(self.unknownMark):
			print("unknownMark为空")
			return False
		if self.checkEmpty(self.fileSplitStr):
			print("fileSplitStr为空")
			return False
		if self.checkEmpty(self.attribute_names):
			print("attribute_names为空")
			return False
		if self.checkEmpty(self.attribute_types):
			print("attribute_types为空")
			return False
		return True
		pass

	# 检测是否为空
	def checkEmpty(self,obj):
		if len(obj) == 0:
			return True
		return False
		pass


	def debugPrint(self,*args):
		if not self.canPrint:
			return
		printStr = ""
		for arg in args:
			printStr += (str(arg) + " ")
		print(printStr)


	def predict(self,fileName):
		if len(self.tree) == 0:
			print("no tree")

		predictDataSet = []
		trueResult = []
		fr = open(fileName)
		for line in fr.readlines():
			newLine = line.strip().split(",")
			attributes = [item.strip() for item in newLine]
			trueResult.append(attributes[-1].strip("."))
			del(attributes[-1])
			predictDataSet.append(attributes)
		errorNum = 0
		totalNum = len(predictDataSet)
		errorIndex = []
		for index in range(len(predictDataSet)):
			data = predictDataSet[index]
			if len(data) != len(self.attribute_names):
				print("第",index,"条数据，参数数量不符合")
				continue
				pass
			result = self.predictDataWithTree(data,self.tree)
			if result != trueResult[index]:
				print("index:",index,"trueResult:",trueResult[index],"result:",result)
				errorNum += 1
				errorIndex.append(index)
		e = errorNum/totalNum
		return e
	
	def predictDataWithTree(self,data,tree):
		data.append(float(1))
		predictList = self.getPredictListWithTree(data,tree)
		resultDic = {}
		for item in predictList:
			key = item[0]
			resultDic[key] = resultDic.get(key,float(0)) + item[1]
		radio = 0
		result = ""
		for key in resultDic:
			value = resultDic[key]
			if value >= radio:
				radio = value
				result = key
		return result

	def getPredictListWithTree(self,data,tree):
		copyData = data[:]
		key = list(tree.keys())[0]
		index = self.predictAttrIndexDic[key]
		attr_type = self.attribute_types[index]
		isContinuous = (attr_type == "continuous")
		value = copyData[index]
		itemList = []
		if value == self.unknownMark:
			pdic = tree[key][self.unknownMark]
			for key_dic in pdic.keys():
				copyData_ = copyData[:]
				copyData_[-1] *= float(pdic[key_dic])
				item = (tree[key][key_dic],copyData_)
				itemList.append(item)
				pass
		else:
			sub_keys = list(tree[key].keys())
			if isContinuous:
				sub_key = sub_keys[0]
				if "<=" in sub_key:
					attr_value = float(sub_key.split("<=")[-1])
				else:
					attr_value = float(sub_key.split(">")[-1])
				for key_item in sub_keys:
					if "<=" in key_item:
						if float(value) <= attr_value:
							obj = tree[key][key_item]
							break
					elif ">" in key_item:
						if float(value) > attr_value:
							obj = tree[key][key_item]
							break
			else:
				if value in sub_keys:
					obj = tree[key][value]
				else:
					obj = tree[key][self.resultTreeOtherSituationMark]
			itemList.append((obj,copyData))
		returnList = []
		for item in itemList:
			s_tree,p_data = item
			if self.isTree(s_tree):
				returnList.extend(self.getPredictListWithTree(p_data,s_tree))
			else:
				returnList.append([s_tree,p_data[-1]])
		return returnList

	def isTree(self,obj):
		return (type(obj).__name__=='dict')

	def isList(self,obj):
		return (type(obj).__name__=='list')
		pass



# 给定训练数据，设定相应参数，生成决策树
def testA(testP=0):
	adult = Tree()
	# 指定需要读取的训练数据文件名，应保证该文件与decisionTree.py在同一文件夹下
	adult.fileName = "adult.data.txt"
	# 指定训练数据使用的分割符号
	adult.fileSplitStr = ","
	# 指定训练数据各特征值名称
	adult.attribute_names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship',\
			    'race','sex','capital-gain','capital-loss','hours-per-week','native-country']
	# 指定训练数据各特征值类型：continuous表示该特征值为连续型；discrete表示该特征值为离散型
	adult.attribute_types = ['continuous','discrete','continuous','discrete','continuous','discrete','discrete','discrete',\
			    'discrete','discrete','continuous','continuous','continuous','discrete']
	# 指定训练数据中缺省值的表示符号
	adult.unknownMark = "?"
	# 指定得到的决策树中，若该节点特征值取值不全时，其它项的表示文字
	adult.resultTreeOtherSituationMark = "其它:"
	if testP > 0.5:
		print("用于测试的比例过高")
		return None
	# 指定用于测试的样本所占比例，默认为0，若为0，则不对树进行剪枝处理
	adult.testNumPercent = testP
	adult.tree = adult.trainTree()
	return adult
	pass
# 由于树剪枝随机抽取数据划入测试集，故训练多次取平均错误率，testP指定用于测试的样本所占比例，num指定训练次数
def getArrageError(testP=0,num=1):
	totalE = 0
	for i in range(num):
		adult = testA(testP)
		e = adult.predict("adult.test.txt")
		totalE += e
		pass
	return totalE/num
	pass

